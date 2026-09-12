"""Resilient launcher for BTC historical research.

Design goals:
- Never depend on Binance Futures REST availability in hosted CI.
- Use Binance Public Data as the authoritative historical fallback.
- Discover archive objects from the official S3 index before downloading,
  rather than guessing filenames when an archive family changes.
- Verify every downloaded ZIP with Binance's published SHA-256 checksum.
- Treat mark/premium feeds as optional enrichments; core futures klines remain
  mandatory because they define the research sample.
- Avoid repeated archive parsing with in-process caches.
"""
from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import io
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

import historical_research as hr

USER_AGENT = "BTC-Prediction-Research/10.0"
CORE_ENDPOINT = "klines"
OPTIONAL_ENDPOINTS = {"markPriceKlines", "premiumIndexKlines"}
FALLBACK_ENDPOINTS = {CORE_ENDPOINT, *OPTIONAL_ENDPOINTS}
RETRYABLE_HTTP = (
    "HTTP Error 403", "HTTP Error 429", "HTTP Error 451",
    "HTTP Error 500", "HTTP Error 502", "HTTP Error 503", "HTTP Error 504",
)
ARCHIVE_SAFETY_DAYS = 3
ARCHIVE_BASE = "https://data.binance.vision"
ARCHIVE_S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
ARCHIVE_CACHE = Path("data/historical_research/archive_cache")
ARCHIVE_CACHE.mkdir(parents=True, exist_ok=True)

_ORIGINAL_REQ_JSON = hr.req_json

# Cache archive discovery and parsed ZIP payloads for the whole process.
_OBJECT_CACHE: dict[str, str] = {}
_ZIP_CACHE: dict[str, bytes] = {}


def _download_bytes(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def _archive_candidates(symbol: str, interval: str, endpoint: str, day):
    d = day.isoformat()
    ym = day.strftime("%Y-%m")
    root = f"data/futures/um"
    return [
        f"{root}/daily/{endpoint}/{symbol}/{interval}/{symbol}-{interval}-{d}.zip",
        f"{root}/monthly/{endpoint}/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip",
    ]


def _s3_find_key(prefix: str, wanted_names: set[str]) -> str | None:
    """Ask Binance's public S3 index for the exact object name.

    This avoids hard-coding archive-family details beyond the documented
    prefix. If the listing service is unavailable, callers fall back to the
    documented URL convention.
    """
    cache_key = "S3:" + prefix
    if cache_key in _OBJECT_CACHE:
        return _OBJECT_CACHE[cache_key] or None

    query = urllib.parse.urlencode({
        "list-type": "2",
        "prefix": prefix,
        "max-keys": "1000",
    })
    url = f"{ARCHIVE_S3}?{query}"
    try:
        xml_bytes = _download_bytes(url, timeout=45)
        root = ET.fromstring(xml_bytes)
        for elem in root.iter():
            if elem.tag.endswith("}Key") or elem.tag == "Key":
                key = (elem.text or "").strip()
                if key in wanted_names:
                    _OBJECT_CACHE[cache_key] = key
                    return key
    except Exception:
        pass

    _OBJECT_CACHE[cache_key] = ""
    return None


def _resolve_archive_url(symbol: str, interval: str, endpoint: str, day, monthly: bool) -> str:
    candidates = _archive_candidates(symbol, interval, endpoint, day)
    target = candidates[1] if monthly else candidates[0]
    prefix = target.rsplit("/", 1)[0] + "/"
    wanted = {target}
    key = _s3_find_key(prefix, wanted)
    if key:
        return f"{ARCHIVE_BASE}/{key}"
    return f"{ARCHIVE_BASE}/{target}"


def _cache_file(url: str) -> Path:
    return ARCHIVE_CACHE / (hashlib.sha256(url.encode()).hexdigest() + ".zip")


def _verified_zip_payload(url: str) -> bytes:
    if url in _ZIP_CACHE:
        return _ZIP_CACHE[url]

    cache = _cache_file(url)
    if cache.exists():
        payload = cache.read_bytes()
    else:
        payload = _download_bytes(url)

    checksum_text = _download_bytes(url + ".CHECKSUM").decode("utf-8", errors="replace").strip()
    expected = checksum_text.split()[0].lower() if checksum_text else ""
    actual = hashlib.sha256(payload).hexdigest().lower()
    if not expected or expected != actual:
        raise RuntimeError(f"Binance archive checksum mismatch: {url}")

    if not cache.exists():
        cache.write_bytes(payload)
    _ZIP_CACHE[url] = payload
    return payload


def _verified_zip_rows(url: str, start_ms: int, end_ms: int):
    payload = _verified_zip_payload(url)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if not names:
            return []
        rows = []
        with zf.open(names[0]) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8", newline="")
            for row in csv.reader(text):
                if not row:
                    continue
                try:
                    ts = int(float(row[0]))
                except (ValueError, TypeError):
                    continue
                # Ignore an optional CSV header and enforce half-open range.
                if start_ms <= ts < end_ms:
                    rows.append(row)
        return rows


def _safe_archive_end_ms() -> int:
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    return int((now - timedelta(days=ARCHIVE_SAFETY_DAYS)).timestamp() * 1000)


def _archive_fallback(url: str, optional: bool = False):
    parsed = urllib.parse.urlsplit(url)
    qs = urllib.parse.parse_qs(parsed.query)
    symbol = qs.get("symbol", [None])[0]
    interval = qs.get("interval", ["1m"])[0]
    start_ms = int(qs.get("startTime", [0])[0])
    requested_end_ms = int(qs.get("endTime", [0])[0])
    endpoint = parsed.path.split("/fapi/v1/")[-1]

    if not symbol or not start_ms or not requested_end_ms:
        if optional:
            return []
        raise RuntimeError("archive fallback could not parse symbol/time range")
    if endpoint not in FALLBACK_ENDPOINTS:
        if optional:
            return []
        raise RuntimeError(f"unsupported archive endpoint: {endpoint}")

    # Never request an archive for a date that Binance cannot have published
    # yet. This also prevents the daily-file 404 loop seen in CI.
    end_ms = min(requested_end_ms, _safe_archive_end_ms())
    if start_ms >= end_ms:
        return []

    start_day = datetime.fromtimestamp(start_ms / 1000, timezone.utc).date()
    end_day = datetime.fromtimestamp((end_ms - 1) / 1000, timezone.utc).date()
    current_month = datetime.now(timezone.utc).date().replace(day=1)
    rows = []
    day = start_day
    months: dict = {}
    while day <= end_day:
        months.setdefault(day.replace(day=1), []).append(day)
        day += timedelta(days=1)

    for month_start, days in months.items():
        month_end = (month_start + timedelta(days=32)).replace(day=1)
        a = max(
            start_ms,
            int(datetime.combine(month_start, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000),
        )
        b = min(
            end_ms,
            int(datetime.combine(month_end, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000),
        )

        # Completed months: monthly archive first. Current month: daily files.
        if month_end <= current_month:
            monthly_url = _resolve_archive_url(symbol, interval, endpoint, month_start, monthly=True)
            try:
                rows.extend(_verified_zip_rows(monthly_url, a, b))
                continue
            except Exception:
                pass

        for d in days:
            daily_url = _resolve_archive_url(symbol, interval, endpoint, d, monthly=False)
            da = max(a, int(datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000))
            db = min(b, int(datetime.combine(d + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000))
            try:
                rows.extend(_verified_zip_rows(daily_url, da, db))
            except Exception as exc:
                if optional:
                    continue
                raise RuntimeError(
                    f"no verified Binance core archive for {symbol} {endpoint} {d}: {exc}"
                )

    # Binance's official archives have documented missing/duplicate timestamps
    # in some historical mark/index datasets. Deduplicate by timestamp but do
    # not invent/interpolate missing rows.
    dedup = {int(r[0]): r for r in rows}
    return [dedup[k] for k in sorted(dedup)]


def resilient_req_json(url: str, timeout=30, retries=5):
    try:
        return _ORIGINAL_REQ_JSON(url, timeout=timeout, retries=retries)
    except RuntimeError as exc:
        message = str(exc)
        if not any(code in message for code in RETRYABLE_HTTP):
            raise
        if "/fapi/v1/" not in url:
            raise
        endpoint = url.split("/fapi/v1/", 1)[1].split("?", 1)[0]
        if endpoint == CORE_ENDPOINT:
            return _archive_fallback(url, optional=False)
        if endpoint in OPTIONAL_ENDPOINTS:
            return _archive_fallback(url, optional=True)
        raise


def resilient_load_market(start, end):
    """Load the research panel with mandatory/optional feed isolation."""
    jobs = {}
    for name, sym in hr.SYMS.items():
        jobs[f"{name}_fut"] = lambda sy=sym, n=name: hr.fetch_klines_range(
            sy, start, end, "/fapi/v1/klines", f"{n}_fut", 1500
        )
        jobs[f"{name}_mark"] = lambda sy=sym, n=name: hr.fetch_klines_range(
            sy, start, end, "/fapi/v1/markPriceKlines", f"{n}_mark", 1500
        )
        jobs[f"{name}_premium"] = lambda sy=sym, n=name: hr.fetch_klines_range(
            sy, start, end, "/fapi/v1/premiumIndexKlines", f"{n}_premium", 1500
        )
    jobs["btc_spot"] = lambda: hr.fetch_klines_range(
        "BTCUSDT", start, end, "/api/v3/klines", "btc_spot", 1000
    )
    jobs["funding"] = lambda: hr.fetch_funding("BTCUSDT", start, end)
    jobs["oi"] = lambda: hr.fetch_oi("BTCUSDT", start, end)

    required = {"btc_fut", "eth_fut", "sol_fut"}
    out = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        future_map = {ex.submit(fn): name for name, fn in jobs.items()}
        for future in concurrent.futures.as_completed(future_map):
            name = future_map[future]
            try:
                out[name] = future.result()
            except Exception as exc:
                if name in required:
                    raise
                print(f"[WARN] optional feed skipped: {name}: {exc}")
                out[name] = []

    if not out.get("btc_spot"):
        out["btc_spot"] = out.get("btc_fut", [])
        print("[WARN] BTC spot feed unavailable; using BTC futures close as spot proxy")
    out.setdefault("funding", [])
    out.setdefault("oi", [])
    return out


hr.req_json = resilient_req_json
hr.load_market = resilient_load_market

if __name__ == "__main__":
    hr.main()
