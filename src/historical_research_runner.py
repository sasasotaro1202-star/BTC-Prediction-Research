"""Resilient launcher for BTC historical research.

Design goals:
- Never recurse into the wrapped request function.
- Use Binance REST when available.
- Use verified Binance Public Data archives for historical USD-M klines when REST is blocked.
- Treat mark/premium/funding/OI as optional enrichments: their temporary unavailability
  must never destroy the core OOS dataset.
- Never request an unpublished current UTC day from the archive.
"""
from __future__ import annotations

import csv
import hashlib
import io
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import historical_research as hr

USER_AGENT = "BTC-Prediction-Research/8.0"
CORE_ENDPOINT = "klines"
OPTIONAL_ENDPOINTS = {"markPriceKlines", "premiumIndexKlines"}
RETRYABLE_HTTP = (
    "HTTP Error 403", "HTTP Error 429", "HTTP Error 451",
    "HTTP Error 500", "HTTP Error 502", "HTTP Error 503", "HTTP Error 504",
)
ARCHIVE_SAFETY_DAYS = 3
ARCHIVE_CACHE = Path("data/historical_research/archive_cache")
ARCHIVE_CACHE.mkdir(parents=True, exist_ok=True)
_ORIGINAL_REQ_JSON = hr.req_json


def _download_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.read()


def _cache_file(url: str) -> Path:
    return ARCHIVE_CACHE / (hashlib.sha256(url.encode()).hexdigest() + ".zip")


def _verified_zip_payload(url: str) -> bytes:
    cache = _cache_file(url)
    if cache.exists():
        return cache.read_bytes()
    payload = _download_bytes(url)
    checksum_text = _download_bytes(url + ".CHECKSUM").decode("utf-8", errors="replace").strip()
    expected = checksum_text.split()[0].lower() if checksum_text else ""
    actual = hashlib.sha256(payload).hexdigest().lower()
    if not expected or expected != actual:
        raise RuntimeError(f"Binance archive checksum mismatch: {url}")
    cache.write_bytes(payload)
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
                if start_ms <= ts < end_ms:
                    rows.append(row)
        return rows


def _archive_urls(symbol: str, interval: str, endpoint: str, day):
    base = "https://data.binance.vision/data/futures/um"
    d = day.isoformat()
    ym = day.strftime("%Y-%m")
    daily = f"{base}/daily/{endpoint}/{symbol}/{interval}/{symbol}-{interval}-{d}.zip"
    monthly = f"{base}/monthly/{endpoint}/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
    return daily, monthly


def _safe_archive_end_ms() -> int:
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    safe = now - timedelta(days=ARCHIVE_SAFETY_DAYS)
    return int(safe.timestamp() * 1000)


def _archive_fallback(url: str, *, optional: bool = False):
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
    if endpoint != CORE_ENDPOINT and endpoint not in OPTIONAL_ENDPOINTS:
        if optional:
            return []
        raise RuntimeError(f"unsupported archive endpoint: {endpoint}")

    end_ms = min(requested_end_ms, _safe_archive_end_ms())
    if start_ms >= end_ms:
        return []

    start_day = datetime.fromtimestamp(start_ms / 1000, timezone.utc).date()
    end_day = datetime.fromtimestamp((end_ms - 1) / 1000, timezone.utc).date()
    current_month = datetime.now(timezone.utc).date().replace(day=1)

    all_rows = []
    day = start_day
    months = {}
    while day <= end_day:
        month_start = day.replace(day=1)
        month_end = (month_start + timedelta(days=32)).replace(day=1)
        months.setdefault(month_start, []).append(day)
        day += timedelta(days=1)

    for month_start, days in months.items():
        month_end = (month_start + timedelta(days=32)).replace(day=1)
        a = max(start_ms, int(datetime.combine(month_start, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000))
        b = min(end_ms, int(datetime.combine(month_end, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000))

        # Completed months: monthly archive is much faster.
        monthly_ok = month_end <= current_month
        if monthly_ok:
            _, monthly = _archive_urls(symbol, interval, endpoint, month_start)
            try:
                all_rows.extend(_verified_zip_rows(monthly, a, b))
                continue
            except Exception:
                pass

        # Current month or missing monthly archive: use daily archives.
        for d in days:
            daily, _ = _archive_urls(symbol, interval, endpoint, d)
            try:
                da = max(a, int(datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000))
                db = min(b, int(datetime.combine(d + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000))
                all_rows.extend(_verified_zip_rows(daily, da, db))
            except Exception as exc:
                if optional:
                    # Optional enrichments may be absent from public archives.
                    # Do not make the entire research run fail because of them.
                    continue
                raise RuntimeError(f"no verified Binance core archive for {symbol} {endpoint} {d}: {exc}")

    dedup = {int(r[0]): r for r in all_rows}
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


# Only replace the original request function after keeping a permanent reference
# to it above. This prevents the recursion seen in the previous CI run.
hr.req_json = resilient_req_json

if __name__ == "__main__":
    hr.main()
