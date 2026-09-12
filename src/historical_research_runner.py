"""Resilient launcher for BTC historical research.

Core rule: the OOS research must not fail merely because an optional Binance
feed is unavailable from GitHub Actions. Core futures klines are mandatory;
mark/premium/funding/OI and spot are optional enrichments with safe fallbacks.
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

USER_AGENT = "BTC-Prediction-Research/9.0"
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
_ORIGINAL_LOAD_MARKET = hr.load_market


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
    checksum = _download_bytes(url + ".CHECKSUM").decode("utf-8", errors="replace").strip()
    expected = checksum.split()[0].lower() if checksum else ""
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
    rows = []
    day = start_day
    months = {}
    while day <= end_day:
        months.setdefault(day.replace(day=1), []).append(day)
        day += timedelta(days=1)

    for month_start, days in months.items():
        month_end = (month_start + timedelta(days=32)).replace(day=1)
        a = max(start_ms, int(datetime.combine(month_start, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000))
        b = min(end_ms, int(datetime.combine(month_end, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000))

        if month_end <= current_month:
            _, monthly = _archive_urls(symbol, interval, endpoint, month_start)
            try:
                rows.extend(_verified_zip_rows(monthly, a, b))
                continue
            except Exception:
                pass

        for d in days:
            daily, _ = _archive_urls(symbol, interval, endpoint, d)
            da = max(a, int(datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000))
            db = min(b, int(datetime.combine(d + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000))
            try:
                rows.extend(_verified_zip_rows(daily, da, db))
            except Exception as exc:
                if optional:
                    continue
                raise RuntimeError(f"no verified Binance core archive for {symbol} {endpoint} {d}: {exc}")

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


def _safe_fetch(fn, label):
    try:
        return fn()
    except Exception as exc:
        print(f"[WARN] optional feed unavailable: {label}: {exc}")
        return []


def resilient_load_market(start, end):
    """Load the same panel as historical_research.load_market, but never let
    optional feeds abort the complete research run.
    """
    jobs = {}
    for name, sym in hr.SYMS.items():
        jobs[f"{name}_fut"] = lambda sy=sym: hr.fetch_klines_range(sy, start, end, "/fapi/v1/klines", f"{name}_fut", 1500)
        jobs[f"{name}_mark"] = lambda sy=sym: hr.fetch_klines_range(sy, start, end, "/fapi/v1/markPriceKlines", f"{name}_mark", 1500)
        jobs[f"{name}_premium"] = lambda sy=sym: hr.fetch_klines_range(sy, start, end, "/fapi/v1/premiumIndexKlines", f"{name}_premium", 1500)
    jobs["btc_spot"] = lambda: hr.fetch_klines_range("BTCUSDT", start, end, "/api/v3/klines", "btc_spot", 1000)
    jobs["funding"] = lambda: hr.fetch_funding("BTCUSDT", start, end)
    jobs["oi"] = lambda: hr.fetch_oi("BTCUSDT", start, end)

    required = {"btc_fut", "eth_fut", "sol_fut"}
    out = {}
    with __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=8) as ex:
        future_map = {ex.submit(fn): name for name, fn in jobs.items()}
        for future in __import__("concurrent.futures").futures.as_completed(future_map):
            name = future_map[future]
            try:
                out[name] = future.result()
            except Exception as exc:
                if name in required:
                    raise
                print(f"[WARN] optional feed skipped: {name}: {exc}")
                out[name] = []

    # Spot is an enrichment. If GitHub Actions cannot reach api.binance.com,
    # use the corresponding futures close as a transparent zero-basis proxy.
    if not out.get("btc_spot"):
        out["btc_spot"] = out.get("btc_fut", [])
        print("[WARN] BTC spot feed unavailable; using BTC futures close as spot proxy")

    # Funding/OI are optional features. The research engine already handles
    # missing arrays by using neutral/default values.
    out.setdefault("funding", [])
    out.setdefault("oi", [])
    return out


hr.req_json = resilient_req_json
hr.load_market = resilient_load_market

if __name__ == "__main__":
    hr.main()
