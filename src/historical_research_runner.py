"""Resilient launcher for BTC historical research.

CI can receive HTTP 451 from Binance Futures REST.  This launcher therefore
uses Binance Public Data as a verified fallback.  It also prevents requests
for unpublished UTC days and caches verified archives so a retry never
re-downloads the same file unnecessarily.
"""
from __future__ import annotations

import csv
import hashlib
import io
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import historical_research as hr

USER_AGENT = "BTC-Prediction-Research/7.0"
FALLBACK_ENDPOINTS = {"klines", "markPriceKlines", "premiumIndexKlines"}
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
        payload = cache.read_bytes()
    else:
        payload = _download_bytes(url)
        checksum_url = url + ".CHECKSUM"
        checksum_text = _download_bytes(checksum_url).decode("utf-8", errors="replace").strip()
        expected = checksum_text.split()[0].lower() if checksum_text else ""
        actual = hashlib.sha256(payload).hexdigest().lower()
        if not expected or expected != actual:
            raise RuntimeError(f"Binance archive checksum mismatch: {url}")
        cache.write_bytes(payload)
        return payload

    # Cached bytes were already checksum-verified when stored.  Verify again
    # only if the checksum file is cheaply available; otherwise the cache is
    # treated as immutable content created by this process.
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
    # Binance's official public-data naming convention.
    daily = f"{base}/daily/{endpoint}/{symbol}/{interval}/{symbol}-{interval}-{d}.zip"
    monthly = f"{base}/monthly/{endpoint}/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
    return daily, monthly


def _safe_archive_end_ms() -> int:
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    safe = now - timedelta(days=ARCHIVE_SAFETY_DAYS)
    return int(safe.timestamp() * 1000)


def _archive_fallback(url: str):
    parsed = urllib.parse.urlsplit(url)
    qs = urllib.parse.parse_qs(parsed.query)
    symbol = qs.get("symbol", [None])[0]
    interval = qs.get("interval", ["1m"])[0]
    start_ms = int(qs.get("startTime", [0])[0])
    requested_end_ms = int(qs.get("endTime", [0])[0])
    if not symbol or not start_ms or not requested_end_ms:
        raise RuntimeError("archive fallback could not parse symbol/time range")

    endpoint = parsed.path.split("/fapi/v1/")[-1]
    if endpoint not in FALLBACK_ENDPOINTS:
        raise RuntimeError("archive fallback only supports USD-M kline endpoints")

    # Never request an unpublished day.  The research engine itself also uses
    # this boundary; this is a second defensive layer.
    end_ms = min(requested_end_ms, _safe_archive_end_ms())
    if start_ms >= end_ms:
        return []

    start_day = datetime.fromtimestamp(start_ms / 1000, timezone.utc).date()
    end_day = datetime.fromtimestamp((end_ms - 1) / 1000, timezone.utc).date()

    # Group requests by month.  Monthly archives are preferred for completed
    # months because one verified download replaces up to 31 daily downloads.
    # The current month uses daily archives because the monthly file is not
    # published until the following month.
    current_month = datetime.now(timezone.utc).date().replace(day=1)
    all_rows = []
    day = start_day
    monthly_days = {}
    daily_days = []
    while day <= end_day:
        month_start = day.replace(day=1)
        month_end = (month_start + timedelta(days=32)).replace(day=1)
        if month_end <= current_month:
            monthly_days.setdefault(month_start, []).append(day)
        else:
            daily_days.append(day)
        day += timedelta(days=1)

    def collect_month(month_start, days):
        _, monthly = _archive_urls(symbol, interval, endpoint, month_start)
        month_end = (month_start + timedelta(days=32)).replace(day=1)
        a = max(start_ms, int(datetime.combine(month_start, datetime.min.time(), tzinfo=timezone.utc).timestamp()*1000))
        b = min(end_ms, int(datetime.combine(month_end, datetime.min.time(), tzinfo=timezone.utc).timestamp()*1000))
        return _verified_zip_rows(monthly, a, b)

    # Monthly first, with a daily fallback only if a monthly archive is
    # temporarily unavailable.  This keeps the normal path fast and robust.
    for month_start, days in monthly_days.items():
        try:
            all_rows.extend(collect_month(month_start, days))
            continue
        except Exception as monthly_error:
            for day in days:
                daily, _ = _archive_urls(symbol, interval, endpoint, day)
                try:
                    all_rows.extend(_verified_zip_rows(daily, start_ms, end_ms))
                except Exception as daily_error:
                    raise RuntimeError(
                        f"no verified Binance archive for {symbol} {endpoint} {day}: "
                        f"monthly={monthly_error}; daily={daily_error}"
                    )

    for day in daily_days:
        daily, _ = _archive_urls(symbol, interval, endpoint, day)
        try:
            all_rows.extend(_verified_zip_rows(daily, start_ms, end_ms))
        except Exception as exc:
            raise RuntimeError(f"no verified Binance daily archive for {symbol} {endpoint} {day}: {exc}")

    dedup = {int(r[0]): r for r in all_rows}
    return [dedup[k] for k in sorted(dedup)]


def resilient_req_json(url: str, timeout=30, retries=5):
    try:
        return _ORIGINAL_REQ_JSON(url, timeout=timeout, retries=retries)
    except RuntimeError as exc:
        message = str(exc)
        if not any(code in message for code in RETRYABLE_HTTP):
            raise
        if not any(f"/fapi/v1/{endpoint}?" in url for endpoint in FALLBACK_ENDPOINTS):
            raise
        last = None
        for attempt in range(3):
            try:
                return _archive_fallback(url)
            except Exception as archive_error:
                last = archive_error
                time.sleep(float(attempt + 1))
        raise RuntimeError(
            f"Binance Futures REST unavailable and verified archive fallback failed: {last}"
        ) from exc


hr.req_json = resilient_req_json

if __name__ == "__main__":
    hr.main()
