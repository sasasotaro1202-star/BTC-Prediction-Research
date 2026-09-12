"""Resilient launcher for BTC historical research.

The research engine normally uses Binance REST. Hosted CI can receive HTTP 451
from Binance Futures, so this launcher has a verified Binance Public Data
fallback.  The fallback also protects the run from the publication lag of
Binance daily archives by never requesting unpublished UTC time.
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

import historical_research as hr

USER_AGENT = "BTC-Prediction-Research/6.5"
FALLBACK_ENDPOINTS = {"klines", "markPriceKlines", "premiumIndexKlines"}
RETRYABLE_HTTP = (
    "HTTP Error 403", "HTTP Error 429", "HTTP Error 451",
    "HTTP Error 500", "HTTP Error 502", "HTTP Error 503", "HTTP Error 504",
)

# Daily Binance Public Data is published with a delay. Keep a conservative
# three-day safety margin so CI never depends on a not-yet-published archive.
ARCHIVE_SAFETY_DAYS = 3
_ORIGINAL_REQ_JSON = hr.req_json


def _download_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.read()


def _verified_zip_rows(url: str, start_ms: int, end_ms: int):
    payload = _download_bytes(url)
    checksum_text = _download_bytes(url + ".CHECKSUM").decode("utf-8", errors="replace").strip()
    expected = checksum_text.split()[0].lower() if checksum_text else ""
    actual = hashlib.sha256(payload).hexdigest().lower()
    if not expected or expected != actual:
        raise RuntimeError(f"Binance archive checksum mismatch: {url}")

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if not names:
            return []
        with zf.open(names[0]) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8", newline="")
            rows = []
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
    # These names are taken from Binance's official download scripts.
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

    # Never request a daily archive inside Binance's publication window.
    end_ms = min(requested_end_ms, _safe_archive_end_ms())
    if start_ms >= end_ms:
        return []

    start_day = datetime.fromtimestamp(start_ms / 1000, timezone.utc).date()
    end_day = datetime.fromtimestamp((end_ms - 1) / 1000, timezone.utc).date()
    day = start_day
    all_rows = []
    while day <= end_day:
        daily, monthly = _archive_urls(symbol, interval, endpoint, day)
        loaded = False
        errors = []
        # Prefer daily because Binance documents daily data as the freshest
        # archive and monthly files can be regenerated less frequently.
        for archive_url in (daily, monthly):
            try:
                rows = _verified_zip_rows(archive_url, start_ms, end_ms)
                all_rows.extend(rows)
                loaded = True
                break
            except urllib.error.HTTPError as exc:
                errors.append(f"HTTP {exc.code}")
            except Exception as exc:
                errors.append(str(exc))
        if not loaded:
            # If a day is inside the safe historical window but no official
            # archive exists, fail loudly rather than silently fabricate data.
            raise RuntimeError(
                f"no verified Binance archive for {symbol} {endpoint} {day}: "
                + "; ".join(errors)
            )
        day += timedelta(days=1)

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
                rows = _archive_fallback(url)
                # Empty means the requested chunk is newer than the safe
                # publication boundary. The caller will stop naturally.
                return rows
            except Exception as archive_error:
                last = archive_error
                time.sleep(float(attempt + 1))
        raise RuntimeError(
            f"Binance Futures REST unavailable and verified archive fallback failed: {last}"
        ) from exc


hr.req_json = resilient_req_json

if __name__ == "__main__":
    hr.main()
