"""Resilient launcher for BTC historical research.

Adds a verified Binance Public Data fallback for USD-M Futures kline
endpoints when hosted CI cannot access the Binance Futures REST API.
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

USER_AGENT = "BTC-Prediction-Research/6.4"
# All three USD-M kline sources used by the research engine.
FALLBACK_ENDPOINTS = {"klines", "markPriceKlines", "premiumIndexKlines"}
RETRYABLE_HTTP = (
    "HTTP Error 403", "HTTP Error 429", "HTTP Error 451",
    "HTTP Error 500", "HTTP Error 502", "HTTP Error 503", "HTTP Error 504",
)

# Keep an immutable reference before monkey-patching.
_ORIGINAL_REQ_JSON = hr.req_json


def _download_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def _verified_zip_rows(url: str, start_ms: int, end_ms: int):
    payload = _download_bytes(url)
    checksum_text = _download_bytes(url + ".CHECKSUM").decode("utf-8", errors="replace").strip()
    expected = checksum_text.split()[0].lower() if checksum_text else ""
    actual = hashlib.sha256(payload).hexdigest().lower()
    if expected and expected != actual:
        raise RuntimeError(f"Binance archive checksum mismatch: {url}")

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if not names:
            return []
        with zf.open(names[0]) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8", newline="")
            reader = csv.reader(text)
            rows = []
            for row in reader:
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
    """Return daily then monthly official Binance archive URLs."""
    base = "https://data.binance.vision/data/futures/um"
    d = day.isoformat()
    ym = day.strftime("%Y-%m")
    # Binance public-data naming convention is <symbol>-<interval>-<date>.zip
    daily = (
        f"{base}/daily/{endpoint}/{symbol}/{interval}/"
        f"{symbol}-{interval}-{d}.zip"
    )
    monthly = (
        f"{base}/monthly/{endpoint}/{symbol}/{interval}/"
        f"{symbol}-{interval}-{ym}.zip"
    )
    return daily, monthly


def _archive_fallback(url: str):
    parsed = urllib.parse.urlsplit(url)
    qs = urllib.parse.parse_qs(parsed.query)
    symbol = qs.get("symbol", [None])[0]
    interval = qs.get("interval", ["1m"])[0]
    start_ms = int(qs.get("startTime", [0])[0])
    end_ms = int(qs.get("endTime", [0])[0])
    if not symbol or not start_ms or not end_ms:
        raise RuntimeError("archive fallback could not parse symbol/time range")

    endpoint = parsed.path.split("/fapi/v1/")[-1]
    if endpoint not in FALLBACK_ENDPOINTS:
        raise RuntimeError("archive fallback only supports USD-M kline endpoints")

    start_day = datetime.fromtimestamp(start_ms / 1000, timezone.utc).date()
    end_day = datetime.fromtimestamp((end_ms - 1) / 1000, timezone.utc).date()
    day = start_day
    all_rows = []
    errors = []
    while day <= end_day:
        daily, monthly = _archive_urls(symbol, interval, endpoint, day)
        loaded = False
        for archive_url in (daily, monthly):
            try:
                all_rows.extend(_verified_zip_rows(archive_url, start_ms, end_ms))
                loaded = True
                break
            except urllib.error.HTTPError as exc:
                errors.append(f"{archive_url}: HTTP {exc.code}")
            except Exception as exc:
                errors.append(f"{archive_url}: {exc}")
        if not loaded:
            raise RuntimeError(
                f"no verified Binance archive available for {symbol} {endpoint} {day}: "
                + "; ".join(errors[-4:])
            )
        day += timedelta(days=1)

    # Deduplicate by open timestamp because daily/monthly boundaries can overlap.
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
                if rows:
                    return rows
                raise RuntimeError("verified Binance archive returned no rows")
            except Exception as archive_error:
                last = archive_error
                time.sleep(float(attempt + 1))
        raise RuntimeError(
            f"Binance Futures REST unavailable and verified archive fallback failed: {last}"
        ) from exc


hr.req_json = resilient_req_json

if __name__ == "__main__":
    hr.main()
