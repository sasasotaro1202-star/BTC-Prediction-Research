"""Resilient launcher for BTC historical research.

The research logic remains in historical_research.py. This launcher adds a
verified Binance Public Data fallback when hosted CI cannot access selected
USD-M Futures REST endpoints.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone

import historical_research as hr

USER_AGENT = "BTC-Prediction-Research/6.2"
FALLBACK_ENDPOINTS = {"markPriceKlines", "premiumIndexKlines"}
RETRYABLE_HTTP = ("HTTP Error 403", "HTTP Error 429", "HTTP Error 451", "HTTP Error 500", "HTTP Error 502", "HTTP Error 503", "HTTP Error 504")


def _download_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=45) as response:
        return response.read()


def _verified_zip_rows(url: str, start_ms: int, end_ms: int):
    payload = _download_bytes(url)

    # Binance publishes a matching .CHECKSUM file beside each archive.
    checksum_url = url + ".CHECKSUM"
    checksum_text = _download_bytes(checksum_url).decode("utf-8", errors="replace").strip()
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


def _archive_fallback(url: str):
    parsed = urllib.parse.urlsplit(url)
    qs = urllib.parse.parse_qs(parsed.query)
    symbol = qs.get("symbol", [None])[0]
    start_ms = int(qs.get("startTime", [0])[0])
    end_ms = int(qs.get("endTime", [0])[0])
    if not symbol or not start_ms:
        raise RuntimeError("archive fallback could not parse symbol/startTime")

    endpoint = parsed.path.split("/fapi/v1/")[-1]
    if endpoint not in FALLBACK_ENDPOINTS:
        raise RuntimeError("archive fallback only supports mark/premium klines")

    day = datetime.fromtimestamp(start_ms / 1000, timezone.utc).date()
    kind = endpoint

    # Official Binance Futures public-data layout:
    # futures/um/daily/<kind>/<symbol>/<interval>/<symbol>-<interval>-<date>.zip
    # (not <symbol>-<kind>-<interval>-<date>.zip).
    archive_url = (
        f"https://data.binance.vision/data/futures/um/daily/{kind}/"
        f"{symbol}/1m/{symbol}-1m-{day.isoformat()}.zip"
    )
    return _verified_zip_rows(archive_url, start_ms, end_ms)


def resilient_req_json(url: str, timeout=30, retries=5):
    try:
        return hr.req_json(url, timeout=timeout, retries=retries)
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
