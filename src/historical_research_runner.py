"""Resilient launcher for BTC historical research.

The research logic remains in historical_research.py. This launcher only adds
an exact historical-data fallback when Binance's futures REST endpoint returns
HTTP 451 from a hosted CI runner. Historical mark/premium klines are read from
Binance's public data archive, while the primary REST path remains unchanged.
"""
from __future__ import annotations

import csv
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone

import historical_research as hr


def _archive_rows(url: str, start_ms: int, end_ms: int):
    req = urllib.request.Request(url, headers={"User-Agent": "BTC-Prediction-Research/6.1"})
    with urllib.request.urlopen(req, timeout=45) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = zf.namelist()
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
    if endpoint not in {"markPriceKlines", "premiumIndexKlines"}:
        raise RuntimeError("archive fallback only supports mark/premium klines")

    day = datetime.fromtimestamp(start_ms / 1000, timezone.utc).date()
    kind = "markPriceKlines" if endpoint == "markPriceKlines" else "premiumIndexKlines"
    archive_url = (
        f"https://data.binance.vision/data/futures/um/daily/{kind}/"
        f"{symbol}/1m/{symbol}-{kind}-1m-{day.isoformat()}.zip"
    )
    # Binance archive naming has historically omitted the interval token in
    # some datasets; try the documented daily filename first, then the
    # alternate filename without the interval token.
    try:
        return _archive_rows(archive_url, start_ms, end_ms)
    except urllib.error.HTTPError as first_error:
        alternate = (
            f"https://data.binance.vision/data/futures/um/daily/{kind}/"
            f"{symbol}/1m/{symbol}-{kind}-{day.isoformat()}.zip"
        )
        try:
            return _archive_rows(alternate, start_ms, end_ms)
        except Exception:
            raise first_error


def resilient_req_json(url: str, timeout=30, retries=5):
    try:
        return hr.req_json(url, timeout=timeout, retries=retries)
    except RuntimeError as exc:
        message = str(exc)
        if "HTTP Error 451" not in message:
            raise
        if "/fapi/v1/markPriceKlines?" not in url and "/fapi/v1/premiumIndexKlines?" not in url:
            raise
        last = None
        for attempt in range(3):
            try:
                return _archive_fallback(url)
            except Exception as archive_error:
                last = archive_error
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"Binance REST blocked with 451 and archive fallback failed: {last}") from exc


hr.req_json = resilient_req_json

if __name__ == "__main__":
    hr.main()
