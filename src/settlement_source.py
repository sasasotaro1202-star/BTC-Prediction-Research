from __future__ import annotations

import csv
import json
import io
import math
import zipfile
from datetime import datetime, timezone
from functools import lru_cache
from urllib.request import Request, urlopen
from urllib.parse import urlencode

from market_data import _binance, BINANCE_WS_CACHE
from binance_ws import load_cache as load_binance_ws_cache


def preferred_source_from_scenario(scenario: dict) -> str:
    """Return the fixed production benchmark source.

    Production predictions are now fail-closed and therefore only originate
    from Binance BTCUSDT USD-M futures. Historical/fallback venues must not
    silently redefine the target after the prediction was made.
    """
    quality = scenario.get('data_quality') or {}
    if quality.get('price_feature_fallback') not in (None, '', 'none'):
        raise ValueError('settlement_source_invalid_fallback_prediction')
    if quality.get('binance_futures') != 'ok':
        raise ValueError('settlement_source_missing_binance_futures')
    return 'binance_futures'


def _target_from_rows(rows, start_ms: int) -> float | None:
    for row in rows:
        if int(row[0]) == start_ms:
            return float(row[4])
    return None


def target_close_preferred(target_iso: str, preferred_source: str) -> tuple[float | None, str]:
    if preferred_source == 'coinbase_exchange':
        return _target_coinbase_exchange(target_iso)
    if preferred_source == 'bybit_linear':
        return _target_bybit_linear(target_iso)
    if preferred_source != 'binance_futures':
        raise ValueError('unsupported_settlement_source')
    target = datetime.fromisoformat(target_iso.replace('Z', '+00:00'))
    ts = int(target.timestamp() * 1000)
    start = ts - 60_000
    # Prefer the dedicated closed-kline WebSocket cache because the live
    # predictor may already be using the same Binance USD-M product through WS
    # when Binance REST is geo-blocked/rate-limited on hosted runners.
    # This remains the identical production benchmark venue/product.
    try:
        cached = load_binance_ws_cache(BINANCE_WS_CACHE, 240)
        price = _target_from_ws_cache(cached, start)
        if price is not None:
            return price, 'binance_websocket_cache'
    except Exception:
        pass

    archive_price = _target_binance_daily_archive(start)
    if archive_price is not None:
        return archive_price, 'binance_daily_archive'

    try:
        price = _target_binance(start, ts)
    except Exception:
        return None, 'unavailable'
    return (price, 'binance') if price is not None else (None, 'unavailable')

def _target_from_ws_cache(rows, start_ms: int) -> float | None:
    """Read the exact closed Binance Futures candle from the local WS cache."""
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            open_ms = int(row.get('open_time_ms'))
            close_price = float(row.get('close'))
            closed = bool(row.get('closed', True))
            if closed and open_ms == int(start_ms) and close_price > 0:
                return close_price
        except (TypeError, ValueError):
            continue
    return None


def _target_binance(start: int, end: int) -> float | None:
    rows = _binance('fapi.binance.com/fapi/v1/klines', {
        'symbol': 'BTCUSDT', 'interval': '1m', 'startTime': start, 'endTime': end, 'limit': 2,
    })
    return _target_from_rows(rows, start)


@lru_cache(maxsize=8)
def _daily_archive_rows(date_text: str) -> dict[int, float]:
    """Load one Binance USD-M Futures 1m daily archive into an in-memory index."""
    name = f"BTCUSDT-1m-{date_text}.zip"
    url = f"https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1m/{name}"
    req = Request(url, headers={"User-Agent": "BTC-Prediction-Research/9.0"})
    rows: dict[int, float] = {}
    with urlopen(req, timeout=30) as response:
        raw = response.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"daily archive contains no csv: {name}")
        with zf.open(csv_names[0]) as fh:
            for row in csv.reader(io.TextIOWrapper(fh, encoding="utf-8")):
                if not row or not str(row[0]).isdigit() or len(row) < 6:
                    continue
                try:
                    open_ms = int(row[0])
                    close = float(row[4])
                except (TypeError, ValueError):
                    continue
                if close > 0 and math.isfinite(close):
                    rows[open_ms] = close
    return rows


def _target_binance_daily_archive(start: int) -> float | None:
    dt = datetime.fromtimestamp(int(start) / 1000, timezone.utc)
    day = dt.date().isoformat()
    # Current-day archives may still be partial/non-public; keep REST as the
    # final same-day resolver. Prior days are immutable enough for exact replay.
    if dt.date() >= datetime.now(timezone.utc).date():
        return None
    try:
        return _daily_archive_rows(day).get(int(start))
    except Exception:
        return None


def _target_coinbase_exchange(target_iso: str) -> tuple[float | None, str]:
    target = datetime.fromisoformat(target_iso.replace("Z", "+00:00"))
    ts = int(target.timestamp())
    start = ts - 60
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?" + urlencode({
        "granularity": 60,
        "start": start,
        "end": ts,
    })
    try:
        req = Request(url, headers={
            "User-Agent": "BTC-Prediction-Research/settlement",
            "Accept": "application/json",
        })
        with urlopen(req, timeout=20) as response:
            rows = json.loads(response.read().decode("utf-8"))
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, list) or len(row) < 5:
                continue
            try:
                if int(row[0]) == start:
                    close = float(row[4])
                    if math.isfinite(close) and close > 0:
                        return close, "coinbase_exchange"
            except (TypeError, ValueError):
                continue
    except Exception:
        pass
    return None, "unavailable"


def _target_bybit_linear(target_iso: str) -> tuple[float | None, str]:
    target = datetime.fromisoformat(target_iso.replace("Z", "+00:00"))
    ts_ms = int(target.timestamp() * 1000)
    start_ms = ts_ms - 60_000
    url = "https://api.bybit.com/v5/market/kline?" + urlencode({
        "category": "linear",
        "symbol": "BTCUSDT",
        "interval": "1",
        "start": start_ms,
        "end": ts_ms,
        "limit": 3,
    })
    try:
        req = Request(url, headers={
            "User-Agent": "BTC-Prediction-Research/settlement",
            "Accept": "application/json",
        })
        with urlopen(req, timeout=20) as response:
            obj = json.loads(response.read().decode("utf-8"))
        rows = ((obj.get("result") or {}).get("list") or []) if isinstance(obj, dict) else []
        for row in rows:
            if not isinstance(row, list) or len(row) < 5:
                continue
            try:
                if int(row[0]) == start_ms:
                    close = float(row[4])
                    if math.isfinite(close) and close > 0:
                        return close, "bybit_linear"
            except (TypeError, ValueError):
                continue
    except Exception:
        pass
    return None, "unavailable"
