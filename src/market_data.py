"""BTC-only resilient public market-data adapters for GitHub Actions.

Binance is preferred where reachable. Public Bybit endpoints are used as a
fallback so a venue/geography HTTP block cannot stop the research cycle.
No credentials are required and no missing value is silently fabricated.
"""
from __future__ import annotations
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

UA = "BTC-Prediction-Research/5.2"

def http_json(url: str, timeout: int = 15):
    req = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def _binance(path: str, params: dict):
    return http_json(f"https://{path}?{urlencode(params)}")

def _bybit(path: str, params: dict):
    return http_json(f"https://api.bybit.com/v5/market/{path}?{urlencode(params)}")

def binance_klines(spot: bool = False, limit: int = 120):
    host = "api.binance.com/api/v3/klines" if spot else "fapi.binance.com/fapi/v1/klines"
    return _binance(host, {"symbol": "BTCUSDT", "interval": "1m", "limit": limit})

def bybit_klines(limit: int = 120):
    return _bybit("kline", {"category": "linear", "symbol": "BTCUSDT", "interval": "1", "limit": limit})

def closed_binance(rows):
    import time
    now = int(time.time() * 1000)
    return [r for r in rows if int(r[0]) + 60000 <= now]

def closed_bybit(payload):
    import time
    now = int(time.time() * 1000)
    rows = sorted(payload.get("result", {}).get("list", []), key=lambda r: int(r[0]))
    return [r for r in rows if int(r[0]) + 60000 <= now]

def resilient_1m_series(limit: int = 120):
    status = {}
    try:
        fut = closed_binance(binance_klines(False, limit)); status["binance_futures"] = "ok"
    except Exception as e:
        fut = []; status["binance_futures"] = f"error:{type(e).__name__}"
    try:
        spot = closed_binance(binance_klines(True, limit)); status["binance_spot"] = "ok"
    except Exception as e:
        spot = []; status["binance_spot"] = f"error:{type(e).__name__}"
    try:
        by = closed_bybit(bybit_klines(limit)); status["bybit_futures"] = "ok"
    except Exception as e:
        by = []; status["bybit_futures"] = f"error:{type(e).__name__}"

    if len(fut) < 40 and len(by) >= 40:
        # Bybit kline schema: start, open, high, low, close, volume, turnover.
        fut = [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in by]
        status["price_feature_fallback"] = "bybit"
    else:
        status["price_feature_fallback"] = "none"
    status["spot_fallback"] = "unavailable" if len(spot) < 40 else "none"
    return fut, spot, by, status

def binance_depth():
    return _binance("fapi.binance.com/fapi/v1/depth", {"symbol": "BTCUSDT", "limit": 50})

def bybit_depth():
    return _bybit("orderbook", {"category": "linear", "symbol": "BTCUSDT", "limit": 50})

def binance_premium():
    return _binance("fapi.binance.com/fapi/v1/premiumIndex", {"symbol": "BTCUSDT"})

def binance_oi():
    return _binance("fapi.binance.com/fapi/v1/openInterest", {"symbol": "BTCUSDT"})

def binance_taker():
    return _binance("fapi.binance.com/futures/data/takerBuySellVol", {"symbol": "BTCUSDT", "period": "5m", "limit": 1})

def bybit_funding():
    return _bybit("funding/history", {"category": "linear", "symbol": "BTCUSDT", "limit": 1})

def bybit_mark_price():
    return _bybit("tickers", {"category": "linear", "symbol": "BTCUSDT"})

def target_close_binance(target_iso: str):
    from datetime import datetime
    target = datetime.fromisoformat(target_iso.replace("Z", "+00:00"))
    ts = int(target.timestamp() * 1000)
    start = ts - 60000
    try:
        rows = _binance("fapi.binance.com/fapi/v1/klines", {"symbol": "BTCUSDT", "interval": "1m", "startTime": start, "endTime": ts, "limit": 2})
        for row in rows:
            if int(row[0]) == start:
                return float(row[4]), "binance"
    except Exception:
        pass
    try:
        payload = _bybit("kline", {"category": "linear", "symbol": "BTCUSDT", "interval": "1", "start": start, "end": ts, "limit": 2})
        for row in payload.get("result", {}).get("list", []):
            if int(row[0]) == start:
                return float(row[4]), "bybit"
    except Exception:
        pass
    return None, "unavailable"
