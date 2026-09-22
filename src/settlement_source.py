from __future__ import annotations

from datetime import datetime
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
    if preferred_source != 'binance_futures':
        raise ValueError('only_binance_futures_is_valid_production_benchmark')
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
