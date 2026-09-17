from __future__ import annotations

from datetime import datetime
from market_data import _binance


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
    try:
        price = _target_binance(start, ts)
    except Exception:
        return None, 'unavailable'
    return (price, 'binance') if price is not None else (None, 'unavailable')


def _target_binance(start: int, end: int) -> float | None:
    rows = _binance('fapi.binance.com/fapi/v1/klines', {
        'symbol': 'BTCUSDT', 'interval': '1m', 'startTime': start, 'endTime': end, 'limit': 2,
    })
    return _target_from_rows(rows, start)
