from __future__ import annotations

from datetime import datetime
from market_data import _binance, _bybit, coinbase_rows, kraken_rows, target_close_binance


def preferred_source_from_scenario(scenario: dict) -> str:
    quality = scenario.get('data_quality') or {}
    fallback = str(quality.get('price_feature_fallback', ''))
    if fallback == 'bybit':
        return 'bybit_futures'
    if fallback == 'coinbase':
        return 'coinbase'
    if fallback == 'kraken':
        return 'kraken'
    # A fresh bootstrap cache currently records its source in the cache file;
    # Coinbase is the only live cache fallback accepted by the current predictor.
    if fallback == 'fresh_bootstrap_cache':
        return 'coinbase'
    return 'binance_futures'


def _target_from_rows(rows, start_ms: int) -> float | None:
    for row in rows:
        if int(row[0]) == start_ms:
            return float(row[4])
    return None


def target_close_preferred(target_iso: str, preferred_source: str) -> tuple[float | None, str]:
    target = datetime.fromisoformat(target_iso.replace('Z', '+00:00'))
    ts = int(target.timestamp() * 1000)
    start = ts - 60_000

    attempts = []
    if preferred_source == 'binance_futures':
        attempts.append(('binance', lambda: _target_binance(start, ts)))
    elif preferred_source == 'bybit_futures':
        attempts.append(('bybit', lambda: _target_bybit(start, ts)))
    elif preferred_source == 'coinbase':
        attempts.append(('coinbase', lambda: _target_coinbase(start)))
    elif preferred_source == 'kraken':
        attempts.append(('kraken', lambda: _target_kraken(start)))

    # If the prediction venue is unavailable at settlement time, retain the
    # existing resilient fallback chain rather than losing the outcome.
    for name, loader in attempts + [
        ('binance', lambda: _target_binance(start, ts)),
        ('bybit', lambda: _target_bybit(start, ts)),
        ('coinbase', lambda: _target_coinbase(start)),
        ('kraken', lambda: _target_kraken(start)),
    ]:
        try:
            price = loader()
            if price is not None:
                return price, name
        except Exception:
            continue
    return None, 'unavailable'


def _target_binance(start: int, end: int) -> float | None:
    rows = _binance('fapi.binance.com/fapi/v1/klines', {
        'symbol': 'BTCUSDT', 'interval': '1m', 'startTime': start, 'endTime': end, 'limit': 2,
    })
    return _target_from_rows(rows, start)


def _target_bybit(start: int, end: int) -> float | None:
    payload = _bybit('kline', {
        'category': 'linear', 'symbol': 'BTCUSDT', 'interval': '1', 'start': start, 'end': end, 'limit': 2,
    })
    rows = payload.get('result', {}).get('list', [])
    return _target_from_rows(rows, start)


def _target_coinbase(start: int) -> float | None:
    return _target_from_rows(coinbase_rows(10), start)


def _target_kraken(start: int) -> float | None:
    return _target_from_rows(kraken_rows(10), start)
