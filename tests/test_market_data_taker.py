import time

from src.market_data import derive_binance_taker_from_closed_klines


def _rest_row(open_ms, volume, taker_buy):
    return [
        open_ms, "100", "101", "99", "100", str(volume),
        open_ms + 59_999, "10000", 100, str(taker_buy), "5000", "0"
    ]


def _ws_row(open_ms, volume, taker_buy, event_ms, retrieved_ms):
    return {
        "open_time_ms": open_ms,
        "volume": volume,
        "taker_buy_base": taker_buy,
        "event_time_ms": event_ms,
        "retrieved_at_ms": retrieved_ms,
    }


def test_rest_kline_taker_derivation_is_exact():
    cutoff = 2_000_000_000_000
    rows = [
        _rest_row(cutoff - (4 - i) * 60_000 - 60_000, 10, 7)
        for i in range(5)
    ]
    result = derive_binance_taker_from_closed_klines(rows, 5, cutoff)
    assert result is not None
    assert abs(result["taker_imbalance"] - 0.4) < 1e-12
    assert result["rows"] == 5
    assert result["source"] == "binance_futures_closed_klines"


def test_ws_kline_taker_derivation_preserves_pit_timestamps():
    cutoff = 2_000_000_000_000
    rows = [
        _ws_row(
            cutoff - (4 - i) * 60_000 - 60_000,
            10.0, 5.0, cutoff - 50_000, cutoff - 40_000
        )
        for i in range(5)
    ]
    result = derive_binance_taker_from_closed_klines(rows, 5, cutoff)
    assert result is not None
    assert abs(result["taker_imbalance"]) < 1e-12
    assert result["event_time_ms"] <= cutoff
    assert result["retrieved_at_ms"] <= cutoff


def test_taker_derivation_fails_closed_on_gap_and_future_row():
    cutoff = int(time.time() * 1000)
    rows = [
        _rest_row(cutoff - 5 * 60_000, 10, 5),
        _rest_row(cutoff - 3 * 60_000, 10, 5),
        _rest_row(cutoff - 2 * 60_000, 10, 5),
        _rest_row(cutoff - 60_000, 10, 5),
        _rest_row(cutoff, 10, 5),
    ]
    assert derive_binance_taker_from_closed_klines(rows, 5, cutoff) is None

    future = _rest_row(cutoff + 60_000, 10, 5)
    assert derive_binance_taker_from_closed_klines([future] * 5, 5, cutoff) is None


def test_taker_derivation_rejects_invalid_taker_volume():
    cutoff = 2_000_000_000_000
    rows = [_rest_row(cutoff - (4 - i) * 60_000 - 60_000, 10, 11) for i in range(5)]
    assert derive_binance_taker_from_closed_klines(rows, 5, cutoff) is None
