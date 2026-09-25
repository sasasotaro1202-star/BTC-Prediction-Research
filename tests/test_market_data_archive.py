import csv
import io
import zipfile
from datetime import datetime, timezone
from unittest.mock import patch

from src import market_data


def _zip(rows):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        payload = io.StringIO()
        writer = csv.writer(payload)
        for row in rows:
            writer.writerow(row)
        zf.writestr("BTCUSDT-1m-test.csv", payload.getvalue())
    return buf.getvalue()


def _rows(start_ms, count):
    return [
        [start_ms + i * 60_000, "100.0", "101.0", "99.0", "100.5", "10.0"]
        for i in range(count)
    ]


def test_daily_archive_returns_closed_contiguous_rows():
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = now_ms - 150 * 60_000
    payload = _zip(_rows(start, 150))

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def read(self):
            return payload

    with patch.object(market_data, "urlopen", return_value=Response()):
        rows = market_data.binance_archive_daily_rows(120)

    assert len(rows) == 120
    assert all(rows[i][0] - rows[i - 1][0] == 60_000 for i in range(1, len(rows)))
    assert rows[-1][0] + 60_000 <= now_ms


def test_daily_archive_rejects_crc_corruption_and_fails_closed():
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def read(self):
            return b"not a zip"

    with patch.object(market_data, "urlopen", return_value=Response()):
        try:
            market_data.binance_archive_daily_rows(120)
        except RuntimeError as exc:
            assert "insufficient contiguous rows" in str(exc)
        else:
            raise AssertionError("corrupt archive must fail closed")


def test_daily_archive_never_accepts_current_open_candle():
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = now_ms - 20 * 60_000
    rows = _rows(start, 21)
    rows[-1][0] = now_ms
    payload = _zip(rows)

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def read(self):
            return payload

    with patch.object(market_data, "urlopen", return_value=Response()):
        try:
            market_data.binance_archive_daily_rows(40)
        except RuntimeError:
            pass
        else:
            raise AssertionError("insufficient historical rows must defer")


def test_resilient_series_uses_archive_before_cross_venue_fallback():
    rows = [
        [1_700_000_000_000 + i * 60_000, 100.0, 101.0, 99.0, 100.5, 10.0]
        for i in range(120)
    ]
    network_error = RuntimeError("simulated_transport_failure")

    with patch.object(market_data, "load_binance_ws_cache", return_value=[]), \
         patch.object(market_data, "_capture_ws_suffix", return_value=[]), \
         patch.object(
             market_data,
             "_parallel_result_calls",
             return_value={
                 "bybit": network_error,
                 "binance_spot": network_error,
                 "binance_futures": network_error,
             },
         ), \
         patch.object(market_data, "bybit_mark_price", side_effect=network_error), \
         patch.object(market_data, "binance_archive_daily_rows", return_value=rows):
        fut, spot, bybit, status = market_data.resilient_1m_series(120)

    assert len(fut) == 120
    assert spot == []
    assert bybit == []
    assert status["binance_futures"] == "ok"
    assert status["binance_futures_transport"] == "binance_vision_daily_archive"
    assert status["price_feature_fallback"] == "none"
    assert "binance_futures_archive_retrieved_at_ms" in status


def test_daily_archive_exposes_closed_taker_buy_volume_with_pit_timestamps():
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = now_ms - 20 * 60_000
    rows = []
    for i in range(10):
        open_ms = start + i * 60_000
        rows.append([
            open_ms, "100.0", "101.0", "99.0", "100.5", "10.0",
            open_ms + 59_999, "1000.0", "20", "7.0", "700.0", "0"
        ])
    payload = _zip(rows)

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def read(self):
            return payload

    with patch.object(market_data, "urlopen", return_value=Response()):
        out = market_data.binance_archive_daily_taker_rows(5)
    after_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    assert len(out) == 5
    assert all(row["taker_buy_base"] == 7.0 for row in out)
    assert all(row["open_time_ms"] + 60_000 <= now_ms for row in out)
    assert all(row["event_time_ms"] <= now_ms for row in out)
    assert all(now_ms <= row["retrieved_at_ms"] <= after_ms for row in out)
    assert all(out[i]["open_time_ms"] - out[i - 1]["open_time_ms"] == 60_000 for i in range(1, len(out)))
