import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import market_data


class TestLiveWebSocketPriority(unittest.TestCase):
    def test_fresh_binance_ws_is_primary_for_futures_history(self):
        now = int(time.time() * 1000)
        rows = []
        for i in range(60):
            open_time = now - (59 - i) * 60_000 - 60_000
            rows.append({
                "open_time_ms": open_time,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10.0,
                "taker_buy_base": 4.0,
                "event_time_ms": open_time + 59_000,
                "retrieved_at_ms": now,
            })

        captured = {}

        def fake_parallel(calls):
            captured["keys"] = set(calls)
            return {
                "bybit": {"result": {"list": []}},
                "binance_spot": [],
            }

        with patch.object(market_data, "load_binance_ws_cache", return_value=rows):
            with patch.object(market_data, "_parallel_result_calls", side_effect=fake_parallel):
                fut, spot, by, status = market_data.resilient_1m_series(limit=40)

        self.assertEqual(len(fut), 40)
        self.assertEqual(status["binance_futures"], "ok")
        self.assertEqual(status["binance_futures_transport"], "websocket")
        self.assertNotIn("binance_futures", captured["keys"])


if __name__ == "__main__":
    unittest.main()


class TestBinanceCacheTransportMetadata(unittest.TestCase):
    def test_rest_recovered_cache_is_not_mislabeled_as_websocket(self):
        import json
        import tempfile
        from pathlib import Path
        original = market_data.BINANCE_WS_CACHE
        original_loader = market_data.load_binance_ws_cache
        try:
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / 'binance_ws_1m.json'
                now = 1_800_000_000_000
                rows = []
                for i in range(40):
                    open_ms = now - (39 - i) * 60_000
                    rows.append({
                        'open_time_ms': open_ms,
                        'open': 100.0,
                        'high': 101.0,
                        'low': 99.0,
                        'close': 100.0,
                        'volume': 10.0,
                        'taker_buy_base': 5.0,
                        'event_time_ms': open_ms + 59_999,
                        'retrieved_at_ms': now,
                    })
                path.write_text(json.dumps({
                    'schema_version': 1,
                    'source': 'Binance USD-M Futures WebSocket',
                    'stream': 'btcusdt@kline_1m',
                    'transport': 'binance_futures_rest',
                    'rows': rows,
                }), encoding='utf-8')
                market_data.BINANCE_WS_CACHE = path
                market_data.load_binance_ws_cache = lambda p, limit: rows
                original_time = market_data.time.time
                market_data.time.time = lambda: now / 1000
                try:
                    _, _, _, status = market_data.resilient_1m_series(40)
                finally:
                    market_data.time.time = original_time
                self.assertEqual(status['binance_futures_transport'], 'rest')
        finally:
            market_data.BINANCE_WS_CACHE = original
            market_data.load_binance_ws_cache = original_loader
