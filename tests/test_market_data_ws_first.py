import time
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import market_data


def make_ws_rows(n=40):
    now = int(time.time() * 1000)
    base = now - 120_000
    rows = []
    for i in range(n):
        open_ms = base - (n - 1 - i) * 60_000
        rows.append(
            {
                "open_time_ms": open_ms,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10.0,
                "taker_buy_base": 5.0,
                "event_time_ms": open_ms + 60_000,
                "retrieved_at_ms": now,
            }
        )
    return rows


class TestMarketDataWSFirst(unittest.TestCase):
    def test_fresh_binance_ws_cache_skips_binance_futures_rest(self):
        ws_rows = make_ws_rows()
        bybit_row = [int(ws_rows[-1]["open_time_ms"]), 101.0, 101.0, 101.0, 101.0, 1.0]
        spot_row = [int(ws_rows[-1]["open_time_ms"]), 100.0, 100.0, 100.0, 100.0, 1.0]
        observed = {}

        def fake_parallel(calls):
            observed["keys"] = set(calls)
            self.assertNotIn("binance_futures", calls)
            return {"bybit": [bybit_row], "binance_spot": [spot_row]}

        with patch.object(market_data, "load_binance_ws_cache", return_value=ws_rows):
            with patch.object(market_data, "_parallel_result_calls", side_effect=fake_parallel):
                fut, spot, by, status = market_data.resilient_1m_series(120)

        self.assertEqual(len(fut), 40)
        self.assertEqual(status["binance_futures"], "ok")
        self.assertEqual(status["binance_futures_transport"], "websocket")
        self.assertNotIn("binance_futures", observed["keys"])

    def test_without_fresh_ws_cache_binance_futures_rest_remains_fallback(self):
        observed = {}

        def fake_parallel(calls):
            observed["keys"] = set(calls)
            now = int(time.time() * 1000)
            rows = [
                [now - (39 - i) * 60_000, 100.0, 101.0, 99.0, 100.5, 10.0]
                for i in range(40)
            ]
            return {"bybit": [], "binance_futures": rows, "binance_spot": rows}

        with patch.object(market_data, "load_binance_ws_cache", return_value=[]):
            with patch.object(market_data, "_parallel_result_calls", side_effect=fake_parallel):
                fut, spot, by, status = market_data.resilient_1m_series(120)

        self.assertIn("binance_futures", observed["keys"])
        self.assertEqual(status["binance_futures"], "ok")


if __name__ == "__main__":
    unittest.main()
