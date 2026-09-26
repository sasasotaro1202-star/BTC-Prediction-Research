from __future__ import annotations

import time
import unittest
from unittest.mock import patch

import src.market_data as md


def _rows(count: int, *, latest_age_minutes: int) -> list[list[float]]:
    latest_open = (int(time.time() // 60) - latest_age_minutes) * 60_000
    return [
        [latest_open - (count - 1 - i) * 60_000, 1.0, 1.1, 0.9, 1.05, 10.0]
        for i in range(count)
    ]


class TestMarketDataFallbackChain(unittest.TestCase):
    def test_stale_bybit_history_does_not_block_coinbase_recovery(self):
        stale_bybit = _rows(40, latest_age_minutes=120)
        fresh_coinbase = _rows(120, latest_age_minutes=1)
        fresh_spot = _rows(40, latest_age_minutes=1)
        def fake_parallel(calls):
            return {
                "bybit": stale_bybit,
                "binance_spot": fresh_spot,
                "binance_futures": RuntimeError("binance futures unavailable"),
            }

        with patch.object(md, "load_binance_ws_cache", return_value=[]), \
             patch.object(md, "_capture_ws_suffix", return_value=[]), \
             patch.object(md, "binance_archive_daily_rows", side_effect=RuntimeError("archive unavailable")), \
             patch.object(md, "_parallel_result_calls", side_effect=fake_parallel), \
             patch.object(md, "coinbase_rows", return_value=fresh_coinbase), \
             patch.object(md, "cache_rows", return_value=([], "", None, False)):
            fut, spot, by, status = md.resilient_1m_series(limit=120)

        self.assertEqual(status.get("bybit_fallback"), "stale_or_insufficient")
        self.assertEqual(status.get("price_feature_fallback"), "coinbase")
        self.assertEqual(status.get("coinbase_futures"), "ok_closed")
        self.assertEqual(len(fut), 120)
        self.assertEqual(fut, fresh_coinbase)
        self.assertGreaterEqual(len(spot), 40)
        self.assertGreaterEqual(len(by), 40)


if __name__ == "__main__":
    unittest.main()
