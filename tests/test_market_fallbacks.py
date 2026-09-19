import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import market_data  # noqa: E402
import predict  # noqa: E402


class TestMarketFallbacks(unittest.TestCase):
    def test_bybit_closed_parser_filters_open_candle_and_sorts(self):
        now = 1_800_000_000_000
        original = market_data.time.time
        try:
            market_data.time.time = lambda: now / 1000
            payload = {
                'result': {'list': [
                    [str(now - 60_000), '100', '101', '99', '100.5', '12', '0'],
                    [str(now), '101', '102', '100', '101.5', '13', '0'],
                    [str(now - 120_000), '98', '100', '97', '99', '11', '0'],
                ]}
            }
            rows = market_data.closed_bybit(payload)
            self.assertEqual([r[0] for r in rows], [now - 120_000, now - 60_000])
            self.assertEqual(rows[-1][4], 100.5)
        finally:
            market_data.time.time = original

    def test_cache_fallback_is_optional_and_well_shaped(self):
        result = market_data.cache_rows(120)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        rows, created, age_ms, fresh = result
        self.assertIsInstance(rows, list)
        self.assertIsInstance(created, str)
        self.assertTrue(age_ms is None or isinstance(age_ms, int))
        self.assertIsInstance(fresh, bool)
        if rows:
            self.assertEqual(len(rows[0]), 6)
            self.assertTrue(all(len(r) == 6 for r in rows))

    def test_binance_taker_uses_futures_api_host(self):
        original = market_data._get
        seen = []
        try:
            market_data._get = lambda url: seen.append(url) or [
                {'takerBuyVol': '2', 'takerSellVol': '1'}
            ]
            rows = market_data.binance_taker()
            self.assertEqual(rows[0]['takerBuyVol'], '2')
            self.assertEqual(len(seen), 1)
            self.assertTrue(seen[0].startswith(
                'https://fapi.binance.com/futures/data/takerBuySellVol?'
            ))
        finally:
            market_data._get = original

    def test_bybit_orderbook_imbalance_uses_v5_shape(self):
        payload = {
            'result': {
                'b': [['100', '5'], ['99', '3']],
                'a': [['101', '1'], ['102', '1']],
            }
        }
        self.assertAlmostEqual(predict.imbalance(payload, levels=25), 0.60, places=8)


if __name__ == '__main__':
    unittest.main()
