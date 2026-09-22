import unittest
from unittest.mock import patch

import settlement_source as ss


class TestSettlementSource(unittest.TestCase):
    def good(self):
        return {'data_quality': {'binance_futures': 'ok', 'price_feature_fallback': 'none'}}

    def test_production_source_is_fixed(self):
        self.assertEqual(ss.preferred_source_from_scenario(self.good()), 'binance_futures')

    def test_rejects_fallback_prediction(self):
        row = self.good()
        row['data_quality']['price_feature_fallback'] = 'coinbase'
        with self.assertRaisesRegex(ValueError, 'invalid_fallback'):
            ss.preferred_source_from_scenario(row)

    def test_rejects_missing_binance_futures(self):
        row = self.good()
        row['data_quality']['binance_futures'] = 'error:HTTPError'
        with self.assertRaisesRegex(ValueError, 'missing_binance_futures'):
            ss.preferred_source_from_scenario(row)

    def test_target_requires_binance_benchmark(self):
        with self.assertRaisesRegex(ValueError, 'only_binance_futures'):
            ss.target_close_preferred('2026-09-15T10:05:00+00:00', 'coinbase')

    def test_websocket_cache_target_is_used_before_rest(self):
        rows = [{
            "open_time_ms": 1726394700000,
            "close": "102.5",
            "closed": True,
        }]
        with patch.object(ss, 'load_binance_ws_cache', return_value=rows), \
             patch.object(ss, '_target_binance', side_effect=AssertionError("REST should not be called")):
            price, source = ss.target_close_preferred(
                '2024-09-15T10:06:00+00:00', 'binance_futures'
            )
        self.assertEqual(price, 102.5)
        self.assertEqual(source, 'binance_websocket_cache')

    def test_websocket_cache_ignores_wrong_or_open_candle(self):
        rows = [
            {"open_time_ms": 1000, "close": "101", "closed": False},
            {"open_time_ms": 2000, "close": "0", "closed": True},
        ]
        with patch.object(ss, 'load_binance_ws_cache', return_value=rows):
            self.assertIsNone(ss._target_from_ws_cache(rows, 1000))
            self.assertIsNone(ss._target_from_ws_cache(rows, 2000))

    def test_binance_target_is_used(self):
        with patch.object(ss, '_target_binance', return_value=102.0) as binance:
            price, source = ss.target_close_preferred('2026-09-15T10:05:00+00:00', 'binance_futures')
        self.assertEqual(price, 102.0)
        self.assertEqual(source, 'binance')
        binance.assert_called_once()

    def test_binance_target_failure_is_unavailable(self):
        with patch.object(ss, '_target_binance', side_effect=RuntimeError('temporary')):
            price, source = ss.target_close_preferred('2026-09-15T10:05:00+00:00', 'binance_futures')
        self.assertIsNone(price)
        self.assertEqual(source, 'unavailable')


if __name__ == '__main__':
    unittest.main()
