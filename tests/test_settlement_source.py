import unittest
from unittest.mock import patch

import settlement_source as ss


class TestSettlementSource(unittest.TestCase):
    def test_preferred_source_mapping(self):
        self.assertEqual(ss.preferred_source_from_scenario({'data_quality': {'price_feature_fallback': 'bybit'}}), 'bybit_futures')
        self.assertEqual(ss.preferred_source_from_scenario({'data_quality': {'price_feature_fallback': 'coinbase'}}), 'coinbase')
        self.assertEqual(ss.preferred_source_from_scenario({'data_quality': {'price_feature_fallback': 'kraken'}}), 'kraken')
        self.assertEqual(ss.preferred_source_from_scenario({'data_quality': {'price_feature_fallback': 'fresh_bootstrap_cache'}}), 'coinbase')
        self.assertEqual(ss.preferred_source_from_scenario({}), 'binance_futures')

    def test_preferred_venue_is_used_before_fallback(self):
        target = '2026-09-15T10:05:00+00:00'
        start = 1757930700000
        with patch.object(ss, '_target_coinbase', return_value=101.0) as coinbase, \
             patch.object(ss, '_target_binance', return_value=102.0) as binance:
            price, source = ss.target_close_preferred(target, 'coinbase')
        self.assertEqual(price, 101.0)
        self.assertEqual(source, 'coinbase')
        coinbase.assert_called_once()
        binance.assert_not_called()

    def test_preferred_venue_failure_falls_back(self):
        target = '2026-09-15T10:05:00+00:00'
        with patch.object(ss, '_target_coinbase', side_effect=RuntimeError('temporary')), \
             patch.object(ss, '_target_binance', return_value=102.0) as binance:
            price, source = ss.target_close_preferred(target, 'coinbase')
        self.assertEqual(price, 102.0)
        self.assertEqual(source, 'binance')
        binance.assert_called_once()

    def test_no_source_returns_unavailable(self):
        target = '2026-09-15T10:05:00+00:00'
        with patch.object(ss, '_target_binance', return_value=None), \
             patch.object(ss, '_target_bybit', return_value=None), \
             patch.object(ss, '_target_coinbase', return_value=None), \
             patch.object(ss, '_target_kraken', return_value=None):
            price, source = ss.target_close_preferred(target, 'coinbase')
        self.assertIsNone(price)
        self.assertEqual(source, 'unavailable')

    def test_coinbase_rows_use_exact_target_minute(self):
        target = '2026-09-15T10:05:00+00:00'
        start = int(__import__('datetime').datetime.fromisoformat(target).timestamp() * 1000) - 60000
        rows = [[start - 60000, 1, 1, 1, 99, 1], [start, 1, 1, 1, 101, 1], [start + 60000, 1, 1, 1, 103, 1]]
        with patch.object(ss, 'coinbase_rows', return_value=rows):
            self.assertEqual(ss._target_coinbase(start), 101.0)


if __name__ == '__main__':
    unittest.main()
