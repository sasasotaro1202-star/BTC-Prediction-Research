import unittest

from src.microstructure_features import (
    derive_market_flow_features,
    range_compression,
    volume_burst,
    vwap_distance,
)


class TestMicrostructureFeatures(unittest.TestCase):
    def _rows(self, n=50):
        rows = []
        for i in range(n):
            price = 100.0 + i * 0.1
            rows.append([i * 60_000, price - 0.05, price + 0.20, price - 0.20, price, 10.0 + i])
        return rows

    def _ws_rows(self, n=15):
        rows = []
        for i in range(n):
            rows.append({
                "open_time_ms": i * 60_000,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 10.0,
                "taker_buy_base": 7.0,
                "event_time_ms": i * 60_000 + 59_000,
                "retrieved_at_ms": i * 60_000 + 59_500,
            })
        return rows

    def test_vwap_distance_uses_only_supplied_closed_rows(self):
        rows = self._rows(15)
        value = vwap_distance(rows, 15)
        self.assertIsNotNone(value)
        self.assertGreater(value, 0.0)

    def test_volume_burst_and_range_compression_are_finite(self):
        rows = self._rows(40)
        self.assertGreater(volume_burst(rows, 5, 15), 0.0)
        self.assertGreater(range_compression(rows, 5, 15), 0.0)

    def test_derived_flow_requires_contiguous_taker_windows(self):
        values = derive_market_flow_features(self._rows(50), ws_rows=self._ws_rows(15))
        self.assertIsNotNone(values["vwap_distance_30m"])
        self.assertIsNotNone(values["volume_burst_15m"])
        self.assertIsNotNone(values["range_compression_15m"])
        self.assertAlmostEqual(values["taker_imbalance_5m"], 0.4)
        self.assertAlmostEqual(values["taker_imbalance_15m"], 0.4)
        self.assertAlmostEqual(values["taker_imbalance_delta_5m_15m"], 0.0)

    def test_missing_history_fails_closed_without_imputation(self):
        values = derive_market_flow_features(self._rows(10), ws_rows=[])
        self.assertIsNone(values["vwap_distance_30m"])
        self.assertIsNone(values["volume_burst_15m"])
        self.assertIsNone(values["range_compression_15m"])
        self.assertIsNone(values["taker_imbalance_5m"])
        self.assertIsNone(values["taker_imbalance_15m"])

    def test_zero_volume_vwap_fails_closed(self):
        rows = self._rows(10)
        for row in rows[-5:]:
            row[5] = 0.0
        self.assertIsNone(vwap_distance(rows, 5))


if __name__ == "__main__":
    unittest.main()
