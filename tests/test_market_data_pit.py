import unittest
from unittest.mock import patch

from src.market_data import _fresh_closed_candle_suffix


class MarketDataPitTests(unittest.TestCase):
    @staticmethod
    def _rows(latest_open_ms):
        return [
            [latest_open_ms - 39 * 60_000 + i * 60_000, 1.0, 1.0, 1.0, 1.0, 1.0]
            for i in range(40)
        ]

    def test_future_candle_is_rejected(self):
        latest_open = 1_000_000_000_000
        with patch("src.market_data.time.time", return_value=(latest_open + 60_000 + 5_000) / 1000):
            rows = _fresh_closed_candle_suffix(self._rows(latest_open), 40)
        self.assertEqual(len(rows), 40)

        future_open = latest_open + 60_000
        with patch("src.market_data.time.time", return_value=(latest_open + 60_000 + 5_000) / 1000):
            rows = _fresh_closed_candle_suffix(self._rows(future_open), 40)
        self.assertEqual(rows, [])

    def test_stale_candle_is_rejected(self):
        latest_open = 1_000_000_000_000
        now_ms = latest_open + 60_000 + 180_001
        with patch("src.market_data.time.time", return_value=now_ms / 1000):
            rows = _fresh_closed_candle_suffix(self._rows(latest_open), 40)
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
