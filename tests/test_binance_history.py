import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import binance_history  # noqa: E402


class TestBinanceHistory(unittest.TestCase):
    def test_contiguous_suffix_drops_older_rows_before_gap(self):
        rows = [
            [0, 1, 1, 1, 1, 1],
            [60_000, 1, 1, 1, 1, 1],
            [120_000, 1, 1, 1, 1, 1],
            [240_000, 1, 1, 1, 1, 1],
            [300_000, 1, 1, 1, 1, 1],
        ]
        out = binance_history._contiguous_suffix(rows)
        self.assertEqual([r[0] for r in out], [240_000, 300_000])

    def test_contiguous_suffix_is_deduplicated_and_sorted(self):
        rows = [
            [120_000, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [60_000, 1, 1, 1, 1, 1],
            [60_000, 2, 2, 2, 2, 2],
        ]
        out = binance_history._contiguous_suffix(rows)
        self.assertEqual([r[0] for r in out], [0, 60_000, 120_000])
        self.assertEqual(out[-1][0], 120_000)

    def test_contiguous_suffix_fail_closed_when_latest_row_stands_alone(self):
        rows = [[0, 1, 1, 1, 1, 1], [120_000, 1, 1, 1, 1, 1]]
        self.assertEqual(len(binance_history._contiguous_suffix(rows)), 1)


if __name__ == "__main__":
    unittest.main()
