import unittest
from src import historical_research as hr


class HistoricalTargetTests(unittest.TestCase):
    def test_label_uses_exact_elapsed_minutes(self):
        rows = [
            (0, [1, 2], 100.0),
            (60_000, [1, 2], 101.0),
            (120_000, [1, 2], 102.0),
        ]
        _, y, ids, _ = hr.labels(rows, 2)
        self.assertEqual(ids.tolist(), [0])
        self.assertEqual(y.tolist(), ["UP"])

    def test_gap_does_not_stretch_target(self):
        rows = [
            (0, [1, 2], 100.0),
            (60_000, [1, 2], 101.0),
            (180_000, [1, 2], 103.0),
            (240_000, [1, 2], 104.0),
        ]
        _, y, ids, _ = hr.labels(rows, 2)
        self.assertEqual(ids.tolist(), [60_000])
        self.assertEqual(y.tolist(), ["UP"])

    def test_feature_window_must_be_contiguous(self):
        self.assertTrue(hr._window_is_contiguous([0, 60_000, 120_000]))
        self.assertFalse(hr._window_is_contiguous([0, 60_000, 180_000]))

    def test_invalid_base_price_is_skipped(self):
        rows = [
            (0, [1, 2], 0.0),
            (120_000, [1, 2], 101.0),
        ]
        _, y, ids, bases = hr.labels(rows, 2)
        self.assertEqual(ids.tolist(), [])
        self.assertEqual(y.tolist(), [])
        self.assertEqual(bases.tolist(), [])


if __name__ == "__main__":
    unittest.main()
