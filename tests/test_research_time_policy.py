import unittest

from src.research_time_policy import consecutive_minute_window, exact_elapsed_pairs


class ResearchTimePolicyTests(unittest.TestCase):
    def test_exact_elapsed_pairs_ignore_row_count_when_gap_exists(self):
        rows = [(0, 100.0), (60_000, 101.0), (180_000, 103.0)]
        pairs = list(exact_elapsed_pairs(rows, 2))
        self.assertEqual(pairs, [((60_000, 101.0), (180_000, 103.0))])

    def test_exact_elapsed_pairs_support_real_consecutive_horizon(self):
        rows = [(0, 100.0), (60_000, 101.0), (120_000, 102.0)]
        pairs = list(exact_elapsed_pairs(rows, 2))
        self.assertEqual(pairs, [((0, 100.0), (120_000, 102.0))])

    def test_consecutive_window_rejects_missing_minute(self):
        self.assertFalse(consecutive_minute_window([(0, 0.0), (60_000, 0.0), (180_000, 0.0)]))

    def test_consecutive_window_accepts_minute_grid(self):
        self.assertTrue(consecutive_minute_window([(0, 0.0), (60_000, 0.0), (120_000, 0.0)]))


if __name__ == "__main__":
    unittest.main()
