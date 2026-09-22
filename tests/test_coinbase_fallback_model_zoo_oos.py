import unittest
import numpy as np

from src.coinbase_fallback_model_zoo_oos import metrics


class CoinbaseFallbackModelZooTests(unittest.TestCase):
    def test_metrics_normalize_probabilities(self):
        result = metrics(
            ["DOWN", "FLAT", "UP"],
            [[2, 1, 1], [1, 2, 1], [1, 1, 2]],
        )
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["accuracy"], 1.0)

    def test_probabilities_with_zero_entries_are_clipped(self):
        result = metrics(
            ["DOWN"],
            [[0, 0, 1]],
        )
        self.assertTrue(np.isfinite(result["logloss"]))


if __name__ == "__main__":
    unittest.main()
