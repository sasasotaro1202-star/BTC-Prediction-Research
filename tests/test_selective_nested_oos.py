import unittest

import numpy as np

from src.selective_nested_oos import _wilson_lower, choose_policy, evaluate_policy


class SelectiveNestedOOSTests(unittest.TestCase):
    def test_wilson_lower_is_below_observed_accuracy(self):
        assert _wilson_lower(80, 100) < 0.80

    def test_policy_requires_minimum_history(self):
        rows = [{"production": [0.8, 0.1, 0.1], "y": "UP"} for _ in range(100)]
        self.assertIsNone(choose_policy(rows))

    def test_policy_improves_when_high_confidence_subset_is_clean(self):
        rows = []
        for _ in range(500):
            rows.append({"production": [0.90, 0.05, 0.05], "y": "DOWN"})
            rows.append({"production": [0.60, 0.25, 0.15], "y": "UP"})
        policy = choose_policy(rows)
        self.assertIsNotNone(policy)
        result = evaluate_policy(rows, policy["min_confidence"], policy["min_margin"])
        self.assertGreaterEqual(result["accuracy"], 0.70)
        self.assertGreaterEqual(result["coverage"], 0.05)

    def test_probability_normalization_and_margin_are_used(self):
        rows = [
            {"production": [2.0, 0.0, 0.0], "y": "DOWN"},
            {"production": [0.34, 0.33, 0.33], "y": "DOWN"},
        ]
        r = evaluate_policy(rows, 0.55, 0.10)
        self.assertEqual(r["n"], 1)
        self.assertAlmostEqual(r["accuracy"], 1.0)
        self.assertAlmostEqual(r["coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
