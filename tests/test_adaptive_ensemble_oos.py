import unittest
from unittest.mock import patch

from src import adaptive_ensemble_oos as adaptive


class AdaptiveEnsembleHoldoutTests(unittest.TestCase):
    def test_development_selection_and_holdout_are_separate(self):
        rows = [{"y": "DOWN"}] * 2500 + [{"y": "UP"}] * 625
        with patch.object(adaptive, "load_rows", return_value=rows):
            with patch.object(adaptive, "_evaluate_development", return_value=({
                "blocks": 8,
                "stable_blocks": 8,
                "stable_improved_logloss_ratio": 0.75,
                "stable_improved_brier_ratio": 0.75,
                "stable_mean_logloss_delta": -0.01,
                "stable_mean_brier_delta": -0.01,
            }, [], True, True, [])):
                with patch.object(adaptive, "_predict_block", side_effect=[None, None]):
                    result = adaptive.evaluate("5m")
        self.assertEqual(result["status"], "OK")
        self.assertFalse(result["final_holdout_used_for_selection"])
        self.assertTrue(result["final_holdout_protected"])
        self.assertTrue(result["eligible_pending_frozen_holdout_confirmation"])

    def test_insufficient_rows_fails_closed(self):
        with patch.object(adaptive, "load_rows", return_value=[]),              patch.object(adaptive, "load_archive_research_rows", return_value=[]):
            result = adaptive.evaluate("5m")
        self.assertEqual(result["status"], "DEFERRED")
        self.assertEqual(result["reason"], "insufficient_research_rows_for_protected_holdout")

    def test_stable_weights_are_normalized_and_floor_bounded(self):
        import numpy as np

        y = ["DOWN", "FLAT", "UP"] * 120
        parts = []
        rng = np.random.default_rng(42)
        for _ in range(3):
            p = rng.random((len(y), 3))
            p /= p.sum(axis=1, keepdims=True)
            parts.append(p)

        weights = adaptive._rolling_validation_weights(parts, y)
        self.assertIsNotNone(weights)
        self.assertAlmostEqual(sum(weights), 1.0, places=10)
        self.assertTrue(all(w >= 0.10 for w in weights))


if __name__ == "__main__":
    unittest.main()
