import unittest
import numpy as np

from src.maximum_future_generalization_v6 import (
    _panel_stats,
    _route_weights,
    _bootstrap_ci,
    _conformal_eval,
    _compute_tier,
)


class TestMaximumFutureGeneralizationV6(unittest.TestCase):
    def _panel(self):
        return {
            "logreg": np.tile([0.60, 0.20, 0.20], (8, 1)),
            "extra_trees": np.tile([0.30, 0.20, 0.50], (8, 1)),
            "hgb": np.tile([0.40, 0.20, 0.40], (8, 1)),
            "lightgbm": np.tile([0.45, 0.20, 0.35], (8, 1)),
        }

    def test_disagreement_contract(self):
        out = _panel_stats(self._panel())
        for key in (
            "std_probability", "probability_range", "js_divergence",
            "pairwise_class_disagreement", "flip_rate",
        ):
            self.assertIn(key, out)
        self.assertGreaterEqual(out["js_divergence"], 0.0)

    def test_route_weights_normalized_and_minimum_weight_guarded(self):
        state = {
            "error_correlation": {"by_expert": {e: 0.2 for e in (
                "logreg", "extra_trees", "hgb", "lightgbm"
            )}},
            "disagreement": {"pairwise_class_disagreement": 0.2},
            "predictability": {"global": 0.7},
            "drift": {"drift_score": 0.2},
            "uncertainty": {"total": 0.2},
            "source_reliability": 0.9,
            "retrieval": {"success_similarity": 0.8},
        }
        failure = {"by_expert": {e: 0.3 for e in (
            "logreg", "extra_trees", "hgb", "lightgbm"
        )}}
        quality = {e: 1.0 for e in failure["by_expert"]}
        w = _route_weights(state, quality, failure)
        self.assertTrue(np.isclose(w.sum(), 1.0))
        self.assertTrue(np.all(w >= 0.03))

    def test_conformal_evaluation_contract(self):
        probs = np.asarray([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2]])
        out = _conformal_eval(probs, ["DOWN", "FLAT"], {"q": 0.9, "alpha": 0.1})
        self.assertIn("coverage", out)
        self.assertIn("mean_set_size", out)

    def test_compute_tier_is_deterministic(self):
        self.assertEqual(_compute_tier(0.9, 0.1, 0.1), "STANDARD")
        self.assertEqual(_compute_tier(0.2, 0.2, 0.2), "ENSEMBLE")
        self.assertEqual(_compute_tier(0.4, 0.8, 0.2), "HARD_STRESS")
        self.assertEqual(_compute_tier(0.4, 0.2, 0.8), "HARD_STRESS")


if __name__ == "__main__":
    unittest.main()
