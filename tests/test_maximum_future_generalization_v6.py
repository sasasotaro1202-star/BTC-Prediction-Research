import unittest
import numpy as np

from src.maximum_future_generalization_v6 import (
    _panel_stats,
    _route_weights,
    _bootstrap_ci,
    _conformal_eval,
    _compute_tier,
    _causal_current_snapshot,
    _numeric_state,
    _smoothed_binary_rate,
    _retrieval,
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

    def test_route_weights_accepts_bootstrap_state_without_retrieval(self):
        state = {
            "error_correlation": {"by_expert": {e: 0.2 for e in (
                "logreg", "extra_trees", "hgb", "lightgbm"
            )}},
            "disagreement": {"pairwise_class_disagreement": 0.2},
            "predictability": {"global": 0.7},
            "drift": {"drift_score": 0.2},
            "uncertainty": {"total": 0.2},
            "source_reliability": 0.9,
        }
        failure = {"by_expert": {e: 0.3 for e in (
            "logreg", "extra_trees", "hgb", "lightgbm"
        )}}
        quality = {e: 1.0 for e in failure["by_expert"]}
        w = _route_weights(state, quality, failure)
        self.assertTrue(np.isclose(w.sum(), 1.0))
        self.assertTrue(np.all(np.isfinite(w)))

    def test_causal_current_snapshot_excludes_later_block_rows(self):
        rows = [
            {"x": [1.0, 2.0, 3.0]},
            {"x": [9.0, 9.0, 9.0]},
        ]
        panel = self._panel()
        panel = {k: v[:2] for k, v in panel.items()}
        current_rows, current_panel = _causal_current_snapshot(rows, panel)
        self.assertEqual(current_rows, [rows[0]])
        for values in current_panel.values():
            self.assertEqual(values.shape, (1, 3))
        self.assertTrue(np.array_equal(current_panel["logreg"][0], panel["logreg"][0]))


    def test_compute_tier_is_deterministic(self):
        self.assertEqual(_compute_tier(0.9, 0.1, 0.1), "STANDARD")
        self.assertEqual(_compute_tier(0.2, 0.2, 0.2), "ENSEMBLE")
        self.assertEqual(_compute_tier(0.4, 0.8, 0.2), "HARD_STRESS")
        self.assertEqual(_compute_tier(0.4, 0.2, 0.8), "HARD_STRESS")


    def test_retrieval_query_state_bootstrap_is_safe(self):
        state = {
            "disagreement": {
                "std_probability": 0.1,
                "probability_range": 0.2,
                "js_divergence": 0.03,
                "pairwise_class_disagreement": 0.1,
                "flip_rate": 0.0,
                "disagreement_velocity": 0.0,
                "disagreement_acceleration": 0.0,
            },
            "drift": {"feature_drift": 0.1, "prediction_drift": 0.1, "drift_score": 0.1},
            "predictability": {"global": 0.5, "velocity": 0.0, "acceleration": 0.0},
            "uncertainty": {"total": 0.5},
            "error_correlation": {"mean_abs_error_correlation": 0.2},
            "feature_reliability": {"global": 0.9},
            "source_reliability": 0.9,
            "information_shock": {"shock_score": 0.1},
            "prediction_momentum": {"velocity": 0.0, "acceleration": 0.0},
            "regime_transition": {"stay_probability": 1.0},
            "hard_negative_density": 0.2,
        }
        vec = _numeric_state(state)
        self.assertEqual(vec.shape, (24,))
        self.assertTrue(np.isfinite(vec).all())
        out = _retrieval(state, [])
        self.assertEqual(out["neighbors"], [])
        self.assertEqual(out["probability"], [1/3, 1/3, 1/3])


    def test_retrieval_probability_is_json_serializable(self):
        state = {
            "disagreement": {"std_probability": 0.1, "probability_range": 0.1, "js_divergence": 0.01,
                             "pairwise_class_disagreement": 0.1, "flip_rate": 0.0,
                             "disagreement_velocity": 0.0, "disagreement_acceleration": 0.0},
            "drift": {"feature_drift": 0.1, "prediction_drift": 0.1, "drift_score": 0.1},
            "predictability": {"global": 0.6, "velocity": 0.0, "acceleration": 0.0},
            "uncertainty": {"total": 0.4},
            "error_correlation": {"mean_abs_error_correlation": 0.2},
            "feature_reliability": {"global": 0.9},
            "source_reliability": 0.9,
            "information_shock": {"shock_score": 0.1},
            "prediction_momentum": {"velocity": 0.0, "acceleration": 0.0},
            "regime_transition": {"stay_probability": 1.0},
            "hard_negative_density": 0.2,
            "meta_label": {"reliability": 0.6},
        }
        import json
        out = _retrieval(state, [{
            "state": state,
            "y": ["DOWN", "FLAT", "UP"],
            "soft": {"accuracy": 0.5},
        }, {
            "state": state,
            "y": ["UP", "DOWN", "FLAT"],
            "soft": {"accuracy": 0.5},
        }])
        json.dumps(out)
        self.assertIsInstance(out["probability"], list)
        self.assertEqual(len(out["probability"]), 3)


    def test_failure_prior_is_shrunk_and_prequential(self):
        self.assertAlmostEqual(_smoothed_binary_rate([]), 0.5)
        low_n = _smoothed_binary_rate([1])
        self.assertGreater(low_n, 0.5)
        self.assertLess(low_n, 0.7)
        high_n = _smoothed_binary_rate([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(high_n, 0.5)


if __name__ == "__main__":
    unittest.main()
