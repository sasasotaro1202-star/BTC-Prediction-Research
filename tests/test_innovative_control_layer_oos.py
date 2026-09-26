import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.innovative_control_layer_oos import (
    _bootstrap_ci,
    _coverage_metrics,
    _route_weights,
    _past_quality_logloss,
    _stress_test,
    disagreement_features,
)


class TestInnovativeControlLayer(unittest.TestCase):
    def _panel(self, n=8):
        return {
            "logreg": np.tile([0.60, 0.20, 0.20], (n, 1)),
            "extra_trees": np.tile([0.30, 0.20, 0.50], (n, 1)),
            "hgb": np.tile([0.40, 0.20, 0.40], (n, 1)),
            "lightgbm": np.tile([0.45, 0.20, 0.35], (n, 1)),
        }

    def test_disagreement_features_have_expected_contract(self):
        out = disagreement_features(self._panel())
        self.assertIn("probability_range", out)
        self.assertIn("pairwise_disagreement", out)
        self.assertGreaterEqual(out["pairwise_disagreement"], 0.0)
        self.assertLessEqual(out["pairwise_disagreement"], 1.0)

    def test_first_block_quality_prior_never_uses_current_labels(self):
        current = {
            "logreg": {"logloss": 0.01},
            "extra_trees": {"logloss": 0.02},
            "hgb": {"logloss": 0.03},
            "lightgbm": {"logloss": 0.04},
        }
        out = _past_quality_logloss([], current)
        for value in out.values():
            self.assertAlmostEqual(value, float(np.log(3.0)), places=9)

    def test_soft_router_weights_are_normalized_and_smoothed(self):
        q = {e: 1.0 for e in ("logreg", "extra_trees", "hgb", "lightgbm")}
        d = disagreement_features(self._panel())
        drift = {"drift_score": 0.5}
        risks = {e: 0.5 for e in q}
        prev = np.full(4, 0.25)
        w = _route_weights(q, d, drift, 0.3, risks, previous=prev)
        self.assertTrue(np.isclose(w.sum(), 1.0))
        self.assertTrue(np.all(w > 0))

    def test_moving_block_ci_handles_short_series_fail_closed(self):
        self.assertEqual(_bootstrap_ci([0.1, 0.2])["low"], None)

    def test_coverage_metrics_reports_requested_fields(self):
        probs = np.asarray([
            [0.8, 0.1, 0.1],
            [0.4, 0.3, 0.3],
            [0.2, 0.7, 0.1],
        ])
        out = _coverage_metrics(probs, ["DOWN", "DOWN", "FLAT"], 0.0)
        for key in ("coverage", "accuracy", "logloss", "brier", "ece"):
            self.assertIn(key, out)

    def test_stress_test_uses_scalar_pairwise_disagreement(self):
        reference = {
            "weights": [0.25, 0.25, 0.25, 0.25],
            "quality_logloss": {e: 1.0 for e in ("logreg", "extra_trees", "hgb", "lightgbm")},
            "disagreement": {"pairwise_disagreement": 0.20},
            "drift": {"drift_score": 0.20},
            "predictability": 0.70,
            "failure_risks": {e: 0.30 for e in ("logreg", "extra_trees", "hgb", "lightgbm")},
        }
        out = _stress_test(reference)
        self.assertEqual(out["status"], "OK")
        self.assertIn("disagreement_spike", out["scenarios"])


if __name__ == "__main__":
    unittest.main()
