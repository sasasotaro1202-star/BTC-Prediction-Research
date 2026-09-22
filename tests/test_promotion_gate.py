import unittest

from src.promotion_gate import evaluate_promotion


class PromotionGateTests(unittest.TestCase):
    def _robust(self):
        return {
            "research_only": True,
            "policy": "diagnostic_only_no_model_input_no_promotion_effect",
            "horizons": {
                "5m": {"status": "ok", "final_holdout_protected": True},
                "10m": {"status": "ok", "final_holdout_protected": True},
            },
        }

    def _pit(self):
        return {"ok": True, "pit_verified": True, "checked_predictions": 100, "violation_count": 0}

    def _cal(self):
        return {
            "5m": {"horizon": "5m", "temperature": 1.0, "model_version": "v5"},
            "10m": {"horizon": "10m", "temperature": 1.0, "model_version": "v5"},
        }

    def test_candidate_rejection_is_safe_hold(self):
        result = evaluate_promotion(
            {"status": "PASS"},
            self._robust(),
            {"5m": {"status": "rejected"}, "10m": {"status": "insufficient_history"}},
            self._pit(), self._cal(), {"ok": True},
        )
        self.assertFalse(result["promotion_allowed"])
        self.assertEqual(result["production_safety_gate"], "PASS")
        self.assertEqual(result["promotion_status"], "HOLD")
        self.assertIn("candidate_not_accepted_for_both_horizons", result["reason"])

    def test_missing_or_invalid_robustness_is_fail_closed(self):
        result = evaluate_promotion(
            {"status": "PASS"},
            {"research_only": True, "horizons": {}},
            {"5m": {"status": "accepted", "holdout_protected": True, "holdout_used_for_selection": False, "holdout_n": 100, "baseline_logloss": 0.50, "candidate_logloss": 0.49, "baseline_brier": 0.30, "candidate_brier": 0.29}, "10m": {"status": "accepted", "holdout_protected": True, "holdout_used_for_selection": False, "holdout_n": 100, "baseline_logloss": 0.55, "candidate_logloss": 0.54, "baseline_brier": 0.32, "candidate_brier": 0.31}},
            self._pit(), self._cal(), {"ok": True},
        )
        self.assertFalse(result["promotion_allowed"])
        self.assertEqual(result["production_safety_gate"], "HOLD")
        self.assertIn("robustness_evidence_invalid_or_incomplete", result["reason"])

    def test_all_evidence_is_only_eligible_not_auto_promoted(self):
        result = evaluate_promotion(
            {"status": "PASS"},
            self._robust(),
            {"5m": {"status": "accepted"}, "10m": {"status": "accepted"}},
            self._pit(), self._cal(), {"ok": True},
        )
        self.assertTrue(result["promotion_allowed"])
        self.assertEqual(result["promotion_status"], "ELIGIBLE_PENDING_EXPLICIT_PROMOTION")
        self.assertTrue(result["research_only"])


if __name__ == "__main__":
    unittest.main()
