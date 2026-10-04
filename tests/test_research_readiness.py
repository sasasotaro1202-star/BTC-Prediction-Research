import unittest

from src.research_readiness import build_readiness, classify_state


class TestResearchReadiness(unittest.TestCase):
    def test_pit_collection_state_is_fail_closed(self):
        state, reasons = classify_state(
            production_integrity="PASS",
            research_health_ok=True,
            pit_verified=False,
            verified_primary_predictions=141,
            min_strict_pit_rows=300,
            promotion_status="HOLD",
            primary_horizon_gate={
                "5m": {"strict_primary_settled": 300, "minimum": 300, "ready": True},
                "10m": {"strict_primary_settled": 300, "minimum": 300, "ready": True},
            },
        )
        self.assertEqual(state, "PIT_COLLECTION")
        self.assertTrue(reasons)

    def test_integrity_blocks_before_pit(self):
        state, reasons = classify_state(
            production_integrity="HOLD",
            research_health_ok=True,
            pit_verified=True,
            verified_primary_predictions=500,
            min_strict_pit_rows=300,
            promotion_status="ELIGIBLE_PENDING_EXPLICIT_PROMOTION",
        )
        self.assertEqual(state, "BLOCKED_INTEGRITY")
        self.assertIn("production_integrity_not_pass", reasons)

    def test_promotion_review_requires_all_safety_evidence(self):
        result = build_readiness(
            {"status": "PASS"},
            {
                "ok": True,
                "checks": {
                    "5m": {"rows": 400, "invalid_probability_rows": 0, "ok": True},
                    "10m": {"rows": 400, "invalid_probability_rows": 0, "ok": True},
                },
            },
            {
                "pit_verified": True,
                "verified_primary_predictions": 320,
                "min_strict_pit_rows": 300,
                "violation_count": 0,
                "primary_horizon_gate": {
                    "5m": {"strict_primary_settled": 320, "minimum": 300, "ready": True},
                    "10m": {"strict_primary_settled": 301, "minimum": 300, "ready": True},
                },
            },
            {"promotion_status": "ELIGIBLE_PENDING_EXPLICIT_PROMOTION", "promotion_allowed": False},
        )
        self.assertEqual(result["readiness_state"], "PROMOTION_REVIEW")
        self.assertFalse(result["source_frontier"]["candidates"][0]["activation_allowed"])
        self.assertEqual(result["source_frontier"]["production_eligible_from_catalog"], 0)

    def test_never_promotes_from_catalog_metadata(self):
        result = build_readiness(
            {"status": "PASS"},
            {"ok": True, "checks": {"5m": {}, "10m": {}}},
            {"pit_verified": False, "verified_primary_predictions": 0, "min_strict_pit_rows": 300, "violation_count": 0},
            {"promotion_status": "HOLD"},
        )
        self.assertTrue(result["no_implicit_activation"])
        self.assertEqual(result["source_frontier"]["production_eligible_from_catalog"], 0)


    def test_calibration_collection_is_explicit_when_pit_is_ready_but_calibration_is_not(self):
        result = build_readiness(
            {"status": "PASS"},
            {"ok": True, "checks": {"5m": {}, "10m": {}}},
            {"pit_verified": True, "verified_primary_predictions": 392, "min_strict_pit_rows": 300, "violation_count": 0,
             "primary_horizon_gate": {
                 "5m": {"strict_primary_settled": 392, "minimum": 300, "ready": True},
                 "10m": {"strict_primary_settled": 392, "minimum": 300, "ready": True},
             }},
            {"promotion_status": "HOLD"},
            {
                "5m": {"status": "WAITING", "n_settled": 249, "remaining_rows": 151},
                "10m": {"status": "WAITING", "n_settled": 392, "remaining_rows": 8},
            },
        )
        self.assertEqual(result["readiness_state"], "CALIBRATION_COLLECTION")
        self.assertFalse(result["calibration"]["all_horizons_ready"])
        self.assertEqual(result["calibration"]["horizons"]["10m"]["remaining_rows"], 8)

    def test_per_horizon_gate_blocks_when_one_horizon_is_below_minimum(self):
        result = build_readiness(
            {"status": "PASS"},
            {"ok": True, "checks": {"5m": {}, "10m": {}}},
            {
                "pit_verified": True,
                "verified_primary_predictions": 700,
                "min_strict_pit_rows": 300,
                "violation_count": 0,
                "primary_horizon_gate": {
                    "5m": {"strict_primary_settled": 299, "minimum": 300, "ready": False},
                    "10m": {"strict_primary_settled": 401, "minimum": 300, "ready": True},
                },
            },
            {"promotion_status": "HOLD"},
            {
                "5m": {"status": "READY", "n_settled": 500},
                "10m": {"status": "READY", "n_settled": 500},
            },
        )
        self.assertEqual(result["readiness_state"], "PIT_COLLECTION")
        self.assertIn("per_horizon_strict_pit_gate_failed", result["reasons"][0])

    def test_missing_per_horizon_gate_is_fail_closed(self):
        result = build_readiness(
            {"status": "PASS"},
            {"ok": True, "checks": {"5m": {}, "10m": {}}},
            {
                "pit_verified": True,
                "verified_primary_predictions": 800,
                "min_strict_pit_rows": 300,
                "violation_count": 0,
            },
            {"promotion_status": "HOLD"},
            {
                "5m": {"status": "READY", "n_settled": 500},
                "10m": {"status": "READY", "n_settled": 500},
            },
        )
        self.assertEqual(result["readiness_state"], "PIT_COLLECTION")
        self.assertEqual(result["reasons"], ["per_horizon_strict_pit_gate_missing_or_invalid"])

    def test_calibration_waits_on_generation_or_artifact_hash_mismatch(self):
        result = build_readiness(
            {"status": "PASS"},
            {"ok": True, "checks": {"5m": {}, "10m": {}}},
            {"pit_verified": True, "verified_primary_predictions": 500, "min_strict_pit_rows": 300, "violation_count": 0,
             "primary_horizon_gate": {
                 "5m": {"strict_primary_settled": 500, "minimum": 300, "ready": True},
                 "10m": {"strict_primary_settled": 500, "minimum": 300, "ready": True},
             }},
            {"promotion_status": "HOLD"},
            {
                "5m": {"status": "WAITING", "n_settled": 500, "model_version": "old.v1", "binding_ok": False},
                "10m": {"status": "READY", "n_settled": 500, "model_version": "current.v1", "binding_ok": True},
            },
        )
        self.assertEqual(result["readiness_state"], "CALIBRATION_COLLECTION")
        self.assertFalse(result["calibration"]["all_horizons_ready"])


if __name__ == "__main__":
    unittest.main()
