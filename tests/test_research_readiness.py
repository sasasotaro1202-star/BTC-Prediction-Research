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


if __name__ == "__main__":
    unittest.main()
