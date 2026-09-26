import unittest

from src.ultimate_final_v13 import prediction_policy, validate_contract


class TestUltimateFinalV13(unittest.TestCase):
    def test_policy_escalates_with_risk(self):
        low = prediction_policy(
            predictability=0.90, failure_risk=0.10, drift=0.10, uncertainty=0.10
        )
        high = prediction_policy(
            predictability=0.10, failure_risk=0.90, drift=0.90, uncertainty=0.90
        )
        self.assertIn(low["action"], {"MAINTAIN_OR_LIGHT_UPDATE", "RECALCULATE"})
        self.assertIn(high["action"], {"DEEP_RECALCULATE", "FALLBACK", "ABSTAIN"})

    def test_abstain_for_extreme_unknown(self):
        out = prediction_policy(
            predictability=0.10,
            failure_risk=0.90,
            drift=0.70,
            uncertainty=0.95,
            ood=0.95,
        )
        self.assertEqual(out["action"], "ABSTAIN")

    def test_contract_requires_pit(self):
        valid = {
            "prediction_time": "2026-01-01T00:00:00+00:00",
            "valid_until": "2026-01-01T00:05:00+00:00",
            "data_snapshot": {},
            "model_version": "v13",
            "strategy": "STANDARD_ENSEMBLE",
            "action": "MAINTAIN_OR_LIGHT_UPDATE",
            "output_format": "PROBABILITY",
            "confidence": 0.5,
            "predictability": 0.5,
            "uncertainty": 0.2,
            "pit_status": "PASS",
            "research_only": True,
            "production_changed": False,
        }
        self.assertEqual(validate_contract(valid), (True, []))
        invalid = dict(valid, pit_status="FAIL_CLOSED")
        ok, errors = validate_contract(invalid)
        self.assertFalse(ok)
        self.assertIn("pit_not_verified", errors)


if __name__ == "__main__":
    unittest.main()
