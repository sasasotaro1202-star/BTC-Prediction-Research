import unittest

from src.ultimate_final_v13_meta_control import (
    active_information,
    adaptive_compute,
    probability_contract,
    select_output,
    select_strategy,
    update_policy,
)


class TestUltimateFinalV13MetaControl(unittest.TestCase):
    def _state(self, **updates):
        state = {
            "predictability": 0.60,
            "failure_risk_mean": 0.20,
            "failure_risk_max": 0.25,
            "drift": 0.10,
            "uncertainty": 0.20,
            "source_reliability": 0.90,
            "feature_unreliability": 0.10,
            "ood": 0.10,
            "disagreement": 0.20,
            "prediction_velocity": 0.0,
            "prediction_acceleration": 0.0,
        }
        state.update(updates)
        return state

    def test_probability_contract_fail_closed(self):
        self.assertEqual(probability_contract([0.2, 0.3, 0.5])["status"], "PASS")
        self.assertEqual(probability_contract([1.0, -1.0, 1.0])["status"], "FAIL")

    def test_strategy_escalates_and_safe_stops(self):
        evidence = {"oos_summary": {}}
        normal = select_strategy(self._state(), evidence)
        hard = select_strategy(self._state(uncertainty=0.90), evidence)
        unsafe = select_strategy(self._state(ood=0.95), evidence)
        self.assertIn(normal["strategy"], {"STANDARD_ENSEMBLE", "ADAPTIVE_ENSEMBLE", "THREE_LAYERS"})
        self.assertEqual(hard["strategy"], "ABSTAIN")
        self.assertEqual(unsafe["strategy"], "FALLBACK")

    def test_output_policy(self):
        self.assertEqual(
            select_output(self._state(predictability=0.80, uncertainty=0.15))["format"],
            "SINGLE",
        )
        self.assertIn(
            select_output(self._state(predictability=0.25, uncertainty=0.70))["format"],
            {"RANGE", "SET_OR_SCENARIO"},
        )
        self.assertEqual(
            select_output(self._state(ood=0.95))["format"],
            "ABSTAIN",
        )

    def test_update_and_compute_escalate(self):
        self.assertEqual(update_policy(self._state())["action"], "MAINTAIN")
        high = update_policy(self._state(predictability=0.10, uncertainty=0.80, drift=0.70, failure_risk_mean=0.75))
        self.assertEqual(high["action"], "DEEP_RECOMPUTE")
        self.assertLess(
            adaptive_compute(self._state())["compute_budget_units"],
            adaptive_compute(self._state(predictability=0.10, uncertainty=0.80, failure_risk_mean=0.75, drift=0.70))["compute_budget_units"],
        )

    def test_active_information_requires_incremental_oos_value(self):
        out = active_information({
            "active_information_sources": {
                "news": {"pit_status": "PASS", "incremental_oos_gain": None},
                "depth": {"pit_status": "PASS", "incremental_oos_gain": 0.4, "cost": 0.1},
            }
        })
        self.assertEqual(out["selected"]["source"], "depth")
        news = [x for x in out["candidates"] if x["source"] == "news"][0]
        self.assertEqual(news["status"], "DEFERRED")

    def test_invisible_pit_source_is_not_selected(self):
        out = active_information({
            "active_information_sources": {
                "news": {"pit_status": "UNKNOWN", "incremental_oos_gain": 99.0},
                "depth": {"pit_status": "PASS", "incremental_oos_gain": 0.1},
            }
        })
        self.assertEqual(out["selected"]["source"], "depth")
        self.assertEqual(
            [x for x in out["candidates"] if x["source"] == "news"][0]["status"],
            "DEFERRED",
        )


if __name__ == "__main__":
    unittest.main()
