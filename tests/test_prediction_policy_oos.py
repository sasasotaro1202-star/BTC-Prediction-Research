import unittest

from src.prediction_policy_oos import (
    VALID_ACTIONS,
    VALID_OUTPUTS,
    VALID_STRATEGIES,
    _state,
    select_action,
    select_output_format,
    select_strategy,
    select_compute_tier,
    rank_information_sources,
    project_prediction_trajectory,
    validate_record,
    evaluate_policy_case,
    summarize_policy_blocks,
)


class TestPredictionPolicyV13(unittest.TestCase):
    def _result(self):
        return {
            "generated_at_utc": "2026-09-26T12:00:00+00:00",
            "predictability": {
                "global": 0.72,
                "velocity": -0.03,
                "acceleration": -0.01,
            },
            "uncertainty": {"total": 0.28},
            "information_shock": {"shock_score": 0.15},
            "prediction_momentum": {"velocity": 0.08, "reversal": 0.05},
            "regime_transition": {
                "current": "RANGE",
                "next_regime": "TREND",
                "next_probability": {"TREND": 0.62, "RANGE": 0.38},
                "stay_probability": 0.38,
            },
            "counterfactual_stability": {"instability": 0.12},
            "hidden_state": {"stress": 0.15},
            "failure_monitoring": {
                "latest_failure_risk": {
                    "mean": 0.18,
                    "max": 0.26,
                    "expected_time_to_failure_blocks": 2.0,
                }
            },
        }

    def test_compute_tier_bounded(self):
        state = _state(self._result())
        self.assertEqual(select_compute_tier(state), "STANDARD")
        self.assertIn(select_compute_tier({**state, "uncertainty": 0.90}), {"DEEP", "ENSEMBLE"})

    def test_strategy_output_action_are_contract_values(self):
        state = _state(self._result())
        strategy = select_strategy(state)
        output = select_output_format(state, strategy["strategy"])
        action = select_action(state)
        self.assertIn(strategy["strategy"], VALID_STRATEGIES)
        self.assertIn(output["format"], VALID_OUTPUTS)
        self.assertIn(action["action"], VALID_ACTIONS)

    def test_hysteresis_maintains_stable_strategy(self):
        state = _state(self._result())
        current = select_strategy(state)["strategy"]
        action = select_action(
            state,
            previous_strategy=current,
            previous_predictability=0.72,
        )
        self.assertEqual(action["action"], "MAINTAIN")

    def test_information_value_is_not_fabricated(self):
        state = _state(self._result())
        ranked = rank_information_sources(state)
        self.assertGreaterEqual(len(ranked), 1)
        for item in ranked:
            self.assertIsNone(item["measured_incremental_oos_value"])
            self.assertEqual(item["measurement_status"], "UNVERIFIED_UNTIL_ABLATION")

    def test_trajectory_is_explicitly_proxy(self):
        state = _state(self._result())
        out = project_prediction_trajectory(state, steps=3)
        self.assertEqual(out["status"], "PROXY_NOT_OOS_VERIFIED")
        self.assertEqual(len(out["rows"]), 4)

    def test_validate_record_rejects_nonfinite_state(self):
        state = _state(self._result())
        record = {
            "research_only": True,
            "production_changed": False,
            "strategy_selection": {"strategy": "STANDARD_MODEL"},
            "output_selection": {"format": "SINGLE_PREDICTION"},
            "action_selection": {"action": "MAINTAIN"},
            "state": state,
        }
        validate_record(record)
        record["state"]["predictability"] = float("nan")
        with self.assertRaises(ValueError):
            validate_record(record)


    def test_policy_case_maps_to_existing_variant_and_is_causal(self):
        state = _state(self._result())
        block = {
            "index": 0,
            "y": ["DOWN", "UP"],
            "state": {
                "predictability": {
                    "global": 0.72,
                    "velocity": -0.01,
                    "acceleration": 0.0,
                },
                "uncertainty": {"total": 0.28},
                "regime_transition": {
                    "current": "RANGE",
                    "next_regime": "TREND",
                    "next_probability": {"TREND": 0.62, "RANGE": 0.38},
                    "stay_probability": 0.38,
                },
                "information_shock": {"shock_score": 0.15, "update_rate": 0.1},
                "prediction_momentum": {"velocity": 0.08, "acceleration": 0.0, "reversal": 0.05},
                "counterfactual_stability": {"instability": 0.12},
                "feature_reliability": {"global": 0.9},
                "source_reliability": 0.9,
                "error_correlation": {"mean_abs_error_correlation": 0.2},
                "hard_negative_density": 0.1,
                "drift": {"feature_drift": 0.1, "prediction_drift": 0.1, "drift_score": 0.1},
            },
            "failure_state": {
                "mean": 0.18,
                "max": 0.26,
                "expected_time_to_failure_blocks": 2.0,
            },
            "variants": {
                "soft_ensemble": {"probs": [[0.6, 0.2, 0.2], [0.2, 0.2, 0.6]]},
                "adaptive_ensemble": {"probs": [[0.6, 0.2, 0.2], [0.2, 0.2, 0.6]]},
                "three_layers_regime": {"probs": [[0.5, 0.2, 0.3], [0.2, 0.2, 0.6]]},
                "three_layers_retrieval": {"probs": [[0.5, 0.2, 0.3], [0.2, 0.2, 0.6]]},
                "full_architecture": {"probs": [[0.5, 0.2, 0.3], [0.2, 0.2, 0.6]]},
            },
        }
        case = evaluate_policy_case(block)
        self.assertIn(case["variant"], {"soft_ensemble", "adaptive_ensemble", "three_layers_regime", "three_layers_retrieval", "full_architecture"})
        self.assertEqual(len(case["probs"]), 2)

    def test_policy_summary_measures_matched_baseline(self):
        block = {
            "index": 0,
            "y": ["DOWN", "UP"],
            "state": self._result()["predictability"] and {
                "predictability": {"global": 0.72, "velocity": 0.0, "acceleration": 0.0},
                "uncertainty": {"total": 0.28},
                "regime_transition": {"current": "RANGE", "next_regime": "RANGE", "next_probability": {"RANGE": 0.9}, "stay_probability": 0.9},
                "information_shock": {"shock_score": 0.1, "update_rate": 0.1},
                "prediction_momentum": {"velocity": 0.0, "acceleration": 0.0, "reversal": 0.0},
                "counterfactual_stability": {"instability": 0.1},
                "feature_reliability": {"global": 0.9},
                "source_reliability": 0.9,
                "error_correlation": {"mean_abs_error_correlation": 0.2},
                "hard_negative_density": 0.1,
                "drift": {"feature_drift": 0.1, "prediction_drift": 0.1, "drift_score": 0.1},
            },
            "failure_state": {"mean": 0.1, "max": 0.1, "expected_time_to_failure_blocks": 3.0},
            "variants": {
                "soft_ensemble": {"probs": [[0.6,0.2,0.2],[0.2,0.2,0.6]]},
                "adaptive_ensemble": {"probs": [[0.6,0.2,0.2],[0.2,0.2,0.6]]},
                "three_layers_regime": {"probs": [[0.6,0.2,0.2],[0.2,0.2,0.6]]},
                "three_layers_retrieval": {"probs": [[0.6,0.2,0.2],[0.2,0.2,0.6]]},
                "full_architecture": {"probs": [[0.6,0.2,0.2],[0.2,0.2,0.6]]},
            },
        }
        out = summarize_policy_blocks([block])
        self.assertEqual(out["status"], "MEASURED_DEV_OOS")
        self.assertIn("matched_baseline", out)
        self.assertEqual(out["n_samples"], 2)


if __name__ == "__main__":
    unittest.main()
