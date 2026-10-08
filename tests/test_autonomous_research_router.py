import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.autonomous_research_router import choose, validate


class AutonomousResearchRouterTests(unittest.TestCase):
    def _write(self, root: Path, rel: str, obj):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), encoding="utf-8")

    def _healthy_base(self, root: Path):
        evidence = root / "data" / "historical_research"
        self._write(
            evidence,
            "pit_oos_audit.json",
            {
                "ok": True,
                "pit_verified": True,
                "primary_horizon_gate": {
                    "5m": {"ready": True, "strict_primary_settled": 400, "minimum": 300},
                    "10m": {"ready": True, "strict_primary_settled": 400, "minimum": 300},
                },
            },
        )
        self._write(evidence, "research_health.json", {"ok": True})
        self._write(evidence, "data_frontier.json", {"candidates": {}})
        self._write(
            root / "data" / "experience",
            "experience_summary.json",
            {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "horizons": {}},
        )
        self._write(
            evidence,
            "promotion_gate.json",
            {
                "production_safety_gate": "PASS",
                "promotion_allowed": True,
                "reason": "",
            },
        )
        for horizon in ("5m", "10m"):
            self._write(
                root / "models",
                f"{horizon}.calibration.json",
                {
                    "n_settled": 400,
                    "fit_logloss": 1.0,
                    "holdout_logloss": 1.0,
                },
            )

        self._write(
            root / "data" / "historical_research",
            "return_distribution_tail_oos.json",
            {
                "schema_version": 1,
                "research_only": True,
                "production_changed": False,
                "promotion_evidence_eligible": False,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "horizons": {
                    "5m": {"status": "OK"},
                    "10m": {"status": "OK"},
                },
            },
        )

    def test_missing_durable_evidence_fails_closed_to_readiness(self):
        with tempfile.TemporaryDirectory() as td:
            route = choose(Path(td))
            self.assertEqual(route["workflow"], "btc_research_readiness.yml")
            self.assertEqual(route["candidates"][0]["workflow"], "btc_research_readiness.yml")
            self.assertFalse(route["production_impact"])

    def test_calibration_collection_is_highest_priority(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            self._write(
                root / "models",
                "5m.calibration.json",
                {"n_settled": 261, "fit_logloss": None, "holdout_logloss": None},
            )
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_adaptive_calibration_replay.yml")
            self.assertEqual(route["candidates"][0]["priority"], 95)
            self.assertIn("5m:calibration_n=261<400", route["signals"])

    def test_promotion_robustness_hold_is_a_prioritized_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            self._write(
                root / "data" / "historical_research",
                "promotion_gate.json",
                {
                    "production_safety_gate": "HOLD",
                    "promotion_allowed": False,
                    "reason": "robustness_evidence_invalid_or_incomplete;candidate_or_frozen_holdout_non_regression_not_verified",
                },
            )
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_rich_production_challenger.yml")
            self.assertEqual(route["candidates"][0]["priority"], 90)
            self.assertIn("promotion_gate:robustness_blocked", route["signals"])
            self.assertEqual(route["candidates"][-1]["workflow"], "btc_ultimate_final_v13_e2e.yml")

    def test_confidence_overreach_routes_experience_then_selective(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            self._write(
                root / "data" / "experience",
                "experience_summary.json",
                {
                    "horizons": {
                        "5m": {
                            "cases": {
                                "confidence_bucket": {
                                    "0.70+": {"n": 52, "accuracy": 0.38, "avg_confidence": 0.80}
                                }
                            }
                        },
                        "10m": {
                            "cases": {
                                "confidence_bucket": {
                                    "0.70+": {"n": 52, "accuracy": 0.60, "avg_confidence": 0.71}
                                }
                            }
                        },
                    }
                },
            )
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_experience_policy_oos.yml")
            self.assertEqual(route["candidates"][1]["workflow"], "btc_selective_prediction_oos.yml")
            self.assertIn("5m:high_confidence_gap=0.420;n=52", route["signals"])

    def test_future_failure_risk_routes_uncertainty_research(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            self._write(
                root / "data" / "historical_research",
                "innovative_control_5m_future_failure_model.json",
                {
                    "horizon": "5m",
                    "research_only": True,
                    "latest_risk": {
                        "extra_trees": 0.81,
                        "logreg": 0.41,
                    },
                    "meta_samples": {
                        "extra_trees": 22,
                        "logreg": 22,
                    },
                },
            )
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_uncertainty_layer_oos.yml")
            self.assertEqual(route["priority"], 86)
            self.assertEqual(route["evidence_state"], "FUTURE_FAILURE_RISK")
            self.assertIn(
                "5m:extra_trees:future_failure_risk=0.810;n=22",
                route["signals"],
            )

    def test_immature_future_failure_risk_does_not_route(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            self._write(
                root / "data" / "historical_research",
                "innovative_control_5m_future_failure_model.json",
                {
                    "horizon": "5m",
                    "research_only": True,
                    "latest_risk": {"extra_trees": 0.99},
                    "meta_samples": {"extra_trees": 19},
                },
            )
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_ultimate_final_v13_e2e.yml")

    def test_material_drift_routes_uncertainty_research(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            self._write(
                root / "data" / "historical_research",
                "innovative_control_5m_drift_detector.json",
                {
                    "horizon": "5m",
                    "research_only": True,
                    "latest_drift": {"drift_score": 0.045, "model_disagreement_drift": 0.131},
                },
            )
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_uncertainty_layer_oos.yml")
            self.assertEqual(route["priority"], 84)
            self.assertIn("5m:drift_score=0.045;model_disagreement_drift=0.131", route["signals"])

    def test_immature_or_non_research_drift_does_not_route(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            self._write(
                root / "data" / "historical_research",
                "innovative_control_5m_drift_detector.json",
                {
                    "horizon": "5m",
                    "research_only": False,
                    "latest_drift": {"drift_score": 1.0, "model_disagreement_drift": 1.0},
                },
            )
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_ultimate_final_v13_e2e.yml")

    def test_frontier_is_used_when_no_higher_priority_issue_exists(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            self._write(
                root / "data" / "historical_research",
                "data_frontier.json",
                {
                    "candidates": {
                        "candidate:demo": {
                            "lifecycle": {"research_selection_eligible": True}
                        }
                    }
                },
            )
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_autonomous_data_frontier.yml")
            self.assertEqual(
                [x["workflow"] for x in route["candidates"]],
                ["btc_autonomous_data_frontier.yml", "btc_ultimate_final_v13_e2e.yml"],
            )


    def test_material_performance_regression_routes_experience_research(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            self._write(
                root / "data" / "historical_research",
                "performance_change.json",
                {
                    "changed": True,
                    "comparison_available": True,
                    "changes": [
                        {
                            "horizon": "10m",
                            "metric": "experience_recent100_accuracy",
                            "delta": -0.08,
                        },
                        {
                            "horizon": "10m",
                            "metric": "strict_pit_logloss",
                            "delta": 0.03,
                        },
                    ],
                },
            )
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_experience_policy_oos.yml")
            self.assertEqual(route["priority"], 87)
            self.assertEqual(route["evidence_state"], "PERFORMANCE_REGRESSION")
            self.assertIn(
                "10m:experience_recent100_accuracy:delta=-0.0800",
                route["signals"],
            )

    def test_persistent_recent_accuracy_floor_routes_even_without_score_delta(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            self._write(
                root / "data" / "historical_research",
                "performance_change.json",
                {
                    "changed": False,
                    "comparison_available": True,
                    "changes": [],
                    "scores": {
                        "5m": {
                            "experience_recent100_n": 100,
                            "experience_recent100_accuracy": 0.44,
                        },
                        "10m": {
                            "experience_recent100_n": 100,
                            "experience_recent100_accuracy": 0.27,
                        },
                    },
                },
            )
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_experience_policy_oos.yml")
            self.assertEqual(route["priority"], 87)
            self.assertEqual(route["evidence_state"], "PERFORMANCE_REGRESSION")
            self.assertIn(
                "10m:experience_recent100_accuracy_floor=0.2700<=0.30",
                route["signals"],
            )

    def test_small_performance_change_does_not_route(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            self._write(
                root / "data" / "historical_research",
                "performance_change.json",
                {
                    "changed": True,
                    "comparison_available": True,
                    "changes": [
                        {
                            "horizon": "5m",
                            "metric": "experience_recent100_accuracy",
                            "delta": -0.01,
                        },
                        {
                            "horizon": "5m",
                            "metric": "strict_pit_logloss",
                            "delta": 0.005,
                        },
                    ],
                },
            )
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_ultimate_final_v13_e2e.yml")

    def test_return_tail_lane_runs_when_evidence_is_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            (root / "data" / "historical_research" / "return_distribution_tail_oos.json").unlink()
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_return_distribution_tail_oos.yml")
            self.assertEqual(route["priority"], 88)
            self.assertIn("return_distribution_tail_evidence_missing", route["signals"])

    def test_return_tail_lane_runs_when_evidence_is_stale(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            self._write(
                root / "data" / "historical_research",
                "return_distribution_tail_oos.json",
                {
                    "schema_version": 1,
                    "research_only": True,
                    "production_changed": False,
                    "promotion_evidence_eligible": False,
                    "generated_at_utc": "2020-01-01T00:00:00Z",
                    "horizons": {
                        "5m": {"status": "DEFERRED"},
                        "10m": {"status": "DEFERRED"},
                    },
                },
            )
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_return_distribution_tail_oos.yml")
            self.assertIn("return_distribution_tail_stale:", route["signals"][0])

    def test_healthy_state_routes_to_bounded_routine_v13(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_ultimate_final_v13_e2e.yml")
            self.assertEqual(route["threshold_seconds"], 86400)
            self.assertFalse(route["production_impact"])
            self.assertEqual(route["reason"], "routine_future_generalization_evidence_refresh")

    def test_insufficient_live_robustness_routes_frontier_instead_of_repeated_rich_refit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            self._write(
                root / "data" / "historical_research",
                "promotion_gate.json",
                {
                    "production_safety_gate": "HOLD",
                    "promotion_allowed": False,
                    "reason": "robustness_evidence_invalid_or_incomplete;candidate_or_frozen_holdout_non_regression_not_verified",
                },
            )
            self._write(
                root / "data" / "historical_research",
                "robustness_oos_report.json",
                {
                    "research_only": True,
                    "policy": "diagnostic_only_no_model_input_no_promotion_effect",
                    "horizons": {
                        "5m": {"status": "insufficient_data", "n": 473, "minimum": 1000, "data_source": "live_binance_primary"},
                        "10m": {"status": "insufficient_data", "n": 473, "minimum": 1000, "data_source": "live_binance_primary"},
                    },
                },
            )
            self._write(
                root / "data" / "historical_research",
                "data_frontier.json",
                {"candidates": {"candidate:demo": {"lifecycle": {"research_selection_eligible": True}}}},
            )
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_autonomous_data_frontier.yml")
            self.assertEqual(route["priority"], 89)
            self.assertEqual(route["evidence_state"], "ROBUSTNESS_MATURATION")
            self.assertNotEqual(route["workflow"], "btc_rich_production_challenger.yml")
            self.assertIn("5m:robustness_live_n=473<1000", route["signals"])

    def test_insufficient_live_robustness_without_frontier_waits_on_routine_lane(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            self._write(
                root / "data" / "historical_research",
                "promotion_gate.json",
                {
                    "production_safety_gate": "HOLD",
                    "promotion_allowed": False,
                    "reason": "robustness_evidence_invalid_or_incomplete;candidate_or_frozen_holdout_non_regression_not_verified",
                },
            )
            self._write(
                root / "data" / "historical_research",
                "robustness_oos_report.json",
                {
                    "research_only": True,
                    "policy": "diagnostic_only_no_model_input_no_promotion_effect",
                    "horizons": {
                        "5m": {"status": "insufficient_data", "n": 473, "minimum": 1000, "data_source": "live_binance_primary"},
                        "10m": {"status": "insufficient_data", "n": 473, "minimum": 1000, "data_source": "live_binance_primary"},
                    },
                },
            )
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_ultimate_final_v13_e2e.yml")
            self.assertEqual(route["evidence_state"], "ROBUSTNESS_MATURATION_WAIT")

    def test_pit_gate_structure_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            self._write(
                root / "data" / "historical_research",
                "pit_oos_audit.json",
                {
                    "ok": True,
                    "pit_verified": True,
                    "primary_horizon_gate": {
                        "5m": {"ready": True, "strict_primary_settled": 400, "minimum": 300},
                        "10m": {"ready": True, "strict_primary_settled": "bad", "minimum": 300},
                    },
                },
            )
            route = choose(root)
            self.assertEqual(route["workflow"], "btc_research_readiness.yml")

    def test_ordered_router_retains_lower_priority_lanes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            self._write(
                root / "data" / "historical_research",
                "promotion_gate.json",
                {
                    "production_safety_gate": "HOLD",
                    "promotion_allowed": False,
                    "reason": "robustness evidence invalid or incomplete",
                },
            )
            self._write(
                root / "data" / "historical_research",
                "data_frontier.json",
                {
                    "candidates": {
                        "candidate:demo": {
                            "lifecycle": {"research_selection_eligible": True}
                        }
                    }
                },
            )
            self._write(
                root / "data" / "experience",
                "experience_summary.json",
                {
                    "horizons": {
                        "5m": {
                            "cases": {
                                "confidence_bucket": {
                                    "0.70+": {"n": 60, "accuracy": 0.30, "avg_confidence": 0.80}
                                }
                            }
                        }
                    }
                },
            )
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_rich_production_challenger.yml")
            self.assertEqual(
                [x["workflow"] for x in route["candidates"]],
                [
                    "btc_rich_production_challenger.yml",
                    "btc_experience_policy_oos.yml",
                    "btc_selective_prediction_oos.yml",
                    "btc_autonomous_data_frontier.yml",
                    "btc_ultimate_final_v13_e2e.yml",
                ],
            )

    def test_invalid_route_is_rejected(self):
        with self.assertRaises(ValueError):
            validate(
                {
                    "workflow": "arbitrary.yml",
                    "threshold_seconds": 60,
                    "production_impact": False,
                    "priority": 1,
                    "reason": "bad",
                    "evidence_state": "BAD",
                    "signals": ["bad"],
                    "candidates": [],
                }
            )

    def test_candidate_threshold_must_match_allowlist(self):
        with self.assertRaises(ValueError):
            validate(
                {
                    "workflow": "btc_ultimate_final_v13_e2e.yml",
                    "threshold_seconds": 86400,
                    "production_impact": False,
                    "priority": 50,
                    "reason": "ok",
                    "evidence_state": "HEALTHY_ROUTINE",
                    "signals": [],
                    "candidates": [
                        {
                            "workflow": "btc_ultimate_final_v13_e2e.yml",
                            "threshold_seconds": 60,
                            "reason": "ok",
                            "production_impact": False,
                            "priority": 50,
                            "evidence_state": "HEALTHY_ROUTINE",
                            "signals": [],
                        }
                    ],
                }
            )

    def test_candidate_order_must_be_priority_descending(self):
        with self.assertRaises(ValueError):
            validate(
                {
                    "workflow": "btc_experience_policy_oos.yml",
                    "threshold_seconds": 21600,
                    "production_impact": False,
                    "priority": 85,
                    "reason": "a",
                    "evidence_state": "A",
                    "signals": [],
                    "candidates": [
                        {
                            "workflow": "btc_experience_policy_oos.yml",
                            "threshold_seconds": 21600,
                            "reason": "a",
                            "production_impact": False,
                            "priority": 85,
                            "evidence_state": "A",
                            "signals": [],
                        },
                        {
                            "workflow": "btc_autonomous_data_frontier.yml",
                            "threshold_seconds": 900,
                            "reason": "b",
                            "production_impact": False,
                            "priority": 90,
                            "evidence_state": "B",
                            "signals": [],
                        },
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()


    def test_external_method_queue_routes_when_higher_priority_issues_are_clear(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            self._write(
                root / "data",
                "external_research_method_queue.json",
                {
                    "schema_version": 2,
                    "candidates": [
                        {
                            "queue_rank": 1,
                            "repository": "model/second",
                            "next_gate": "MODEL_OOS_GATE",
                        }
                    ],
                    "priority_gate": {
                        "immediate_local_reproduction": [
                            {
                                "repository": "model/second",
                                "gate": "MODEL_OOS_GATE",
                            }
                        ]
                    },
                },
            )
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_external_method_research.yml")
            self.assertEqual(route["priority"], 55)
            self.assertEqual(route["evidence_state"], "EXTERNAL_METHOD_RESEARCH_PENDING")

    def test_external_method_queue_is_not_routed_after_terminal_processing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            self._write(
                root / "data",
                "external_research_method_queue.json",
                {
                    "schema_version": 2,
                    "candidates": [
                        {
                            "queue_rank": 1,
                            "repository": "model/second",
                            "next_gate": "MODEL_OOS_GATE",
                        }
                    ],
                    "priority_gate": {
                        "immediate_local_reproduction": [
                            {"repository": "model/second", "gate": "MODEL_OOS_GATE"}
                        ]
                    },
                },
            )
            self._write(
                root / "data",
                "external_research_runtime.json",
                {
                    "schema_version": 1,
                    "research_only": True,
                    "production_changed": False,
                    "results": {"model/second": {"status": "SOURCE_VERIFIED"}},
                },
            )
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_ultimate_final_v13_e2e.yml")
