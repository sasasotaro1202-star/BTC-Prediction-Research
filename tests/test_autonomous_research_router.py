import json
import tempfile
import unittest
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
            {"generated_at_utc": "2026-10-04T00:00:00Z", "horizons": {}},
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

    def test_healthy_state_routes_to_bounded_routine_v13(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_ultimate_final_v13_e2e.yml")
            self.assertEqual(route["threshold_seconds"], 86400)
            self.assertFalse(route["production_impact"])
            self.assertEqual(route["reason"], "routine_future_generalization_evidence_refresh")

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
