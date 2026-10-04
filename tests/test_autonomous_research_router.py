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
            self.assertFalse(route["production_impact"])

    def test_calibration_collection_has_priority(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            self._write(
                root / "models",
                "5m.calibration.json",
                {"n_settled": 261, "fit_logloss": None, "holdout_logloss": None},
            )
            route = choose(root)
            self.assertEqual(route["workflow"], "btc_adaptive_calibration_replay.yml")
            self.assertIn("5m:calibration_n=261<400", route["signals"])

    def test_promotion_robustness_hold_routes_rich_challenger(self):
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
            route = choose(root)
            self.assertEqual(route["workflow"], "btc_rich_production_challenger.yml")
            self.assertEqual(route["threshold_seconds"], 86400)
            self.assertIn("promotion_gate:robustness_blocked", route["signals"])

    def test_confidence_overreach_routes_experience_policy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            self._write(
                root / "data" / "historical_research",
                "promotion_gate.json",
                {"production_safety_gate": "PASS", "promotion_allowed": True, "reason": ""},
            )
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
                        }
                    }
                },
            )
            route = choose(root)
            self.assertEqual(route["workflow"], "btc_experience_policy_oos.yml")
            self.assertIn("5m:high_confidence_gap=0.420;n=52", route["signals"])

    def test_frontier_routes_when_no_higher_priority_issue(self):
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
            route = choose(root)
            self.assertEqual(route["workflow"], "btc_autonomous_data_frontier.yml")

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

    def test_invalid_route_is_rejected(self):
        with self.assertRaises(ValueError):
            validate({
                "workflow": "arbitrary.yml",
                "threshold_seconds": 60,
                "production_impact": False,
                "priority": 1,
                "reason": "bad",
            })

    def test_empty_reason_is_rejected(self):
        with self.assertRaises(ValueError):
            validate({
                "workflow": "btc_ultimate_final_v13_e2e.yml",
                "threshold_seconds": 86400,
                "production_impact": False,
                "priority": 50,
                "reason": "",
            })


if __name__ == "__main__":
    unittest.main()
