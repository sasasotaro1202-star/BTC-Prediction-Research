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
            {
                "generated_at_utc": "2026-10-04T00:00:00Z",
                "horizons": {
                    "5m": {
                        "cases": {"confidence_bucket": {
                            "0.70+": {"n": 20, "accuracy": 0.60, "avg_confidence": 0.71}
                        }}
                    },
                    "10m": {
                        "cases": {"confidence_bucket": {
                            "0.70+": {"n": 20, "accuracy": 0.60, "avg_confidence": 0.71}
                        }}
                    },
                },
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

    def test_frontier_is_second_priority_when_confidence_is_clear(self):
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
            # The base fixture is below the confidence trigger, so frontier wins.
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_autonomous_data_frontier.yml")
            self.assertEqual(
                [x["workflow"] for x in route["candidates"]],
                ["btc_autonomous_data_frontier.yml", "btc_ultimate_final_v13_e2e.yml"],
            )

    def test_calibration_collection_is_first_priority(self):
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
            self.assertEqual(route["candidates"][0]["workflow"], "btc_adaptive_calibration_replay.yml")

    def test_high_confidence_overprediction_routes_diagnosis_then_selective(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            self._write(
                root / "data" / "experience",
                "experience_summary.json",
                {
                    "horizons": {
                        "5m": {"cases": {"confidence_bucket": {
                            "0.70+": {"n": 100, "accuracy": 0.38, "avg_confidence": 0.80}
                        }}},
                        "10m": {"cases": {"confidence_bucket": {
                            "0.70+": {"n": 101, "accuracy": 0.40, "avg_confidence": 0.81}
                        }}},
                    }
                },
            )
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_experience_policy_oos.yml")
            self.assertEqual(route["candidates"][1]["workflow"], "btc_selective_prediction_oos.yml")
            self.assertEqual(route["candidates"][2]["workflow"], "btc_ultimate_final_v13_e2e.yml")

    def test_experience_missing_routes_experience(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            (root / "data" / "experience" / "experience_summary.json").unlink()
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_experience_policy_oos.yml")

    def test_healthy_state_routes_to_bounded_routine_v13(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            route = validate(choose(root))
            # Healthy fixture has no frontier candidates and confidence is below trigger.
            self.assertEqual(route["workflow"], "btc_ultimate_final_v13_e2e.yml")
            self.assertEqual(route["threshold_seconds"], 86400)
            self.assertFalse(route["production_impact"])

    def test_invalid_route_is_rejected(self):
        with self.assertRaises(ValueError):
            validate({
                "workflow": "arbitrary.yml",
                "threshold_seconds": 60,
                "production_impact": False,
                "candidates": [{"workflow": "arbitrary.yml"}],
            })

    def test_candidate_threshold_must_match_allowlist(self):
        with self.assertRaises(ValueError):
            validate({
                "workflow": "btc_ultimate_final_v13_e2e.yml",
                "threshold_seconds": 86400,
                "production_impact": False,
                "candidates": [{
                    "workflow": "btc_ultimate_final_v13_e2e.yml",
                    "threshold_seconds": 60,
                    "production_impact": False,
                }],
            })


if __name__ == "__main__":
    unittest.main()
