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
        self._write(evidence, "research_readiness.json", {"state": "RESEARCH_VALIDATION"})
        self._write(
            evidence,
            "pit_oos_audit.json",
            {
                "ok": True,
                "pit_verified": True,
                "primary_horizon_gate": {
                    "5m": {"ready": True, "strict_primary_settled": 400},
                    "10m": {"ready": True, "strict_primary_settled": 400},
                },
            },
        )
        self._write(evidence, "research_health.json", {"ok": True})
        self._write(
            evidence,
            "data_frontier.json",
            {"candidates": {}},
        )
        self._write(
            evidence,
            "data_frontier_run.json",
            {"next_best_action": "collect_live_and_refresh_pit"},
        )
        self._write(
            root / "data" / "experience",
            "experience_summary.json",
            {"generated_at_utc": "2026-10-04T00:00:00Z"},
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

    def test_missing_readiness_fails_closed_to_readiness(self):
        with tempfile.TemporaryDirectory() as td:
            route = choose(Path(td))
            self.assertEqual(route["workflow"], "btc_research_readiness.yml")
            self.assertFalse(route["production_impact"])

    def test_frontier_repair_has_priority(self):
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

    def test_calibration_collection_has_priority_over_routine_v13(self):
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

    def test_healthy_state_routes_to_bounded_routine_v13(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            route = validate(choose(root))
            self.assertEqual(route["workflow"], "btc_ultimate_final_v13_e2e.yml")
            self.assertEqual(route["threshold_seconds"], 86400)
            self.assertFalse(route["production_impact"])


    def test_missing_generated_readiness_file_does_not_block_routing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._healthy_base(root)
            (root / "data" / "historical_research" / "research_readiness.json").unlink(missing_ok=True)
            self._write(
                root / "models",
                "5m.calibration.json",
                {"n_settled": 261, "fit_logloss": None, "holdout_logloss": None},
            )
            route = choose(root)
            self.assertEqual(route["workflow"], "btc_adaptive_calibration_replay.yml")

    def test_invalid_route_is_rejected(self):
        with self.assertRaises(ValueError):
            validate({
                "workflow": "arbitrary.yml",
                "threshold_seconds": 60,
                "production_impact": False,
            })


if __name__ == "__main__":
    unittest.main()
