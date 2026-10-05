import json
import tempfile
import unittest
from pathlib import Path

import performance_change


class TestPerformanceChange(unittest.TestCase):
    def _write(self, path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), encoding="utf-8")

    def _artifacts(self, root, acc5=0.40, ll10=1.20):
        exp = {
            "schema_version": 1,
            "research_only": True,
            "production_changed": False,
            "parse_errors": 0,
            "total_experiences": 10,
            "horizons": {
                "5m": {"total": {"n": 10, "accuracy": acc5}, "recent": {"100": {"n": 10, "accuracy": 0.50}}},
                "10m": {"total": {"n": 10, "accuracy": 0.60}, "recent": {"100": {"n": 10, "accuracy": 0.70}}},
            },
        }
        flat = {
            "ok": True,
            "settled_metrics": {
                "5m": {
                    "final": {"n": 10, "accuracy": acc5, "logloss": 1.30, "brier": 0.80, "ece": 0.20},
                    "calibrated": {"n": 10, "accuracy": 0.45, "logloss": 1.10, "brier": 0.65, "ece": 0.08},
                },
                "10m": {
                    "final": {"n": 10, "accuracy": 0.60, "logloss": ll10, "brier": 0.70, "ece": 0.17},
                    "calibrated": {"n": 10, "accuracy": 0.62, "logloss": 1.05, "brier": 0.64, "ece": 0.09},
                },
            },
        }
        self._write(root / "data/experience/experience_summary.json", exp)
        self._write(root / "data/historical_research/flat_diagnostic.json", flat)

    def test_initializes_baseline_without_false_delta(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._artifacts(root)
            payload = performance_change.compare_and_record(
                root / "data/historical_research/performance_snapshot.json",
                root / "data/historical_research/performance_change.json",
                root / "data/experience/experience_summary.json",
                root / "data/historical_research/flat_diagnostic.json",
            )
            self.assertFalse(payload["changed"])
            self.assertFalse(payload["comparison_available"])

    def test_reports_changed_metric(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._artifacts(root, acc5=0.40, ll10=1.20)
            snapshot = root / "data/historical_research/performance_snapshot.json"
            change = root / "data/historical_research/performance_change.json"
            performance_change.compare_and_record(
                snapshot, change,
                root / "data/experience/experience_summary.json",
                root / "data/historical_research/flat_diagnostic.json",
            )
            self._artifacts(root, acc5=0.42, ll10=1.17)
            payload = performance_change.compare_and_record(
                snapshot, change,
                root / "data/experience/experience_summary.json",
                root / "data/historical_research/flat_diagnostic.json",
            )
            changed = {(x["horizon"], x["metric"]): x["delta"] for x in payload["changes"]}
            self.assertAlmostEqual(changed[("5m", "experience_total_accuracy")], 0.02)
            self.assertAlmostEqual(changed[("5m", "final_accuracy")], 0.02)
            self.assertAlmostEqual(changed[("10m", "final_logloss")], -0.03)
            self.assertTrue(payload["changed"])

    def test_strict_pit_can_be_restricted_to_registered_model_generation(self):
        import sqlite3

        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "predictions.db"
            with sqlite3.connect(db) as con:
                con.execute(
                    """
                    CREATE TABLE model_registry (
                        horizon TEXT PRIMARY KEY,
                        production_version TEXT
                    )
                    """
                )
                con.execute(
                    "INSERT INTO model_registry VALUES (?, ?)",
                    ("5m", "current.v2"),
                )
                con.execute(
                    """
                    CREATE TABLE predictions (
                        created_at_utc TEXT,
                        target_5m TEXT,
                        actual_direction_5m TEXT,
                        p_up_5m REAL,
                        p_down_5m REAL,
                        p_flat_5m REAL,
                        model_version TEXT,
                        scenario_json TEXT
                    )
                    """
                )

                def scenario(ts):
                    sources = {
                        name: {
                            "status": "ok",
                            "available_at": ts,
                            "retrieved_at": ts,
                            "prediction_cutoff": ts,
                        }
                        for name in (
                            "binance_futures",
                            "binance_depth",
                            "binance_taker",
                            "binance_premium",
                        )
                    }
                    return json.dumps(
                        {
                            "decision_time_utc": ts,
                            "production_mode": "binance_primary",
                            "provenance": {
                                "available_at": ts,
                                "retrieved_at": ts,
                                "prediction_cutoff": ts,
                                "sources": sources,
                            },
                        }
                    )

                con.executemany(
                    "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            "2026-10-05T00:00:00+00:00",
                            "2026-10-05T00:05:00+00:00",
                            "UP",
                            0.8,
                            0.1,
                            0.1,
                            "5m:old.v1|10m:shared",
                            scenario("2026-10-05T00:00:00+00:00"),
                        ),
                        (
                            "2026-10-05T01:00:00+00:00",
                            "2026-10-05T01:05:00+00:00",
                            "DOWN",
                            0.1,
                            0.8,
                            0.1,
                            "10m:shared|5m:current.v2",
                            scenario("2026-10-05T01:00:00+00:00"),
                        ),
                    ],
                )

            self.assertEqual(
                performance_change._current_production_model_version("5m", db),
                "current.v2",
            )
            overall = performance_change._strict_pit_scores("5m", db)
            current = performance_change._strict_pit_scores(
                "5m",
                db,
                required_model_version="current.v2",
            )
            self.assertEqual(overall["n"], 2)
            self.assertEqual(current["n"], 1)
            self.assertEqual(current["required_model_version"], "current.v2")
            self.assertAlmostEqual(current["accuracy"], 1.0)

    def test_sample_count_change_alone_is_not_score_change(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._artifacts(root)
            snapshot = root / "data/historical_research/performance_snapshot.json"
            change = root / "data/historical_research/performance_change.json"
            performance_change.compare_and_record(
                snapshot, change,
                root / "data/experience/experience_summary.json",
                root / "data/historical_research/flat_diagnostic.json",
            )
            exp = json.loads((root / "data/experience/experience_summary.json").read_text())
            exp["horizons"]["5m"]["total"]["n"] = 11
            exp["total_experiences"] = 11
            (root / "data/experience/experience_summary.json").write_text(
                json.dumps(exp)
            )
            payload = performance_change.compare_and_record(
                snapshot, change,
                root / "data/experience/experience_summary.json",
                root / "data/historical_research/flat_diagnostic.json",
            )
            self.assertFalse(payload["changed"])
            self.assertFalse(payload["changes"])

    def test_no_change_after_identical_run(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._artifacts(root)
            snapshot = root / "data/historical_research/performance_snapshot.json"
            change = root / "data/historical_research/performance_change.json"
            performance_change.compare_and_record(
                snapshot, change,
                root / "data/experience/experience_summary.json",
                root / "data/historical_research/flat_diagnostic.json",
            )
            payload = performance_change.compare_and_record(
                snapshot, change,
                root / "data/experience/experience_summary.json",
                root / "data/historical_research/flat_diagnostic.json",
            )
            self.assertFalse(payload["changed"])
            self.assertFalse(payload["changes"])


if __name__ == "__main__":
    unittest.main()
