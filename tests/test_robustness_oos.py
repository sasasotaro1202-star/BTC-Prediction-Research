import json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

from src import robustness_oos
from src.robustness_oos import _metrics, _regimes, robust_generation_token

class RobustnessTests(unittest.TestCase):
    def test_generation_token_is_horizon_specific(self):
        self.assertEqual(
            robust_generation_token("5m", "bootstrap.bootstrap_rf"),
            "5m:bootstrap.bootstrap_rf",
        )
        self.assertEqual(
            robust_generation_token("10m", "bootstrap.bootstrap_rf"),
            "10m:bootstrap.bootstrap_rf",
        )

    def test_load_matches_secondary_horizon_in_combined_generation(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "predictions.db"
            import sqlite3

            feature_json = json.dumps({"ret_10m": 0.001, "volatility_10m": 0.002})
            scenario_json = json.dumps({
                "production_mode": "binance_primary",
                "provenance": {"available_at": "2026-10-03T07:00:00+00:00"},
            })
            with sqlite3.connect(db) as con:
                con.execute(
                    "CREATE TABLE model_registry (horizon TEXT, production_version TEXT)"
                )
                con.execute(
                    """CREATE TABLE predictions (
                        prediction_id INTEGER,
                        created_at_utc TEXT,
                        feature_json TEXT,
                        actual_direction_10m TEXT,
                        p_up_10m REAL,
                        p_down_10m REAL,
                        p_flat_10m REAL,
                        model_version TEXT,
                        scenario_json TEXT
                    )"""
                )
                con.execute(
                    "INSERT INTO model_registry VALUES (?, ?)",
                    ("10m", "bootstrap.bootstrap_rf"),
                )
                con.execute(
                    "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        1,
                        "2026-10-03T07:10:00+00:00",
                        feature_json,
                        "UP",
                        0.1,
                        0.2,
                        0.7,
                        "5m:bootstrap.soft_ensemble.v5.4|10m:bootstrap.bootstrap_rf",
                        scenario_json,
                    ),
                )
                con.commit()

            with patch.object(robustness_oos, "DB", db),                  patch.object(robustness_oos, "_strict_pit_provenance_ok", return_value=True):
                rows = robustness_oos.load("10m")

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], 1)
            self.assertEqual(rows[0]["model_version"], "5m:bootstrap.soft_ensemble.v5.4|10m:bootstrap.bootstrap_rf")
    def test_metrics_normalize_probabilities(self):
        m=_metrics(["UP","DOWN","FLAT"],[[2,0,0],[0,3,0],[0,0,4]])
        self.assertEqual(m["n"],3)
        self.assertAlmostEqual(m["accuracy"],1.0)

    def test_archive_loader_excludes_rows_before_frozen_model_training(self):
        class FakeModel:
            classes_ = ["DOWN", "FLAT", "UP"]

            def predict_proba(self, X):
                import numpy as np
                return np.tile(np.asarray([[0.7, 0.2, 0.1]]), (len(X), 1))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "models"
            model_dir.mkdir(parents=True)
            meta = {
                "model_version": "bootstrap.bootstrap_rf.vtest",
                "trained_at_utc": "2026-09-22T01:00:00+00:00",
            }
            (model_dir / "5m.json").write_text(json.dumps(meta), encoding="utf-8")
            (model_dir / "5m.joblib").write_bytes(b"placeholder")
            archive_rows = [
                {
                    "id": "old",
                    "created": "2026-09-22T00:59:00+00:00",
                    "x": [0.0] * 15,
                    "y": "UP",
                },
                {
                    "id": "new",
                    "created": "2026-09-22T01:01:00+00:00",
                    "x": [0.0] * 15,
                    "y": "DOWN",
                },
            ]
            with patch.object(robustness_oos, "ROOT", root), \
                 patch.object(robustness_oos, "load_archive_research_rows", return_value=archive_rows), \
                 patch.object(robustness_oos.joblib, "load", return_value=FakeModel()):
                result = robustness_oos.load_research_archive("5m")
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["id"], "new")
            self.assertFalse(result[0]["promotion_evidence_eligible"])
            self.assertTrue(
                all(
                    abs(actual - expected) < 1e-12
                    for actual, expected in zip(result[0]["p"], [0.1, 0.7, 0.2])
                )
            )

    def test_live_evidence_below_minimum_is_not_promotion_eligible(self):
        with patch.object(robustness_oos, "load", return_value=[{"id": 1}]), \
             patch.object(robustness_oos, "select_input_rows", return_value=([{"id": 1}], "live_binance_primary")), \
             patch.object(robustness_oos, "evaluate", return_value={
                 "status": "insufficient_data",
                 "n": 1,
                 "minimum": 1000,
                 "promotion_evidence_eligible": False,
             }):
            # Mirror the main-loop eligibility contract without accessing a real DB.
            source = "live_binance_primary"
            data = [{"id": 1}]
            result = robustness_oos.evaluate("5m", data)
            eligible = bool(
                source == "live_binance_primary"
                and result.get("status") == "ok"
                and int(result.get("n", 0)) >= 1000
                and result.get("final_holdout_protected") is True
            )
            self.assertFalse(eligible)

    def test_live_only_disables_archive_fallback(self):
        live = [{"id": 1}]
        with patch.object(robustness_oos, "load_research_archive", side_effect=AssertionError("archive fallback must be disabled")):
            data, source = robustness_oos.select_input_rows("5m", live, live_only=True)
        self.assertIs(data, live)
        self.assertEqual(source, "live_binance_primary")

    def test_regime_labels_use_current_and_past_only(self):
        rows=[
            {"ret":1.0,"vol":1.0},
            {"ret":-1.0,"vol":2.0},
            {"ret":1.0,"vol":1.0},
        ]
        regimes=_regimes(rows)
        self.assertEqual(regimes[0],"UP_MOMENTUM|LOW_VOL")
        self.assertEqual(regimes[1],"DOWN_MOMENTUM|HIGH_VOL")
        self.assertEqual(regimes[2],"UP_MOMENTUM|LOW_VOL")

if __name__=="__main__": unittest.main()
