import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from model_compare import CLASSES, EMBARGO_BARS, PURGE_BARS, _strict_pit_provenance_ok, load_primary_production_strict_rows, metrics, normalize, prediction_precedes_target  # noqa: E402


class TestModelGuards(unittest.TestCase):
    def test_bootstrap_candidate_set_contains_diverse_safe_models(self):
        from bootstrap_train import candidate_factories
        names = [name for name, _ in candidate_factories()]
        self.assertEqual(
            names,
            ["logreg", "rf", "rf_replay", "extra_trees", "hgb", "soft_ensemble"],
        )


    def test_fallback_calibration_is_source_and_version_bound(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from src import predict

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "coinbase_5m.json").write_text(
                json.dumps({"model_version": "coinbase_fallback.rf.v1"}), encoding="utf-8"
            )
            (root / "coinbase_5m.calibration.json").write_text(
                json.dumps({
                    "model_version": "coinbase_fallback.rf.v1",
                    "n": 100,
                    "status": "accepted",
                    "temperature": 1.2,
                    "blend_weight": 0.15,
                }),
                encoding="utf-8",
            )
            with patch.object(predict, "MODEL_DIR", root):
                loaded = predict.load_fallback_calibration("coinbase", "5m")
                self.assertEqual(loaded["status"], "accepted")
                self.assertAlmostEqual(loaded["temperature"], 1.2)
                self.assertAlmostEqual(loaded["blend_weight"], 0.15)

                (root / "coinbase_5m.calibration.json").write_text(
                    json.dumps({
                        "model_version": "coinbase_fallback.rf.v2",
                        "n": 100,
                        "status": "accepted",
                        "temperature": 1.2,
                        "blend_weight": 0.15,
                    }),
                    encoding="utf-8",
                )
                rejected = predict.load_fallback_calibration("coinbase", "5m")
                self.assertEqual(rejected["status"], "stale_model_version")
                self.assertEqual(rejected["blend_weight"], 0.0)


    def test_research_class_order_matches_db_storage_conversion(self):
        self.assertEqual(CLASSES, ['DOWN', 'FLAT', 'UP'])
        stored_up_down_flat = [0.70, 0.10, 0.20]
        research_down_flat_up = [stored_up_down_flat[1], stored_up_down_flat[2], stored_up_down_flat[0]]
        self.assertEqual(research_down_flat_up, [0.10, 0.20, 0.70])
        self.assertEqual(max(range(3), key=lambda i: research_down_flat_up[i]), CLASSES.index('UP'))

    def test_probability_normalization_is_finite_and_sums_to_one(self):
        p = normalize([[0.7, 0.2, 0.1], [10.0, 0.0, 0.0]])
        self.assertEqual(p.shape, (2, 3))
        self.assertTrue(all(math.isfinite(float(x)) for x in p.ravel()))
        self.assertTrue(all(abs(float(row.sum()) - 1.0) < 1e-9 for row in p))

    def test_metrics_respect_research_class_order(self):
        ys = ['UP', 'DOWN', 'FLAT']
        probs = [[0.05, 0.05, 0.90], [0.90, 0.05, 0.05], [0.05, 0.90, 0.05]]
        m = metrics(ys, probs)
        self.assertEqual(m['accuracy'], 1.0)
        self.assertLess(m['logloss'], 0.2)
        self.assertLess(m['brier'], 0.1)

    def test_horizon_purge_and_embargo_are_conservative(self):
        self.assertEqual(PURGE_BARS['5m'], 5)
        self.assertEqual(PURGE_BARS['10m'], 10)
        self.assertGreaterEqual(EMBARGO_BARS['5m'], 60)
        self.assertGreaterEqual(EMBARGO_BARS['10m'], 60)

    def test_strict_pit_requires_complete_used_source_provenance(self):
        scenario = {
            "decision_time_utc": "2026-09-21T19:00:00+00:00",
            "production_mode": "binance_primary",
            "provenance": {
                "available_at": "2026-09-21T18:59:59+00:00",
                "retrieved_at": "2026-09-21T19:00:00+00:00",
                "prediction_cutoff": "2026-09-21T19:00:00+00:00",
                "sources": {
                    "binance_futures": {
                        "status": "ok",
                        "available_at": "2026-09-21T18:59:59+00:00",
                        "retrieved_at": "2026-09-21T19:00:00+00:00",
                        "prediction_cutoff": "2026-09-21T19:00:00+00:00",
                    },
                    "binance_depth": {
                        "status": "ok",
                        "available_at": "2026-09-21T18:59:59+00:00",
                        "retrieved_at": "2026-09-21T19:00:00+00:00",
                        "prediction_cutoff": "2026-09-21T19:00:00+00:00",
                    },
                    "binance_taker": {
                        "status": "ok",
                        "available_at": "2026-09-21T18:59:59+00:00",
                        "retrieved_at": "2026-09-21T19:00:00+00:00",
                        "prediction_cutoff": "2026-09-21T19:00:00+00:00",
                    },
                    "binance_premium": {
                        "status": "ok",
                        "available_at": "2026-09-21T18:59:59+00:00",
                        "retrieved_at": "2026-09-21T19:00:00+00:00",
                        "prediction_cutoff": "2026-09-21T19:00:00+00:00",
                    },
                },
            },
        }
        self.assertTrue(_strict_pit_provenance_ok(scenario, "2026-09-21T19:00:00+00:00"))
        scenario["provenance"]["sources"]["binance_taker"]["available_at"] = None
        self.assertFalse(_strict_pit_provenance_ok(scenario, "2026-09-21T19:00:00+00:00"))
        self.assertFalse(_strict_pit_provenance_ok({}, "2026-09-21T19:00:00+00:00"))


    def test_primary_production_loader_excludes_fallback_rows(self):
        import tempfile
        import sqlite3
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "predictions.db"
            feature = {k: 0.0 for k in __import__("feature_schema").FEATURES}
            rows = []
            for i, mode in enumerate(("binance_primary", "coinbase_fallback")):
                scenario = {
                    "decision_time_utc": "2026-09-21T19:00:00+00:00",
                    "production_mode": mode,
                    "provenance": {
                        "available_at": "2026-09-21T18:59:59+00:00",
                        "retrieved_at": "2026-09-21T19:00:00+00:00",
                        "prediction_cutoff": "2026-09-21T19:00:00+00:00",
                        "sources": {
                            "binance_futures": {"status":"ok","available_at":"2026-09-21T18:59:59+00:00","retrieved_at":"2026-09-21T19:00:00+00:00","prediction_cutoff":"2026-09-21T19:00:00+00:00"},
                            "binance_depth": {"status":"ok","available_at":"2026-09-21T18:59:59+00:00","retrieved_at":"2026-09-21T19:00:00+00:00","prediction_cutoff":"2026-09-21T19:00:00+00:00"},
                            "binance_taker": {"status":"ok","available_at":"2026-09-21T18:59:59+00:00","retrieved_at":"2026-09-21T19:00:00+00:00","prediction_cutoff":"2026-09-21T19:00:00+00:00"},
                            "binance_premium": {"status":"ok","available_at":"2026-09-21T18:59:59+00:00","retrieved_at":"2026-09-21T19:00:00+00:00","prediction_cutoff":"2026-09-21T19:00:00+00:00"},
                        },
                    },
                }
                rows.append((i+1,"2026-09-21T19:00:00+00:00","2026-09-21T19:05:00+00:00","2026-09-21T19:10:00+00:00",json.dumps(feature),"UP",0.8,0.1,0.1,"bootstrap.bootstrap_rf",json.dumps(scenario)))
            with sqlite3.connect(db) as con:
                con.execute("CREATE TABLE predictions (prediction_id INTEGER, created_at_utc TEXT, target_5m TEXT, target_10m TEXT, feature_json TEXT, actual_direction_5m TEXT, p_up_5m REAL, p_down_5m REAL, p_flat_5m REAL, model_version TEXT, scenario_json TEXT)")
                con.executemany("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
            with patch("model_compare.DB", db):
                got = load_primary_production_strict_rows("5m")
            self.assertEqual(len(got), 1)
            self.assertEqual(got[0]["production_mode"], "binance_primary")

    def test_prediction_must_precede_target_strictly(self):
        self.assertTrue(prediction_precedes_target(
            '2026-09-21T18:55:58+00:00',
            '2026-09-21T19:00:00+00:00',
        ))
        self.assertFalse(prediction_precedes_target(
            '2026-09-21T19:00:00+00:00',
            '2026-09-21T19:00:00+00:00',
        ))
        self.assertFalse(prediction_precedes_target(
            '2026-09-21T19:00:01+00:00',
            '2026-09-21T19:00:00+00:00',
        ))
        self.assertFalse(prediction_precedes_target('bad', '2026-09-21T19:00:00+00:00'))


if __name__ == '__main__':
    unittest.main()
