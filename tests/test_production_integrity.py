import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from src import production_integrity


class TestProductionIntegrityDeferred(unittest.TestCase):
    def _db(self, path: Path, created: datetime) -> None:
        with sqlite3.connect(path) as con:
            con.execute(
                """CREATE TABLE predictions (
                    prediction_id INTEGER,
                    created_at_utc TEXT,
                    base_price REAL,
                    p_up_5m REAL,
                    p_down_5m REAL,
                    p_flat_5m REAL,
                    p_up_10m REAL,
                    p_down_10m REAL,
                    p_flat_10m REAL,
                    model_version TEXT,
                    feature_json TEXT,
                    scenario_json TEXT
                )"""
            )
            scenario = {
                "data_quality": {"binance_futures": "error:HTTPError:451"},
                "policy": "safe_degraded_no_directional_claim",
            }
            con.execute(
                "INSERT INTO predictions VALUES (1,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    created.isoformat(),
                    0.0,
                    0.2, 0.3, 0.5,
                    0.2, 0.3, 0.5,
                    "DEGRADED_NO_FRESH_DATA",
                    "{}",
                    json.dumps(scenario),
                ),
            )
            con.commit()


    def test_calibration_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "models"
            model_dir.mkdir()
            model = model_dir / "5m.joblib"
            model.write_bytes(b"model")
            (model_dir / "5m.calibration.json").write_text(
                json.dumps({
                    "horizon": "5m",
                    "model_version": "generation-A",
                    "model_sha256": "a" * 64,
                    "n_settled": 300,
                    "temperature": 1.0,
                }),
                encoding="utf-8",
            )
            with patch.object(production_integrity, "MODEL_DIR", model_dir):
                with self.assertRaisesRegex(RuntimeError, "calibration artifact hash mismatch"):
                    production_integrity.check_calibration_bindings(
                        "5m", "generation-A", "b" * 64
                    )

    def test_calibration_hash_missing_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "models"
            model_dir.mkdir()
            (model_dir / "5m.calibration.json").write_text(
                json.dumps({
                    "horizon": "5m",
                    "model_version": "generation-A",
                    "n_settled": 400,
                    "temperature": 1.0,
                }),
                encoding="utf-8",
            )
            with patch.object(production_integrity, "MODEL_DIR", model_dir):
                with self.assertRaisesRegex(RuntimeError, "calibration artifact hash missing"):
                    production_integrity.check_calibration_bindings(
                        "5m", "generation-A", "b" * 64
                    )

    def test_unvalidated_blend_does_not_require_hash_but_generation_is_bound(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "models"
            model_dir.mkdir()
            (model_dir / "5m.blend.json").write_text(
                json.dumps({
                    "horizon": "5m",
                    "model_version": "generation-A",
                    "status": "rejected",
                    "base_weight": 0.0,
                    "n": 488,
                }),
                encoding="utf-8",
            )
            with patch.object(production_integrity, "MODEL_DIR", model_dir):
                result = production_integrity.check_blend_binding(
                    "5m", "generation-A", "b" * 64
                )
            self.assertEqual(result["status"], "rejected")

    def test_stale_prediction_without_fresh_deferred_status_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "predictions.db"
            self._db(db, datetime.now(timezone.utc) - timedelta(hours=2))
            with patch.object(production_integrity, "DB", db),                  patch.object(production_integrity, "LIVE_STATUS", root / "live_cycle_status.json"),                  patch.dict(
                     production_integrity.os.environ,
                     {
                         "BTC_INTEGRITY_REQUIRE_FRESH": "true",
                         "BTC_INTEGRITY_MAX_PREDICTION_AGE_SECONDS": "900",
                     },
                     clear=False,
                 ):
                with self.assertRaisesRegex(RuntimeError, "stale or future-dated"):
                    production_integrity.check_db()

    def test_archive_mode_allows_stale_snapshot_but_rejects_future_timestamp(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "predictions.db"
            stale = datetime.now(timezone.utc) - timedelta(hours=2)
            self._db(db, stale)
            with patch.object(production_integrity, "DB", db),                  patch.object(production_integrity, "LIVE_STATUS", root / "live_cycle_status.json"),                  patch.dict(
                     production_integrity.os.environ,
                     {
                         "BTC_INTEGRITY_REQUIRE_FRESH": "false",
                         "BTC_INTEGRITY_MAX_PREDICTION_AGE_SECONDS": "900",
                     },
                     clear=False,
                 ):
                result = production_integrity.check_db()
            self.assertTrue(result["prediction_ok"])
            self.assertFalse(result["freshness_required"])

            future_db = root / "future.db"
            self._db(future_db, datetime.now(timezone.utc) + timedelta(seconds=120))
            with patch.object(production_integrity, "DB", future_db),                  patch.dict(
                     production_integrity.os.environ,
                     {
                         "BTC_INTEGRITY_REQUIRE_FRESH": "false",
                         "BTC_INTEGRITY_MAX_PREDICTION_AGE_SECONDS": "900",
                     },
                     clear=False,
                 ):
                with self.assertRaisesRegex(RuntimeError, "future-dated"):
                    production_integrity.check_db()

    def test_fresh_deferred_status_allows_safe_stale_state(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "predictions.db"
            self._db(db, datetime.now(timezone.utc) - timedelta(hours=2))
            status = root / "live_cycle_status.json"
            status.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "mode": "deferred",
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(production_integrity, "DB", db),                  patch.object(production_integrity, "LIVE_STATUS", status),                  patch.dict(
                     production_integrity.os.environ,
                     {
                         "BTC_INTEGRITY_REQUIRE_FRESH": "true",
                         "BTC_INTEGRITY_MAX_PREDICTION_AGE_SECONDS": "900",
                     },
                     clear=False,
                 ):
                result = production_integrity.check_db()
            self.assertTrue(result["prediction_ok"])
            self.assertTrue(result["safe_deferred"])


if __name__ == "__main__":
    unittest.main()
