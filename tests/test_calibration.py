import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import calibration  # noqa: E402


class _FakeResult:
    def __init__(self, rows=None):
        self.rows = rows or []

    def fetchall(self):
        return self.rows


class _FakeConnection:
    def __init__(self, rows=None):
        self.sql = None
        self.params = None
        self.rows = rows or []

    def execute(self, sql, params):
        self.sql = sql
        self.params = params
        return _FakeResult(self.rows)


class TestCalibration(unittest.TestCase):


    def test_frozen_model_temperature_requires_generation_match(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            (model_dir / "10m.json").write_text(json.dumps({
                "model_version": "bootstrap.bootstrap_rf",
                "temperature": 0.8,
            }), encoding="utf-8")
            with patch.object(calibration_module, "MODEL_DIR", model_dir):
                self.assertEqual(
                    calibration_module._frozen_model_temperature("10m", "bootstrap.bootstrap_rf"),
                    0.8,
                )
                self.assertIsNone(
                    calibration_module._frozen_model_temperature("10m", "other_generation"),
                )

    def test_zero_settled_preserves_generation_matched_calibration(self):
        import json
        import sqlite3
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "models"
            model_dir.mkdir()
            cached = {
                "horizon": "10m",
                "temperature": 0.8,
                "n_settled": 2153,
                "model_version": "bootstrap.bootstrap_rf",
            }
            (model_dir / "10m.calibration.json").write_text(json.dumps(cached), encoding="utf-8")
            db = root / "predictions.db"
            with sqlite3.connect(db) as con:
                con.execute("CREATE TABLE model_registry (horizon TEXT PRIMARY KEY, production_version TEXT, updated_at_utc TEXT)")
                con.execute("CREATE TABLE predictions (actual_direction_5m TEXT, actual_direction_10m TEXT, model_version TEXT, p_up_5m REAL, p_down_5m REAL, p_flat_5m REAL, p_up_10m REAL, p_down_10m REAL, p_flat_10m REAL, scenario_json TEXT)")
                con.execute("INSERT INTO model_registry VALUES ('5m','bootstrap.bootstrap_rf','2026-09-22T00:00:00+00:00')")
                con.execute("INSERT INTO model_registry VALUES ('10m','bootstrap.bootstrap_rf','2026-09-22T00:00:00+00:00')")
            with patch.object(calibration_module, "DB", db), patch.object(calibration_module, "MODEL_DIR", model_dir):
                # Helper uses the same artifact shape as production and must return
                # true without requiring fresh rows.
                obj = calibration_module._calibration_state(model_dir / "10m.calibration.json")
                self.assertTrue(calibration_module._can_reuse_cached_calibration(obj, "10m", "bootstrap.bootstrap_rf", 2153))

    def test_settled_rows_uses_existing_horizon_suffix(self):
        con = _FakeConnection()
        calibration._settled_rows(
            con,
            '5m',
            'actual_direction_5m',
            'bootstrap.example.v1',
        )
        self.assertIn('p_up_5m', con.sql)
        self.assertIn('p_down_5m', con.sql)
        self.assertIn('p_flat_5m', con.sql)
        self.assertNotIn('p_up_5mm', con.sql)
        self.assertNotIn('p_down_5mm', con.sql)
        self.assertNotIn('p_flat_5mm', con.sql)
        self.assertEqual(con.params, ('5m:bootstrap.example.v1|%',))

    def test_settled_rows_excludes_fallback_venue_from_production_calibration(self):
        rows = [
            (0.80, 0.10, 0.10, 'UP', '{"production_mode":"coinbase_fallback"}'),
            (0.70, 0.20, 0.10, 'UP', '{"production_mode":"binance_primary"}'),
        ]
        con = _FakeConnection(rows)
        out = calibration._settled_rows(
            con, '5m', 'actual_direction_5m', 'bootstrap.example.v1'
        )
        self.assertEqual(out, [(0.70, 0.20, 0.10, 'UP')])

    def test_settled_rows_uses_10m_schema(self):
        con = _FakeConnection()
        calibration._settled_rows(
            con,
            '10m',
            'actual_direction_10m',
            'bootstrap.example.v1',
        )
        self.assertIn('p_up_10m', con.sql)
        self.assertNotIn('p_up_10mm', con.sql)
        self.assertEqual(con.params, ('10m:bootstrap.example.v1|%',))

    def test_invalid_horizon_returns_no_rows_without_query(self):
        con = _FakeConnection()
        rows = calibration._settled_rows(
            con,
            '5',
            'actual_direction_5m',
            'bootstrap.example.v1',
        )
        self.assertEqual(rows, [])
        self.assertIsNone(con.sql)

    def test_cache_reuse_requires_same_generation_and_sample_count(self):
        cached = {
            'horizon': '5m',
            'model_version': 'bootstrap.example.v1',
            'n_settled': 500,
            'temperature': 1.1,
        }
        self.assertTrue(calibration._can_reuse_cached_calibration(cached, '5m', 'bootstrap.example.v1', 500))
        self.assertFalse(calibration._can_reuse_cached_calibration(cached, '10m', 'bootstrap.example.v1', 500))
        self.assertFalse(calibration._can_reuse_cached_calibration(cached, '5m', 'bootstrap.example.v2', 500))
        self.assertFalse(calibration._can_reuse_cached_calibration(cached, '5m', 'bootstrap.example.v1', 501))

    def test_uses_canonical_root_model_directory(self):
        self.assertEqual(calibration.MODEL_DIR, ROOT / 'models')
        self.assertNotEqual(calibration.MODEL_DIR, ROOT / 'data' / 'models')

    def test_cache_reuse_rejects_malformed_temperature(self):
        cached = {
            'horizon': '5m',
            'model_version': 'bootstrap.example.v1',
            'n_settled': 500,
            'temperature': 'not-a-number',
        }
        self.assertFalse(calibration._can_reuse_cached_calibration(cached, '5m', 'bootstrap.example.v1', 500))


if __name__ == '__main__':
    unittest.main()
