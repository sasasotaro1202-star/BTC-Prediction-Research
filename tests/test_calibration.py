import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import calibration  # noqa: E402


class _FakeResult:
    def fetchall(self):
        return []


class _FakeConnection:
    def __init__(self):
        self.sql = None
        self.params = None

    def execute(self, sql, params):
        self.sql = sql
        self.params = params
        return _FakeResult()


class TestCalibration(unittest.TestCase):
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
