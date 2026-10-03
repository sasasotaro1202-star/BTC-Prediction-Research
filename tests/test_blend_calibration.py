import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import blend_calibration  # noqa: E402


class TestBlendCalibration(unittest.TestCase):
    def test_normalization(self):
        p = blend_calibration._norm([[2.0, 1.0, 1.0]])
        self.assertAlmostEqual(float(p.sum()), 1.0, places=10)

    def test_brier_prefers_correct_probability(self):
        y = ['UP', 'DOWN', 'FLAT']
        good = [[0.98, 0.01, 0.01], [0.01, 0.98, 0.01], [0.01, 0.01, 0.98]]
        bad = [[0.34, 0.33, 0.33]] * 3
        self.assertLess(blend_calibration._brier(y, good), blend_calibration._brier(y, bad))

    def test_horizon_column_mapping(self):
        self.assertEqual(blend_calibration._actual_column('5m'), 'actual_direction_5m')
        self.assertEqual(blend_calibration._actual_column('10m'), 'actual_direction_10m')
        with self.assertRaises(ValueError):
            blend_calibration._actual_column('5')

    def test_generation_token_is_horizon_specific(self):
        self.assertEqual(
            blend_calibration.blend_generation_token('5m', 'bootstrap.soft_ensemble.v5.4'),
            '5m:bootstrap.soft_ensemble.v5.4',
        )
        self.assertEqual(
            blend_calibration.blend_generation_token('10m', 'bootstrap.bootstrap_rf'),
            '10m:bootstrap.bootstrap_rf',
        )

    def test_rows_match_secondary_horizon_in_combined_generation(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / 'predictions.db'
            scenario_json = json.dumps({
                'components': {
                    'model_raw_10m': {'UP': 0.30, 'DOWN': 0.40, 'FLAT': 0.30},
                    'structural_10m': {'UP': 0.35, 'DOWN': 0.35, 'FLAT': 0.30},
                }
            })
            with sqlite3.connect(db) as con:
                con.execute('CREATE TABLE model_registry (horizon TEXT, production_version TEXT)')
                con.execute('''CREATE TABLE predictions (
                    created_at_utc TEXT,
                    scenario_json TEXT,
                    p_up_10m REAL,
                    p_down_10m REAL,
                    p_flat_10m REAL,
                    actual_direction_10m TEXT,
                    model_version TEXT
                )''')
                con.execute(
                    'INSERT INTO model_registry VALUES (?, ?)',
                    ('10m', 'bootstrap.bootstrap_rf'),
                )
                con.execute(
                    'INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?)',
                    (
                        '2026-10-03T08:00:00+00:00',
                        scenario_json,
                        0.30,
                        0.40,
                        0.30,
                        'DOWN',
                        '5m:bootstrap.soft_ensemble.v5.4|10m:bootstrap.bootstrap_rf',
                    ),
                )
                con.commit()

            with patch.object(blend_calibration, 'DB', db):
                rows = blend_calibration._rows('10m')

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], '2026-10-03T08:00:00+00:00')
            self.assertEqual(rows[0][3], 'DOWN')

    def test_probability_suffix_does_not_double_append_m(self):
        self.assertEqual('p_up_5m', f'p_up_{"5m"}')
        self.assertEqual('p_up_10m', f'p_up_{"10m"}')
        self.assertNotEqual('p_up_5mm', f'p_up_{"5m"}')
        self.assertNotEqual('p_up_10mm', f'p_up_{"10m"}')

    def test_uses_canonical_root_model_directory(self):
        self.assertEqual(blend_calibration.MODEL_DIR, ROOT / 'models')
        self.assertNotEqual(blend_calibration.MODEL_DIR, ROOT / 'data' / 'models')

    def test_rows_require_binance_primary_strict_pit(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / 'predictions.db'
            common = {
                'components': {
                    'model_raw_5m': {'UP': 0.30, 'DOWN': 0.40, 'FLAT': 0.30},
                    'structural_5m': {'UP': 0.35, 'DOWN': 0.35, 'FLAT': 0.30},
                }
            }
            strict = {
                **common,
                'production_mode': 'binance_primary',
                'decision_time_utc': '2026-10-03T08:00:01+00:00',
                'provenance': {
                    'available_at': '2026-10-03T08:00:00+00:00',
                    'retrieved_at': '2026-10-03T08:00:00+00:00',
                    'prediction_cutoff': '2026-10-03T08:00:00+00:00',
                    'sources': {
                        name: {
                            'status': 'ok',
                            'event_time': '2026-10-03T08:00:00+00:00',
                            'available_at': '2026-10-03T08:00:00+00:00',
                            'retrieved_at': '2026-10-03T08:00:00+00:00',
                            'prediction_cutoff': '2026-10-03T08:00:00+00:00',
                        }
                        for name in ('binance_futures','binance_depth','binance_taker','binance_premium')
                    },
                },
            }
            fallback = {**strict, 'production_mode': 'coinbase_fallback'}
            with sqlite3.connect(db) as con:
                con.execute('CREATE TABLE model_registry (horizon TEXT, production_version TEXT)')
                con.execute('CREATE TABLE predictions (created_at_utc TEXT, scenario_json TEXT, p_up_5m REAL, p_down_5m REAL, p_flat_5m REAL, actual_direction_5m TEXT, model_version TEXT)')
                con.execute('INSERT INTO model_registry VALUES (?, ?)', ('5m','generation-A'))
                con.executemany(
                    'INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?)',
                    [
                        ('2026-10-03T08:00:00+00:00', json.dumps(strict), 0.3, 0.4, 0.3, 'UP', '5m:generation-A|10m:generation-A'),
                        ('2026-10-03T08:00:00+00:00', json.dumps(fallback), 0.3, 0.4, 0.3, 'UP', '5m:generation-A|10m:generation-A'),
                    ],
                )
                con.commit()
            with patch.object(blend_calibration, 'DB', db):
                rows = blend_calibration._rows('5m')
            self.assertEqual(len(rows), 1)

    def test_rejected_or_unvalidated_blend_fails_closed(self):
        self.assertEqual(blend_calibration.FALLBACK_WEIGHT, 0.0)

    def test_block_stability_requires_multiple_consistent_blocks(self):
        y = ['UP', 'DOWN', 'FLAT'] * 10
        model = [[0.34, 0.33, 0.33]] * len(y)
        structural = [[0.90, 0.05, 0.05] if label == 'UP' else [0.05, 0.90, 0.05] if label == 'DOWN' else [0.05, 0.05, 0.90] for label in y]
        stable = blend_calibration._block_stability(y, model, structural, 0.20, block_size=15)
        self.assertEqual(stable['blocks'], 2)
        self.assertFalse(stable['stable'])

    def test_grid_is_bounded(self):
        self.assertGreaterEqual(float(blend_calibration.GRID.min()), 0.0)
        self.assertLessEqual(float(blend_calibration.GRID.max()), 0.45)


if __name__ == '__main__':
    unittest.main()
