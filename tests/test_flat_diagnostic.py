import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import flat_diagnostic


class TestFlatDiagnostic(unittest.TestCase):
    def test_locates_flat_disappearance_by_stage(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / 'predictions.db'
            con = sqlite3.connect(db)
            con.execute('''CREATE TABLE predictions (
                prediction_id INTEGER PRIMARY KEY,
                actual_direction_5m TEXT,
                p_down_5m REAL, p_flat_5m REAL, p_up_5m REAL,
                actual_direction_10m TEXT,
                p_down_10m REAL, p_flat_10m REAL, p_up_10m REAL,
                scenario_json TEXT)''')
            scenario = {'components': {
                'model_raw_5m': [0.2, 0.6, 0.2],
                'structural_5m': [0.7, 0.1, 0.2],
                'fused_raw_5m': [0.55, 0.2, 0.25],
                'calibrated_5m': [0.5, 0.25, 0.25],
                'model_raw_10m': [0.2, 0.5, 0.3],
                'structural_10m': [0.4, 0.2, 0.4],
                'fused_raw_10m': [0.4, 0.2, 0.4],
                'calibrated_10m': [0.35, 0.25, 0.4],
            }}
            con.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?)',
                         (1, 'FLAT', .50, .25, .25, 'FLAT', .35, .25, .40, json.dumps(scenario)))
            con.commit(); con.close()
            with patch.object(flat_diagnostic, 'OUT', Path(td) / 'flat_diagnostic.json'):
                report = flat_diagnostic.build_report(db, window=10)
            self.assertTrue(report['ok'])
            self.assertEqual(report['counts']['5m']['model_raw'], {'FLAT': 1})
            self.assertEqual(report['counts']['5m']['final'], {'DOWN': 1})
            self.assertEqual(report['counts']['10m']['model_raw'], {'FLAT': 1})
            summary = report['probability_summary']['10m']['model_raw']
            self.assertEqual(summary['flat_argmax_rate'], 1.0)
            self.assertAlmostEqual(summary['mean_probability']['FLAT'], 0.5)
            self.assertAlmostEqual(summary['flat_margin_mean'], 0.2)
            self.assertEqual(summary['flat_within_0.02_rate'], 1.0)
            self.assertEqual(summary['flat_above_0.30_rate'], 1.0)
            self.assertEqual(report['policy'], 'diagnostic_only_no_model_input_no_promotion_effect')

    def test_empty_history_is_not_healthy(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / 'predictions.db'
            con = sqlite3.connect(db)
            con.execute('CREATE TABLE predictions (prediction_id INTEGER PRIMARY KEY, actual_direction_5m TEXT, p_down_5m REAL, p_flat_5m REAL, p_up_5m REAL, actual_direction_10m TEXT, p_down_10m REAL, p_flat_10m REAL, p_up_10m REAL, scenario_json TEXT)')
            con.commit(); con.close()
            with patch.object(flat_diagnostic, 'OUT', Path(td) / 'flat_diagnostic.json'):
                report = flat_diagnostic.build_report(db)
            self.assertFalse(report['ok'])


if __name__ == '__main__':
    unittest.main()
