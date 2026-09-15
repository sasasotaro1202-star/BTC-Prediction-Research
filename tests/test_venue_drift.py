import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import venue_drift


class TestVenueDrift(unittest.TestCase):
    def test_report_counts_sources_and_gaps_without_model_effect(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / 'predictions.db'
            con = sqlite3.connect(db)
            con.execute('CREATE TABLE predictions (prediction_id INTEGER PRIMARY KEY, scenario_json TEXT)')
            scenarios = [
                {'data_quality': {'price_feature_fallback': 'coinbase', 'bybit_series': False}, 'microstructure': {'cross_exchange_gap': 0.001}},
                {'data_quality': {'price_feature_fallback': 'bybit', 'bybit_series': True}, 'microstructure': {'cross_exchange_gap': -0.002}},
                {'data_quality': {'price_feature_fallback': 'coinbase'}, 'microstructure': {'cross_exchange_gap': 0.003}},
            ]
            for i, obj in enumerate(scenarios, 1):
                con.execute('INSERT INTO predictions VALUES (?,?)', (i, json.dumps(obj)))
            con.commit(); con.close()
            with patch.object(venue_drift, 'OUT', Path(td) / 'venue_drift.json'):
                report = venue_drift.build_report(db, window=10)
            self.assertTrue(report['ok'])
            self.assertEqual(report['source_counts'], {'bybit': 1, 'coinbase': 2})
            self.assertEqual(report['coverage_counts']['bybit_series_available'], 1)
            self.assertEqual(report['cross_exchange_gap']['n'], 3)
            self.assertEqual(report['policy'], 'monitor_only_no_model_input_no_promotion_effect')

    def test_empty_history_is_not_reported_as_healthy(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / 'predictions.db'
            con = sqlite3.connect(db)
            con.execute('CREATE TABLE predictions (prediction_id INTEGER PRIMARY KEY, scenario_json TEXT)')
            con.commit(); con.close()
            with patch.object(venue_drift, 'OUT', Path(td) / 'venue_drift.json'):
                report = venue_drift.build_report(db)
            self.assertFalse(report['ok'])
            self.assertEqual(report['window'], 0)


if __name__ == '__main__':
    unittest.main()
