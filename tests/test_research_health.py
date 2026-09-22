import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import research_health


class ResearchHealthDuplicateTests(unittest.TestCase):
    def test_duplicate_prediction_events_are_not_healthy(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "predictions.db"
            con = sqlite3.connect(db)
            con.execute("""CREATE TABLE predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_utc TEXT, target_5m TEXT, target_10m TEXT, feature_json TEXT,
                p_up_5m REAL, p_down_5m REAL, p_flat_5m REAL,
                p_up_10m REAL, p_down_10m REAL, p_flat_10m REAL,
                model_version TEXT, scenario_json TEXT,
                actual_direction_5m TEXT, actual_direction_10m TEXT
            )""")
            row = ("2026-09-15T10:00:00+00:00","2026-09-15T10:05:00+00:00","2026-09-15T10:10:00+00:00","{\"a\":1}")
            vals = row + (0.3,0.3,0.4,0.3,0.3,0.4,"v1","{}", "UP","DOWN")
            con.execute("""INSERT INTO predictions(
                created_at_utc,target_5m,target_10m,feature_json,
                p_up_5m,p_down_5m,p_flat_5m,p_up_10m,p_down_10m,p_flat_10m,
                model_version,scenario_json,actual_direction_5m,actual_direction_10m
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", vals)
            vals2 = row + (0.2,0.4,0.4,0.3,0.3,0.4,"v2","{\"retry\":true}", "UP","DOWN")
            con.execute("""INSERT INTO predictions(
                created_at_utc,target_5m,target_10m,feature_json,
                p_up_5m,p_down_5m,p_flat_5m,p_up_10m,p_down_10m,p_flat_10m,
                model_version,scenario_json,actual_direction_5m,actual_direction_10m
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", vals2)
            con.commit()
            self.assertEqual(research_health.count_duplicate_prediction_events(con), 1)
            con.close()


if __name__ == "__main__":
    unittest.main()
