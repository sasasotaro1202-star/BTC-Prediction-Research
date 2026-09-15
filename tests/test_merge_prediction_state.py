import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'merge_prediction_state.py'


PREDICTION_COLUMNS = (
    'created_at_utc,target_5m,target_10m,base_price,'
    'p_up_5m,p_down_5m,p_flat_5m,p_up_10m,p_down_10m,p_flat_10m,'
    'model_version,feature_json,scenario_json,actual_price_5m,actual_direction_5m,'
    'correct_5m,settled_5m_at_utc,actual_price_10m,actual_direction_10m,'
    'correct_10m,settled_10m_at_utc'
)


def make_db(path, rows):
    con = sqlite3.connect(path)
    con.execute(f'''CREATE TABLE predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at_utc TEXT NOT NULL, target_5m TEXT NOT NULL, target_10m TEXT NOT NULL,
        base_price REAL NOT NULL, p_up_5m REAL NOT NULL, p_down_5m REAL NOT NULL,
        p_flat_5m REAL NOT NULL, p_up_10m REAL NOT NULL, p_down_10m REAL NOT NULL,
        p_flat_10m REAL NOT NULL, model_version TEXT NOT NULL, feature_json TEXT NOT NULL,
        scenario_json TEXT NOT NULL, actual_price_5m REAL, actual_direction_5m TEXT,
        correct_5m INTEGER, settled_5m_at_utc TEXT, actual_price_10m REAL,
        actual_direction_10m TEXT, correct_10m INTEGER, settled_10m_at_utc TEXT
    )''')
    for row in rows:
        con.execute(f'INSERT INTO predictions ({PREDICTION_COLUMNS}) VALUES ({",".join("?" for _ in PREDICTION_COLUMNS.split(","))})', row)
    con.commit(); con.close()


class TestMergePredictionState(unittest.TestCase):
    def test_settled_and_unsettled_same_prediction_merge_once(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'target.db'; local = Path(td) / 'local.db'
            immutable = ('2026-09-15T10:00:00+00:00', '2026-09-15T10:05:00+00:00',
                         '2026-09-15T10:10:00+00:00', 100.0, .4, .3, .3, .4, .3, .3,
                         'v1', '{}', '{}')
            unsettled = immutable + (None, None, None, None, None, None, None, None)
            settled = immutable + (101.0, 'UP', 1, '2026-09-15T10:05:01+00:00', None, None, None, None)
            make_db(target, [unsettled]); make_db(local, [settled])
            subprocess.run([sys.executable, str(SCRIPT), str(local), str(target)], check=True)
            con = sqlite3.connect(target)
            rows = con.execute('SELECT actual_price_5m, actual_direction_5m FROM predictions').fetchall()
            con.close()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0], (None, None))

    def test_distinct_prediction_events_are_not_collapsed(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'target.db'; local = Path(td) / 'local.db'
            common = ('2026-09-15T10:00:00+00:00', '2026-09-15T10:05:00+00:00',
                      '2026-09-15T10:10:00+00:00', 100.0, .4, .3, .3, .4, .3, .3,
                      'v1', '{}', '{}')
            row_a = common + (None, None, None, None, None, None, None, None)
            row_b = ('2026-09-15T10:00:01+00:00',) + common[1:] + (None, None, None, None, None, None, None, None)
            make_db(target, [row_a]); make_db(local, [row_b])
            subprocess.run([sys.executable, str(SCRIPT), str(local), str(target)], check=True)
            con = sqlite3.connect(target)
            count = con.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
            con.close()
            self.assertEqual(count, 2)


if __name__ == '__main__':
    unittest.main()
