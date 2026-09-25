import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from scripts.merge_prediction_state import expected_compacted_event_count


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
        con.execute(
            f'INSERT INTO predictions ({PREDICTION_COLUMNS}) VALUES '
            f'({",".join("?" for _ in PREDICTION_COLUMNS.split(","))})',
            row,
        )
    con.commit()
    con.close()


def add_registry(path, version, updated_at):
    con = sqlite3.connect(path)
    con.execute('''CREATE TABLE model_registry (
        horizon TEXT PRIMARY KEY,
        production_version TEXT NOT NULL,
        updated_at_utc TEXT NOT NULL
    )''')
    con.executemany(
        'INSERT INTO model_registry(horizon,production_version,updated_at_utc) VALUES(?,?,?)',
        [('5m', version, updated_at), ('10m', version, updated_at)],
    )
    con.commit()
    con.close()


class TestMergePredictionState(unittest.TestCase):
    def test_settled_and_unsettled_same_prediction_merge_once_and_reconcile(self):
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
            rows = con.execute('SELECT actual_price_5m, actual_direction_5m, correct_5m FROM predictions').fetchall()
            con.close()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0], (101.0, 'UP', 1))

    def test_existing_target_settlement_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'target.db'; local = Path(td) / 'local.db'
            immutable = ('2026-09-15T10:00:00+00:00', '2026-09-15T10:05:00+00:00',
                         '2026-09-15T10:10:00+00:00', 100.0, .4, .3, .3, .4, .3, .3,
                         'v1', '{}', '{}')
            target_row = immutable + (102.0, 'UP', 1, '2026-09-15T10:05:02+00:00', None, None, None, None)
            local_row = immutable + (101.0, 'UP', 1, '2026-09-15T10:05:01+00:00', None, None, None, None)
            make_db(target, [target_row]); make_db(local, [local_row])
            subprocess.run([sys.executable, str(SCRIPT), str(local), str(target)], check=True)
            con = sqlite3.connect(target)
            row = con.execute('SELECT actual_price_5m, settled_5m_at_utc FROM predictions').fetchone()
            con.close()
            self.assertEqual(row, (102.0, '2026-09-15T10:05:02+00:00'))

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

    def test_brand_new_settled_prediction_preserves_settlement_fields(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'target.db'; local = Path(td) / 'local.db'
            immutable = ('2026-09-15T10:01:00+00:00', '2026-09-15T10:06:00+00:00',
                         '2026-09-15T10:11:00+00:00', 100.0, .4, .3, .3, .4, .3, .3,
                         'v1', '{}', '{}')
            settled = immutable + (101.0, 'UP', 1, '2026-09-15T10:06:01+00:00', 102.0, 'UP', 1, '2026-09-15T10:11:01+00:00')
            make_db(target, []); make_db(local, [settled])
            subprocess.run([sys.executable, str(SCRIPT), str(local), str(target)], check=True)
            con = sqlite3.connect(target)
            row = con.execute('''SELECT actual_price_5m, actual_direction_5m, correct_5m,
                                        settled_5m_at_utc, actual_price_10m, actual_direction_10m,
                                        correct_10m, settled_10m_at_utc
                                 FROM predictions''').fetchone()
            con.close()
            self.assertEqual(row, (101.0, 'UP', 1, '2026-09-15T10:06:01+00:00', 102.0, 'UP', 1, '2026-09-15T10:11:01+00:00'))

    def test_repeated_merge_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'target.db'; local = Path(td) / 'local.db'
            immutable = ('2026-09-15T10:02:00+00:00', '2026-09-15T10:07:00+00:00',
                         '2026-09-15T10:12:00+00:00', 100.0, .4, .3, .3, .4, .3, .3,
                         'v1', '{}', '{}')
            settled = immutable + (101.0, 'UP', 1, '2026-09-15T10:07:01+00:00', 102.0, 'UP', 1, '2026-09-15T10:12:01+00:00')
            make_db(target, []); make_db(local, [settled])
            for _ in range(2):
                subprocess.run([sys.executable, str(SCRIPT), str(local), str(target)], check=True)
            con = sqlite3.connect(target)
            count = con.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
            row = con.execute('SELECT actual_price_5m, actual_price_10m FROM predictions').fetchone()
            con.close()
            self.assertEqual(count, 1)
            self.assertEqual(row, (101.0, 102.0))


    def test_compact_exact_duplicate_preserves_settlement(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'target.db'
            immutable = ('2026-09-15T10:00:00+00:00', '2026-09-15T10:05:00+00:00',
                         '2026-09-15T10:10:00+00:00', 100.0, .4, .3, .3, .4, .3, .3,
                         'v1', '{"ret_1m":0.1}', '{}')
            unsettled = immutable + (None, None, None, None, None, None, None, None)
            settled = immutable + (101.0, 'UP', 1, '2026-09-15T10:05:01+00:00', 102.0, 'UP', 1, '2026-09-15T10:10:01+00:00')
            make_db(target, [unsettled, settled])
            subprocess.run([sys.executable, str(SCRIPT), '--compact', str(target)], check=True)
            con = sqlite3.connect(target)
            rows = con.execute(
                'SELECT COUNT(*), actual_price_5m, actual_price_10m FROM predictions'
            ).fetchone()
            con.close()
            self.assertEqual(rows, (1, 101.0, 102.0))

    def test_compact_conflicting_settlement_quarantines_horizon_without_guessing(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'target.db'
            immutable = ('2026-09-15T10:03:00+00:00', '2026-09-15T10:08:00+00:00',
                         '2026-09-15T10:13:00+00:00', 100.0, .4, .3, .3, .4, .3, .3,
                         'v1', '{"ret_1m":0.1}', '{}')
            settled_up = immutable + (101.0, 'UP', 1, '2026-09-15T10:08:01+00:00', None, None, None, None)
            settled_flat = immutable + (100.02, 'FLAT', 0, '2026-09-15T10:08:02+00:00', None, None, None, None)
            make_db(target, [settled_up, settled_flat])

            subprocess.run([sys.executable, str(SCRIPT), '--compact', str(target)], check=True)

            con = sqlite3.connect(target)
            row = con.execute('''SELECT COUNT(*), actual_price_5m, actual_direction_5m, correct_5m,
                                      actual_price_10m, actual_direction_10m, correct_10m
                               FROM predictions''').fetchone()
            conflict = con.execute('''SELECT identity_sha256, resolution, conflicts_json
                                      FROM prediction_settlement_conflicts''').fetchone()
            con.close()

            self.assertEqual(row, (1, None, None, None, None, None, None))
            self.assertIsNotNone(conflict)
            self.assertEqual(conflict[1], 'QUARANTINED_CONFLICTING_SETTLEMENT')
            self.assertIn('actual_direction_5m', conflict[2])
            self.assertIn('actual_price_5m', conflict[2])

    def test_compact_keeps_distinct_model_versions_and_probabilities(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'target.db'
            common = ('2026-09-15T10:00:00+00:00', '2026-09-15T10:05:00+00:00',
                      '2026-09-15T10:10:00+00:00', 100.0, .4, .3, .3, .4, .3, .3,
                      'v1', '{"ret_1m":0.1}', '{}')
            row_same = common + (None, None, None, None, None, None, None, None)
            row_model = ('2026-09-15T10:00:00+00:00', common[1], common[2], 100.0,
                         .5, .2, .3, .5, .2, .3, 'v2', '{"ret_1m":0.1}', '{}',
                         None, None, None, None, None, None, None, None)
            make_db(target, [row_same, row_model])
            subprocess.run([sys.executable, str(SCRIPT), '--compact', str(target)], check=True)
            con = sqlite3.connect(target)
            count = con.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
            con.close()
            self.assertEqual(count, 2)


    def test_expected_compacted_count_matches_exact_identity_groups(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'target.db'
            immutable = ('2026-09-15T10:00:00+00:00', '2026-09-15T10:05:00+00:00',
                         '2026-09-15T10:10:00+00:00', 100.0, .4, .3, .3, .4, .3, .4,
                         'v1', '{"ret_1m":0.1}', '{}')
            same_a = immutable + (None, None, None, None, None, None, None, None)
            same_b = immutable + (None, None, None, None, None, None, None, None)
            distinct_model = ('2026-09-15T10:00:00+00:00', immutable[1], immutable[2], 100.0,
                              .5, .2, .3, .4, .3, .3, 'v2', immutable[11], '{}',
                              None, None, None, None, None, None, None, None)
            make_db(target, [same_a, same_b, distinct_model])
            con = sqlite3.connect(target)
            self.assertEqual(expected_compacted_event_count(con), 2)
            con.close()

    def test_settlement_conflict_evidence_survives_state_merge(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'target.db'
            local = Path(td) / 'local.db'
            make_db(target, [])
            make_db(local, [])

            con = sqlite3.connect(local)
            con.execute(
                '''CREATE TABLE prediction_settlement_conflicts (
                    identity_sha256 TEXT PRIMARY KEY,
                    detected_at_utc TEXT NOT NULL,
                    identity_json TEXT NOT NULL,
                    conflicts_json TEXT NOT NULL,
                    resolution TEXT NOT NULL
                )'''
            )
            con.execute(
                '''INSERT INTO prediction_settlement_conflicts
                   (identity_sha256, detected_at_utc, identity_json, conflicts_json, resolution)
                   VALUES (?, ?, ?, ?, ?)''',
                ('abc123', '2026-09-25T16:00:00+00:00', '{}', '{"actual_price_5m":["1","2"]}',
                 'QUARANTINED_CONFLICTING_SETTLEMENT'),
            )
            con.commit()
            con.close()

            subprocess.run([sys.executable, str(SCRIPT), str(local), str(target)], check=True)

            con = sqlite3.connect(target)
            row = con.execute(
                '''SELECT identity_sha256, resolution, conflicts_json
                   FROM prediction_settlement_conflicts'''
            ).fetchone()
            con.close()
            self.assertEqual(
                row,
                ('abc123', 'QUARANTINED_CONFLICTING_SETTLEMENT', '{"actual_price_5m":["1","2"]}')
            )

    def test_model_registry_prefers_newer_state_and_does_not_roll_back(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'target.db'; local = Path(td) / 'local.db'
            make_db(target, []); make_db(local, [])
            add_registry(target, 'new_model', '2026-09-15T12:00:00+00:00')
            add_registry(local, 'old_model', '2026-09-15T11:00:00+00:00')
            subprocess.run([sys.executable, str(SCRIPT), str(local), str(target)], check=True)
            con = sqlite3.connect(target)
            row = con.execute('SELECT production_version, updated_at_utc FROM model_registry WHERE horizon="5m"').fetchone()
            con.close()
            self.assertEqual(row, ('new_model', '2026-09-15T12:00:00+00:00'))

    def test_model_registry_promotes_newer_local_state(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'target.db'; local = Path(td) / 'local.db'
            make_db(target, []); make_db(local, [])
            add_registry(target, 'old_model', '2026-09-15T11:00:00+00:00')
            add_registry(local, 'new_model', '2026-09-15T12:00:00+00:00')
            subprocess.run([sys.executable, str(SCRIPT), str(local), str(target)], check=True)
            con = sqlite3.connect(target)
            row = con.execute('SELECT production_version, updated_at_utc FROM model_registry WHERE horizon="5m"').fetchone()
            con.close()
            self.assertEqual(row, ('new_model', '2026-09-15T12:00:00+00:00'))


if __name__ == '__main__':
    unittest.main()
