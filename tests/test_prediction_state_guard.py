import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.prediction_state_guard import prediction_table_fingerprint


def make_db(path: Path) -> None:
    with sqlite3.connect(path) as con:
        con.execute(
            """CREATE TABLE predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_utc TEXT NOT NULL,
                target_5m TEXT NOT NULL,
                target_10m TEXT NOT NULL,
                p_up_5m REAL NOT NULL,
                p_down_5m REAL NOT NULL,
                p_flat_5m REAL NOT NULL,
                model_version TEXT NOT NULL,
                feature_json TEXT NOT NULL,
                actual_direction_5m TEXT
            )"""
        )
        con.executemany(
            """INSERT INTO predictions(
                created_at_utc,target_5m,target_10m,p_up_5m,p_down_5m,
                p_flat_5m,model_version,feature_json,actual_direction_5m
            ) VALUES(?,?,?,?,?,?,?,?,?)""",
            [
                ("2026-09-22T00:00:00+00:00","2026-09-22T00:05:00+00:00","2026-09-22T00:10:00+00:00",.4,.3,.3,"v1","{\"ret1\":0.1}",None),
                ("2026-09-22T00:01:00+00:00","2026-09-22T00:06:00+00:00","2026-09-22T00:11:00+00:00",.2,.5,.3,"v1","{\"ret1\":0.2}",None),
            ],
        )
        con.commit()


class TestPredictionStateGuard(unittest.TestCase):
    def test_identical_state_has_identical_fingerprint(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "predictions.db"
            make_db(db)
            self.assertEqual(prediction_table_fingerprint(db), prediction_table_fingerprint(db))

    def test_settlement_mutation_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "predictions.db"
            make_db(db)
            before = prediction_table_fingerprint(db)
            with sqlite3.connect(db) as con:
                con.execute(
                    "UPDATE predictions SET actual_direction_5m='UP' WHERE prediction_id=1"
                )
                con.commit()
            after = prediction_table_fingerprint(db)
            self.assertNotEqual(before["sha256"], after["sha256"])
            self.assertEqual(before["count"], after["count"])

    def test_append_changes_count_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "predictions.db"
            make_db(db)
            before = prediction_table_fingerprint(db)
            with sqlite3.connect(db) as con:
                con.execute(
                    """INSERT INTO predictions(
                        created_at_utc,target_5m,target_10m,p_up_5m,p_down_5m,
                        p_flat_5m,model_version,feature_json,actual_direction_5m
                    ) VALUES(?,?,?,?,?,?,?,?,?)""",
                    ("2026-09-22T00:02:00+00:00","2026-09-22T00:07:00+00:00","2026-09-22T00:12:00+00:00",.5,.2,.3,"v1","{\"ret1\":0.3}",None),
                )
                con.commit()
            after = prediction_table_fingerprint(db)
            self.assertEqual(after["count"], before["count"] + 1)
            self.assertNotEqual(after["sha256"], before["sha256"])

    def test_missing_table_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "empty.db"
            sqlite3.connect(db).close()
            with self.assertRaises(ValueError):
                prediction_table_fingerprint(db)


if __name__ == "__main__":
    unittest.main()
