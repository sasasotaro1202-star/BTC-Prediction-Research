import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.merge_prediction_state import compact_predictions
from src.prediction_identity import canonical_compaction_snapshot


def make_db(path: Path, rows):
    with sqlite3.connect(path) as con:
        con.execute(
            """CREATE TABLE predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_utc TEXT NOT NULL,
                target_5m TEXT NOT NULL,
                target_10m TEXT NOT NULL,
                base_price REAL NOT NULL,
                p_up_5m REAL NOT NULL,
                p_down_5m REAL NOT NULL,
                p_flat_5m REAL NOT NULL,
                p_up_10m REAL NOT NULL,
                p_down_10m REAL NOT NULL,
                p_flat_10m REAL NOT NULL,
                model_version TEXT NOT NULL,
                feature_json TEXT NOT NULL,
                scenario_json TEXT NOT NULL,
                actual_price_5m REAL,
                actual_direction_5m TEXT,
                correct_5m INTEGER,
                settled_5m_at_utc TEXT,
                actual_price_10m REAL,
                actual_direction_10m TEXT,
                correct_10m INTEGER,
                settled_10m_at_utc TEXT
            )"""
        )
        con.executemany(
            """INSERT INTO predictions(
                created_at_utc,target_5m,target_10m,base_price,
                p_up_5m,p_down_5m,p_flat_5m,p_up_10m,p_down_10m,p_flat_10m,
                model_version,feature_json,scenario_json,
                actual_price_5m,actual_direction_5m,correct_5m,settled_5m_at_utc,
                actual_price_10m,actual_direction_10m,correct_10m,settled_10m_at_utc
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )


def row(ts, model="v1", p=(0.4, 0.3, 0.3), settlement=None):
    settled = settlement or (None,) * 8
    return (
        ts,
        "2026-09-22T00:05:00+00:00",
        "2026-09-22T00:10:00+00:00",
        100.0,
        p[0], p[1], p[2],
        p[0], p[1], p[2],
        model,
        '{"ret_1m":0.1}',
        '{}',
        *settled,
    )


class TestPredictionIdentityDigest(unittest.TestCase):
    def test_duplicate_compaction_preserves_identity_and_settlement_digests(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "predictions.db"
            settled = (101.0, "UP", 1, "2026-09-22T00:05:01+00:00", 102.0, "UP", 1, "2026-09-22T00:10:01+00:00")
            make_db(db, [row("2026-09-22T00:00:00+00:00"), row("2026-09-22T00:00:00+00:00", settlement=settled)])
            with sqlite3.connect(db) as con:
                before = canonical_compaction_snapshot(con)
                result = compact_predictions(con)
                con.commit()
                after = canonical_compaction_snapshot(con)
            self.assertEqual(result["total_events"], 1)
            self.assertEqual(before["unique_event_count"], after["unique_event_count"])
            self.assertEqual(before["immutable_identity_sha256"], after["immutable_identity_sha256"])
            self.assertEqual(before["settlement_state_sha256"], after["settlement_state_sha256"])
            self.assertFalse(after["settlement_conflicts"])

    def test_same_cardinality_replacement_changes_identity_digest(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "predictions.db"
            make_db(db, [row("2026-09-22T00:00:00+00:00"), row("2026-09-22T00:01:00+00:00", model="v2")])
            with sqlite3.connect(db) as con:
                before = canonical_compaction_snapshot(con)
                con.execute(
                    "UPDATE predictions SET model_version='v3' WHERE prediction_id=2"
                )
                con.commit()
                after = canonical_compaction_snapshot(con)
            self.assertEqual(before["unique_event_count"], after["unique_event_count"])
            self.assertNotEqual(
                before["immutable_identity_sha256"],
                after["immutable_identity_sha256"],
            )

    def test_different_settlement_timestamps_are_reconciled_not_conflicted(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "predictions.db"
            early = (101.0, "UP", 1, "2026-09-22T00:05:01+00:00", 102.0, "UP", 1, "2026-09-22T00:10:01+00:00")
            late = (101.0, "UP", 1, "2026-09-22T00:05:02+00:00", 102.0, "UP", 1, "2026-09-22T00:10:02+00:00")
            make_db(
                db,
                [
                    row("2026-09-22T00:00:00+00:00", settlement=early),
                    row("2026-09-22T00:00:00+00:00", settlement=late),
                ],
            )
            with sqlite3.connect(db) as con:
                before = canonical_compaction_snapshot(con)
                self.assertFalse(before["settlement_conflicts"])
                result = compact_predictions(con)
                con.commit()
                after = canonical_compaction_snapshot(con)
                settled = con.execute(
                    "SELECT settled_5m_at_utc, settled_10m_at_utc FROM predictions"
                ).fetchone()
            self.assertEqual(result["total_events"], 1)
            self.assertEqual(settled, (early[3], early[7]))
            self.assertEqual(
                before["settlement_state_sha256"],
                after["settlement_state_sha256"],
            )

    def test_invalid_settlement_timestamp_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "predictions.db"
            bad = (101.0, "UP", 1, "not-a-timestamp", None, None, None, None)
            make_db(db, [row("2026-09-22T00:00:00+00:00", settlement=bad)])
            with sqlite3.connect(db) as con:
                snapshot = canonical_compaction_snapshot(con)
            self.assertTrue(snapshot["settlement_conflicts"])

    def test_conflicting_non_null_settlement_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "predictions.db"
            a = (101.0, "UP", 1, "2026-09-22T00:05:01+00:00", None, None, None, None)
            b = (102.0, "DOWN", 0, "2026-09-22T00:05:02+00:00", None, None, None, None)
            make_db(
                db,
                [
                    row("2026-09-22T00:00:00+00:00", settlement=a),
                    row("2026-09-22T00:00:00+00:00", settlement=b),
                ],
            )
            with sqlite3.connect(db) as con:
                snapshot = canonical_compaction_snapshot(con)
                self.assertTrue(snapshot["settlement_conflicts"])
                with self.assertRaises(RuntimeError):
                    compact_predictions(con)


if __name__ == "__main__":
    unittest.main()
