import json
import sqlite3
import unittest

from src.feature_schema import FEATURES
from src.research_input_audit import audit_horizon


def feature_json():
    return json.dumps({name: float(i + 1) for i, name in enumerate(FEATURES)}, sort_keys=True)


SCHEMA = """
CREATE TABLE predictions (
  prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at_utc TEXT NOT NULL,
  target_5m TEXT NOT NULL,
  target_10m TEXT NOT NULL,
  model_version TEXT NOT NULL,
  feature_json TEXT NOT NULL,
  actual_direction_5m TEXT,
  actual_direction_10m TEXT,
  p_up_5m REAL NOT NULL,
  p_down_5m REAL NOT NULL,
  p_flat_5m REAL NOT NULL,
  p_up_10m REAL NOT NULL,
  p_down_10m REAL NOT NULL,
  p_flat_10m REAL NOT NULL,
  scenario_json TEXT NOT NULL
)
"""


class ResearchInputAuditTests(unittest.TestCase):
    def _db(self):
        con = sqlite3.connect(":memory:")
        con.executescript(SCHEMA)
        return con

    def _row(self, model="model-v1", p=(0.2, 0.7, 0.1)):
        return (
            "2026-09-22T00:00:00+00:00",
            "2026-09-22T00:05:00+00:00",
            "2026-09-22T00:10:00+00:00",
            model,
            feature_json(),
            "UP",
            "UP",
            *p,
            *p,
            "{}",
        )

    def _insert(self, con, rows):
        con.executemany(
            """INSERT INTO predictions(
                created_at_utc,target_5m,target_10m,model_version,feature_json,
                actual_direction_5m,actual_direction_10m,
                p_up_5m,p_down_5m,p_flat_5m,p_up_10m,p_down_10m,p_flat_10m,
                scenario_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )

    def test_same_event_and_model_probabilities_are_duplicate(self):
        con = self._db()
        row = self._row()
        self._insert(con, [row, row])
        result = audit_horizon(con, "5m")
        self.assertEqual(result["duplicate_exact_key_row_excess"], 1)

    def test_unsettled_duplicate_events_are_not_invisible_to_the_audit(self):
        con = self._db()
        row = list(self._row())
        row[5] = None
        row[6] = None
        self._insert(con, [tuple(row), tuple(row)])
        result = audit_horizon(con, "5m")
        self.assertEqual(result["total_rows"], 2)
        self.assertEqual(result["settled_rows"], 0)
        self.assertEqual(result["duplicate_exact_key_row_excess"], 1)

    def test_malformed_event_identity_is_reported(self):
        con = self._db()
        row = list(self._row())
        row[4] = '{"ret_1m": "not-a-number"}'
        self._insert(con, [tuple(row)])
        result = audit_horizon(con, "5m")
        self.assertEqual(result["malformed_identity_rows"], 1)

    def test_different_model_versions_are_distinct_events(self):
        con = self._db()
        self._insert(con, [self._row("model-v1"), self._row("model-v2")])
        result = audit_horizon(con, "5m")
        self.assertEqual(result["duplicate_exact_key_row_excess"], 0)

    def test_different_probabilities_are_distinct_events(self):
        con = self._db()
        self._insert(con, [self._row(p=(0.2, 0.7, 0.1)), self._row(p=(0.3, 0.6, 0.1))])
        result = audit_horizon(con, "5m")
        self.assertEqual(result["duplicate_exact_key_row_excess"], 0)


if __name__ == "__main__":
    unittest.main()
