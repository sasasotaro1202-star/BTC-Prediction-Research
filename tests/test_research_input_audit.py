import sqlite3
import unittest

from src.research_input_audit import audit_horizon


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
  scenario_json TEXT NOT NULL
)
"""


class ResearchInputAuditTests(unittest.TestCase):
    def _db(self):
        con = sqlite3.connect(":memory:")
        con.executescript(SCHEMA)
        return con

    def test_same_event_and_model_version_is_duplicate(self):
        con = self._db()
        row = (
            "2026-09-22T00:00:00+00:00",
            "2026-09-22T00:05:00+00:00",
            "2026-09-22T00:10:00+00:00",
            "model-v1",
            '{"x": 1}',
            "UP",
            "UP",
            "{}",
        )
        con.executemany(
            """INSERT INTO predictions(
                created_at_utc,target_5m,target_10m,model_version,
                feature_json,actual_direction_5m,actual_direction_10m,scenario_json
            ) VALUES(?,?,?,?,?,?,?,?)""",
            [row, row],
        )
        result = audit_horizon(con, "5m")
        self.assertEqual(result["duplicate_exact_key_row_excess"], 1)

    def test_different_model_versions_are_distinct_events(self):
        con = self._db()
        base = (
            "2026-09-22T00:00:00+00:00",
            "2026-09-22T00:05:00+00:00",
            "2026-09-22T00:10:00+00:00",
            '{"x": 1}',
            "UP",
            "UP",
            "{}",
        )
        rows = [
            base[:3] + ("model-v1",) + base[3:],
            base[:3] + ("model-v2",) + base[3:],
        ]
        con.executemany(
            """INSERT INTO predictions(
                created_at_utc,target_5m,target_10m,model_version,
                feature_json,actual_direction_5m,actual_direction_10m,scenario_json
            ) VALUES(?,?,?,?,?,?,?,?)""",
            rows,
        )
        result = audit_horizon(con, "5m")
        self.assertEqual(result["duplicate_exact_key_row_excess"], 0)


if __name__ == "__main__":
    unittest.main()
