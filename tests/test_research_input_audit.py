import json
import sqlite3
import unittest

from src.feature_schema import FEATURES
from src.research_input_audit import audit_horizon, recent_pit_stats


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

    def test_degraded_rows_are_explicitly_quarantined_not_malformed(self):
        con = self._db()
        row = list(self._row("DEGRADED_NO_FRESH_DATA"))
        row[4] = "{}"
        row[5] = None
        row[6] = None
        self._insert(con, [tuple(row)])
        result = audit_horizon(con, "5m")
        self.assertEqual(result["malformed_identity_rows"], 0)
        self.assertEqual(result["quarantined_identity_rows"], 1)
        self.assertEqual(
            result["quarantine_reasons"],
            {"degraded_prediction_no_feature_snapshot": 1},
        )
        self.assertEqual(result["strict_pit_rows"], 0)

    def test_legacy_coinbase_v1_incomplete_snapshot_is_explicitly_quarantined(self):
        con = self._db()
        row = list(self._row("5m:v1.0|10m:v1.0"))
        row[4] = json.dumps({"ret_1m": 0.1})
        row[13] = json.dumps({"price_source": "coinbase_btc_usd"})
        self._insert(con, [tuple(row)])
        result = audit_horizon(con, "5m")
        self.assertEqual(result["malformed_identity_rows"], 0)
        self.assertEqual(result["quarantined_identity_rows"], 1)
        self.assertEqual(
            result["quarantine_reasons"],
            {"legacy_coinbase_v1_incomplete_feature_snapshot": 1},
        )
        self.assertEqual(result["duplicate_exact_key_row_excess"], 0)

    def test_nonlegacy_incomplete_snapshot_stays_malformed(self):
        con = self._db()
        row = list(self._row("model-v2"))
        row[4] = json.dumps({"ret_1m": 0.1})
        row[13] = json.dumps({"price_source": "coinbase_btc_usd"})
        self._insert(con, [tuple(row)])
        result = audit_horizon(con, "5m")
        self.assertEqual(result["malformed_identity_rows"], 1)
        self.assertEqual(result["quarantined_identity_rows"], 0)


    def test_legacy_coinbase_precontract_missing_cutoff_is_quarantined(self):
        con = self._db()
        created = "2026-09-21T12:00:00+00:00"
        scenario = {
            "provenance": {
                "event_time": created,
                "available_at": created,
                "retrieved_at": created,
                "prediction_cutoff": created,
                "sources": {
                    "coinbase_futures": {
                        "status": "ok",
                        "event_time": created,
                        "available_at": created,
                        "retrieved_at": created,
                        "prediction_cutoff": None,
                    }
                },
            }
        }
        row = list(self._row("5m:coinbase_fallback.rf.v1|10m:coinbase_fallback.rf.v1"))
        row[0] = created
        row[1] = "2026-09-21T12:05:00+00:00"
        row[2] = "2026-09-21T12:10:00+00:00"
        row[13] = json.dumps(scenario)
        self._insert(con, [tuple(row)])
        result = audit_horizon(con, "5m")
        self.assertEqual(result["malformed_identity_rows"], 0)
        self.assertEqual(result["quarantined_identity_rows"], 1)
        self.assertEqual(result["quarantine_reasons"], {"legacy_coinbase_precontract_missing_cutoff": 1})

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


    def test_recent_pit_stats_excludes_pre_contract_rows_from_post_fix_window(self):
        con = self._db()
        row = list(self._row("model-old"))
        row[0] = "2026-09-25T19:50:00+00:00"
        row[1] = "2026-09-25T19:55:00+00:00"
        row[13] = json.dumps({
            "production_mode": "coinbase_fallback",
            "provenance": {
                "available_at": "2026-09-25T19:49:59+00:00",
                "retrieved_at": "2026-09-25T19:50:00+00:00",
                "prediction_cutoff": "2026-09-25T19:50:00+00:00",
                "sources": {
                    "coinbase_futures": {
                        "status": "ok",
                        "event_time": "2026-09-25T19:50:01+00:00",
                        "available_at": "2026-09-25T19:50:00+00:00",
                        "retrieved_at": "2026-09-25T19:50:00+00:00",
                        "prediction_cutoff": "2026-09-25T19:50:00+00:00",
                    }
                },
            },
        })
        self._insert(con, [tuple(row)])
        result = recent_pit_stats(con, "5m", limit=20)
        self.assertEqual(result["strict_pit"], 0)
        self.assertEqual(result["observed"], 0)
        self.assertEqual(result["failure_reasons"], {})
        self.assertEqual(result["pre_contract_quarantined"], 1)
        self.assertFalse(result["all_strict"])


    def test_recent_pit_stats_detects_missing_provenance(self):
        con = self._db()
        row = list(self._row("model-v3"))
        row[0] = "2026-09-26T00:00:00+00:00"
        row[1] = "2026-09-26T00:05:00+00:00"
        row[2] = "2026-09-26T00:10:00+00:00"
        scenario = {
            "production_mode": "binance_primary",
            "decision_time_utc": "2026-09-26T00:00:00+00:00",
            "provenance": {
                "available_at": "2026-09-25T23:59:50+00:00",
                "retrieved_at": "2026-09-25T23:59:55+00:00",
                "prediction_cutoff": "2026-09-26T00:00:00+00:00",
                "sources": {
                    name: {
                        "status": "ok",
                        "event_time": "2026-09-25T23:59:40+00:00",
                        "available_at": "2026-09-25T23:59:50+00:00",
                        "retrieved_at": "2026-09-25T23:59:55+00:00",
                        "prediction_cutoff": "2026-09-26T00:00:00+00:00",
                    }
                    for name in ("binance_futures", "binance_depth", "binance_taker", "binance_premium")
                },
            },
        }
        row[13] = json.dumps(scenario)
        self._insert(con, [tuple(row)])
        result = recent_pit_stats(con, "5m", limit=20)
        self.assertTrue(result["all_strict"])
        self.assertEqual(result["rate"], 1.0)

        row2 = list(self._row("model-v4"))
        row2[0] = "2026-09-26T00:20:00+00:00"
        row2[1] = "2026-09-26T00:25:00+00:00"
        row2[2] = "2026-09-26T00:30:00+00:00"
        self._insert(con, [tuple(row2)])
        result2 = recent_pit_stats(con, "5m", limit=20)
        self.assertFalse(result2["all_strict"])
        self.assertLess(result2["rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
