import json
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pit_oos_audit  # noqa: E402


class TestPITOOSAudit(unittest.TestCase):
    def make_db(self, td, rows):
        db = Path(td) / "predictions.db"
        with sqlite3.connect(db) as con:
            con.execute("CREATE TABLE predictions (prediction_id INTEGER PRIMARY KEY, created_at_utc TEXT, target_5m TEXT, target_10m TEXT, model_version TEXT, scenario_json TEXT)")
            con.executemany("INSERT INTO predictions VALUES (?,?,?,?,?,?)", rows)
        return db

    def test_accepts_ordered_future_targets(self):
        with tempfile.TemporaryDirectory() as td:
            now = datetime.now(timezone.utc).replace(microsecond=0)
            scenario = {
                "decision_time_utc": now.isoformat(),
                "market_data_cutoff_utc": (now - timedelta(minutes=1)).isoformat(),
                "provenance": {
                    "event_time": now.isoformat(),
                    "available_at": now.isoformat(),
                    "retrieved_at": now.isoformat(),
                    "prediction_cutoff": now.isoformat(),
                    "sources": {
                        "binance_futures": {
                            "event_time": now.isoformat(),
                            "available_at": now.isoformat(),
                            "retrieved_at": now.isoformat(),
                            "prediction_cutoff": now.isoformat(),
                        }
                    },
                },
            }
            db = self.make_db(td, [(1, now.isoformat(), (now + timedelta(minutes=5)).isoformat(), (now + timedelta(minutes=10)).isoformat(), "v1", json.dumps(scenario))])
            with patch.object(pit_oos_audit, "DB", db), patch.object(pit_oos_audit, "OUT", Path(td) / "audit.json"):
                result = pit_oos_audit.audit()
                self.assertTrue(result["ok"], result)


    def test_ignores_failed_unused_source_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            now = datetime.now(timezone.utc).replace(microsecond=0)
            scenario = {
                "decision_time_utc": now.isoformat(),
                "provenance": {
                    "event_time": now.isoformat(),
                    "available_at": now.isoformat(),
                    "retrieved_at": now.isoformat(),
                    "prediction_cutoff": now.isoformat(),
                    "sources": {
                        "binance_futures": {
                            "information_origin": "Binance",
                            "status": "error:HTTPError:451",
                            "available_at": None,
                            "retrieved_at": None,
                            "prediction_cutoff": None,
                        }
                    },
                },
            }
            db = self.make_db(td, [(1, now.isoformat(), (now + timedelta(minutes=5)).isoformat(), (now + timedelta(minutes=10)).isoformat(), "coinbase_fallback.rf.v1", json.dumps(scenario))])
            with patch.object(pit_oos_audit, "DB", db), patch.object(pit_oos_audit, "OUT", Path(td) / "audit.json"):
                result = pit_oos_audit.audit()
                self.assertTrue(result["ok"], result)
                self.assertTrue(result["pit_verified"])

    def test_legacy_rows_do_not_count_as_active_pit_violations(self):
        with tempfile.TemporaryDirectory() as td:
            now = datetime.now(timezone.utc).replace(microsecond=0)
            legacy = [(i, (now - timedelta(minutes=i + 1)).isoformat(),
                       (now + timedelta(minutes=5)).isoformat(),
                       (now + timedelta(minutes=10)).isoformat(),
                       "v1", "{}") for i in range(1, 5)]
            db = self.make_db(td, legacy)
            with patch.object(pit_oos_audit, "DB", db), patch.object(pit_oos_audit, "OUT", Path(td) / "audit.json"):
                result = pit_oos_audit.audit()
                self.assertTrue(result["ok"], result)
                self.assertFalse(result["pit_verified"])
                self.assertEqual(result["legacy_unverified_count"], 4)

    def test_accepts_provenance_cutoff_as_decision_time(self):
        with tempfile.TemporaryDirectory() as td:
            created = datetime.now(timezone.utc).replace(microsecond=0)
            decision = created + timedelta(seconds=20)
            scenario = {
                "provenance": {
                    "event_time": created.isoformat(),
                    "available_at": decision.isoformat(),
                    "retrieved_at": decision.isoformat(),
                    "prediction_cutoff": decision.isoformat(),
                    "sources": {
                        "binance_futures": {
                            "event_time": created.isoformat(),
                            "available_at": decision.isoformat(),
                            "retrieved_at": decision.isoformat(),
                            "prediction_cutoff": decision.isoformat(),
                        }
                    },
                }
            }
            db = self.make_db(td, [(1, created.isoformat(), (decision + timedelta(minutes=5)).isoformat(), (decision + timedelta(minutes=10)).isoformat(), "v1", json.dumps(scenario))])
            with patch.object(pit_oos_audit, "DB", db), patch.object(pit_oos_audit, "OUT", Path(td) / "audit.json"):
                result = pit_oos_audit.audit()
                self.assertTrue(result["ok"], result)

    def test_rejects_target_before_decision(self):
        with tempfile.TemporaryDirectory() as td:
            now = datetime.now(timezone.utc).replace(microsecond=0)
            db = self.make_db(td, [(1, now.isoformat(), (now - timedelta(minutes=1)).isoformat(), (now + timedelta(minutes=10)).isoformat(), "v1", "{}")])
            with patch.object(pit_oos_audit, "DB", db), patch.object(pit_oos_audit, "OUT", Path(td) / "audit.json"):
                result = pit_oos_audit.audit()
                self.assertFalse(result["ok"])
                self.assertIn("5m_target_not_after_decision", result["violations"][0])

    def test_rejects_prediction_time_far_in_future(self):
        with tempfile.TemporaryDirectory() as td:
            now = datetime.now(timezone.utc).replace(microsecond=0)
            created = now + timedelta(minutes=5)
            db = self.make_db(td, [(1, created.isoformat(), (created + timedelta(minutes=5)).isoformat(), (created + timedelta(minutes=10)).isoformat(), "v1", "{}")])
            with patch.object(pit_oos_audit, "DB", db), patch.object(pit_oos_audit, "OUT", Path(td) / "audit.json"):
                result = pit_oos_audit.audit()
                self.assertFalse(result["ok"])
                self.assertTrue(any("prediction_time_in_future" in v for v in result["violations"]))

    def test_rejects_invalid_pit_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            now = datetime.now(timezone.utc).replace(microsecond=0)
            db = self.make_db(td, [(1, now.isoformat(), (now + timedelta(minutes=5)).isoformat(), (now + timedelta(minutes=10)).isoformat(), "v1", json.dumps({"decision_time_utc": "not-a-timestamp"}))])
            with patch.object(pit_oos_audit, "DB", db), patch.object(pit_oos_audit, "OUT", Path(td) / "audit.json"):
                result = pit_oos_audit.audit()
                self.assertFalse(result["ok"])
                self.assertTrue(any("invalid_pit_metadata" in v for v in result["violations"]))

    def test_rejects_degraded_policy_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            now = datetime.now(timezone.utc).replace(microsecond=0)
            db = self.make_db(td, [(1, now.isoformat(), (now + timedelta(minutes=5)).isoformat(), (now + timedelta(minutes=10)).isoformat(), "DEGRADED_NO_FRESH_DATA", json.dumps({"policy": "wrong"}))])
            with patch.object(pit_oos_audit, "DB", db), patch.object(pit_oos_audit, "OUT", Path(td) / "audit.json"):
                result = pit_oos_audit.audit()
                self.assertFalse(result["ok"])
                self.assertTrue(any("degraded_policy_mismatch" in v for v in result["violations"]))

    def test_rejects_market_cutoff_after_decision(self):
        with tempfile.TemporaryDirectory() as td:
            now = datetime.now(timezone.utc).replace(microsecond=0)
            db = self.make_db(td, [(1, now.isoformat(), (now + timedelta(minutes=5)).isoformat(), (now + timedelta(minutes=10)).isoformat(), "v1", json.dumps({"decision_time_utc": now.isoformat(), "market_data_cutoff_utc": (now + timedelta(seconds=1)).isoformat()}))])
            with patch.object(pit_oos_audit, "DB", db), patch.object(pit_oos_audit, "OUT", Path(td) / "audit.json"):
                result = pit_oos_audit.audit()
                self.assertFalse(result["ok"])
                self.assertTrue(any("market_cutoff_after_decision" in v for v in result["violations"]))


    def test_rejects_available_at_after_prediction_cutoff(self):
        with tempfile.TemporaryDirectory() as td:
            now = datetime.now(timezone.utc).replace(microsecond=0)
            scenario = {
                "provenance": {
                    "event_time": now.isoformat(),
                    "available_at": (now + timedelta(seconds=1)).isoformat(),
                    "retrieved_at": (now + timedelta(seconds=2)).isoformat(),
                    "prediction_cutoff": now.isoformat(),
                    "sources": {
                        "binance_futures": {
                            "event_time": now.isoformat(),
                            "available_at": (now + timedelta(seconds=1)).isoformat(),
                            "retrieved_at": (now + timedelta(seconds=2)).isoformat(),
                            "prediction_cutoff": now.isoformat(),
                        }
                    },
                }
            }
            db = self.make_db(td, [(1, now.isoformat(), (now + timedelta(minutes=5)).isoformat(), (now + timedelta(minutes=10)).isoformat(), "v1", json.dumps(scenario))])
            with patch.object(pit_oos_audit, "DB", db), patch.object(pit_oos_audit, "OUT", Path(td) / "audit.json"):
                result = pit_oos_audit.audit()
                self.assertFalse(result["ok"])
                self.assertTrue(any("available_at_after_prediction_cutoff" in v for v in result["violations"]))

    def test_rejects_missing_source_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            now = datetime.now(timezone.utc).replace(microsecond=0)
            scenario = {
                "provenance": {
                    "event_time": now.isoformat(),
                    "available_at": now.isoformat(),
                    "retrieved_at": now.isoformat(),
                    "prediction_cutoff": now.isoformat(),
                }
            }
            db = self.make_db(td, [(1, now.isoformat(), (now + timedelta(minutes=5)).isoformat(), (now + timedelta(minutes=10)).isoformat(), "v1", json.dumps(scenario))])
            with patch.object(pit_oos_audit, "DB", db), patch.object(pit_oos_audit, "OUT", Path(td) / "audit.json"):
                result = pit_oos_audit.audit()
                self.assertFalse(result["ok"])
                self.assertTrue(any("missing_source_provenance" in v for v in result["violations"]))

    def test_accepts_provenance_with_unknown_event_time_but_valid_available_at(self):
        with tempfile.TemporaryDirectory() as td:
            now = datetime.now(timezone.utc).replace(microsecond=0)
            scenario = {
                "provenance": {
                    "event_time": now.isoformat(),
                    "available_at": now.isoformat(),
                    "retrieved_at": now.isoformat(),
                    "prediction_cutoff": now.isoformat(),
                    "sources": {
                        "bybit_futures": {
                            "event_time": None,
                            "available_at": now.isoformat(),
                            "retrieved_at": now.isoformat(),
                            "prediction_cutoff": now.isoformat(),
                        }
                    },
                }
            }
            db = self.make_db(td, [(1, now.isoformat(), (now + timedelta(minutes=5)).isoformat(), (now + timedelta(minutes=10)).isoformat(), "v1", json.dumps(scenario))])
            with patch.object(pit_oos_audit, "DB", db), patch.object(pit_oos_audit, "OUT", Path(td) / "audit.json"):
                result = pit_oos_audit.audit()
                self.assertTrue(result["ok"], result)

if __name__ == "__main__":
    unittest.main()
