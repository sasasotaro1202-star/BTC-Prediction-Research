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
from model_compare import strict_pit_provenance_reason  # noqa: E402


class TestPITOOSAudit(unittest.TestCase):
    def make_db(self, td, rows):
        db = Path(td) / "predictions.db"
        with sqlite3.connect(db) as con:
            con.execute(
                "CREATE TABLE predictions ("
                "prediction_id INTEGER PRIMARY KEY, created_at_utc TEXT, target_5m TEXT, "
                "target_10m TEXT, model_version TEXT, actual_direction_5m TEXT, "
                "actual_direction_10m TEXT, scenario_json TEXT)"
            )
            normalized = []
            for row in rows:
                if len(row) == 6:
                    normalized.append(tuple(row[:5]) + (None, None, row[5]))
                else:
                    normalized.append(tuple(row))
            con.executemany("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?)", normalized)
        return db

    def test_strict_pit_reason_identifies_missing_source_cutoff(self):
        now = datetime.now(timezone.utc).replace(microsecond=0)
        scenario = {
            "decision_time_utc": now.isoformat(),
            "provenance": {
                "available_at": now.isoformat(),
                "retrieved_at": now.isoformat(),
                "prediction_cutoff": now.isoformat(),
                "sources": {
                    "bybit_futures": {
                        "event_time": None,
                        "available_at": now.isoformat(),
                        "retrieved_at": now.isoformat(),
                        "status": "ok",
                    }
                },
            },
            "production_mode": "bybit_fallback",
        }
        self.assertEqual(
            strict_pit_provenance_reason(scenario, now.isoformat()),
            "source_bybit_futures_prediction_cutoff_invalid",
        )

    def test_strict_pit_reason_returns_none_for_valid_bybit(self):
        now = datetime.now(timezone.utc).replace(microsecond=0)
        scenario = {
            "decision_time_utc": now.isoformat(),
            "provenance": {
                "available_at": now.isoformat(),
                "retrieved_at": now.isoformat(),
                "prediction_cutoff": now.isoformat(),
                "sources": {
                    "bybit_futures": {
                        "event_time": None,
                        "available_at": now.isoformat(),
                        "retrieved_at": now.isoformat(),
                        "prediction_cutoff": now.isoformat(),
                        "status": "ok",
                    }
                },
            },
            "production_mode": "bybit_fallback",
        }
        self.assertIsNone(strict_pit_provenance_reason(scenario, now.isoformat()))

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
                        },
                        "coinbase_futures": {
                            "information_origin": "Coinbase Exchange BTC-USD",
                            "status": "ok",
                            "event_time": now.isoformat(),
                            "available_at": now.isoformat(),
                            "retrieved_at": now.isoformat(),
                            "prediction_cutoff": now.isoformat(),
                        }
                    },
                },
            }
            db = self.make_db(td, [(1, now.isoformat(), (now + timedelta(minutes=5)).isoformat(), (now + timedelta(minutes=10)).isoformat(), "coinbase_fallback.rf.v1", json.dumps(scenario))])
            with patch.object(pit_oos_audit, "DB", db), patch.object(pit_oos_audit, "OUT", Path(td) / "audit.json"):
                result = pit_oos_audit.audit()
                self.assertTrue(result["ok"], result)
                self.assertFalse(result["pit_verified"])
                self.assertEqual(result["verified_predictions"], 1)
                self.assertEqual(result["verified_primary_predictions"], 0)
                self.assertEqual(result["verified_fallback_predictions"], 1)
                self.assertIn("insufficient_binance_primary_pit_rows", result["pit_verified_reason"])

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


    def test_fallback_pit_rows_cannot_satisfy_primary_pit_gate(self):
        with tempfile.TemporaryDirectory() as td:
            created = datetime.now(timezone.utc).replace(microsecond=0)
            rows = []
            for i in range(pit_oos_audit.MIN_STRICT_PIT_ROWS):
                at = (created + timedelta(milliseconds=i)).isoformat()
                scenario = {
                    "decision_time_utc": at,
                    "production_mode": "coinbase_fallback",
                    "provenance": {
                        "event_time": at,
                        "available_at": at,
                        "retrieved_at": at,
                        "prediction_cutoff": at,
                        "sources": {
                            "coinbase_futures": {
                                "status": "ok",
                                "event_time": at,
                                "available_at": at,
                                "retrieved_at": at,
                                "prediction_cutoff": at,
                            }
                        },
                    },
                }
                rows.append((i + 1, at, (created + timedelta(minutes=5, milliseconds=i + 1)).isoformat(),
                             (created + timedelta(minutes=10, milliseconds=i + 1)).isoformat(),
                             "coinbase_fallback.rf.v1", json.dumps(scenario)))
            db = self.make_db(td, rows)
            with patch.object(pit_oos_audit, "DB", db), patch.object(pit_oos_audit, "OUT", Path(td) / "audit.json"):
                result = pit_oos_audit.audit()
                self.assertTrue(result["ok"], result)
                self.assertFalse(result["pit_verified"])
                self.assertEqual(result["verified_primary_predictions"], 0)
                self.assertEqual(result["verified_fallback_predictions"], pit_oos_audit.MIN_STRICT_PIT_ROWS)

    def test_pit_verified_requires_strict_row_minimum(self):
        with tempfile.TemporaryDirectory() as td:
            created = datetime.now(timezone.utc).replace(microsecond=0)
            rows = []
            for i in range(pit_oos_audit.MIN_STRICT_PIT_ROWS):
                at = (created + timedelta(milliseconds=i)).isoformat()
                scenario = {
                    "decision_time_utc": at,
                    "provenance": {
                        "event_time": at,
                        "available_at": at,
                        "retrieved_at": at,
                        "prediction_cutoff": at,
                        "sources": {
                            "binance_futures": {
                                "status": "ok",
                                "event_time": at,
                                "available_at": at,
                                "retrieved_at": at,
                                "prediction_cutoff": at,
                            }
                        },
                    },
                }
                rows.append((i + 1, at, (created + timedelta(minutes=5, milliseconds=i + 1)).isoformat(),
                             (created + timedelta(minutes=10, milliseconds=i + 1)).isoformat(),
                             "v1", json.dumps(scenario)))
            db = self.make_db(td, rows)
            with patch.object(pit_oos_audit, "DB", db), patch.object(pit_oos_audit, "OUT", Path(td) / "audit.json"):
                result = pit_oos_audit.audit()
                self.assertTrue(result["ok"], result)
                self.assertTrue(result["pit_verified"])
                self.assertEqual(result["verified_predictions"], pit_oos_audit.MIN_STRICT_PIT_ROWS)
                self.assertEqual(result["verified_primary_predictions"], pit_oos_audit.MIN_STRICT_PIT_ROWS)
                self.assertEqual(result["verified_fallback_predictions"], 0)

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


    def test_precontract_coinbase_missing_cutoff_is_explicitly_quarantined(self):
        with tempfile.TemporaryDirectory() as td:
            created = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
            scenario = {
                "decision_time_utc": created.isoformat(),
                "provenance": {
                    "event_time": created.isoformat(),
                    "available_at": created.isoformat(),
                    "retrieved_at": created.isoformat(),
                    "prediction_cutoff": created.isoformat(),
                    "sources": {
                        "coinbase_futures": {
                            "status": "ok",
                            "event_time": created.isoformat(),
                            "available_at": created.isoformat(),
                            "retrieved_at": created.isoformat(),
                            "prediction_cutoff": None,
                        }
                    },
                },
                "production_mode": "coinbase_fallback",
            }
            db = self.make_db(td, [(1, created.isoformat(),
                                    (created + timedelta(minutes=5)).isoformat(),
                                    (created + timedelta(minutes=10)).isoformat(),
                                    "5m:coinbase_fallback.rf.v1|10m:coinbase_fallback.rf.v1",
                                    json.dumps(scenario))])
            with patch.object(pit_oos_audit, "DB", db), patch.object(pit_oos_audit, "OUT", Path(td) / "audit.json"):
                result = pit_oos_audit.audit()
                self.assertTrue(result["ok"], result)
                self.assertFalse(result["pit_verified"])
                self.assertEqual(result["verified_fallback_predictions"], 0)
                self.assertEqual(result["legacy_unverified_count"], 1)

    def test_postcontract_coinbase_missing_cutoff_is_not_quarantined(self):
        with tempfile.TemporaryDirectory() as td:
            created = datetime(2026, 9, 22, 6, 0, tzinfo=timezone.utc)
            scenario = {
                "decision_time_utc": created.isoformat(),
                "provenance": {
                    "event_time": created.isoformat(),
                    "available_at": created.isoformat(),
                    "retrieved_at": created.isoformat(),
                    "prediction_cutoff": created.isoformat(),
                    "sources": {
                        "coinbase_futures": {
                            "status": "ok",
                            "event_time": created.isoformat(),
                            "available_at": created.isoformat(),
                            "retrieved_at": created.isoformat(),
                            "prediction_cutoff": None,
                        }
                    },
                },
                "production_mode": "coinbase_fallback",
            }
            db = self.make_db(td, [(1, created.isoformat(),
                                    (created + timedelta(minutes=5)).isoformat(),
                                    (created + timedelta(minutes=10)).isoformat(),
                                    "5m:coinbase_fallback.rf.v1|10m:coinbase_fallback.rf.v1",
                                    json.dumps(scenario))])
            with patch.object(pit_oos_audit, "DB", db), patch.object(pit_oos_audit, "OUT", Path(td) / "audit.json"):
                result = pit_oos_audit.audit()
                self.assertFalse(result["ok"])
                self.assertFalse(result["pit_verified"])
                self.assertTrue(any("source:coinbase_futures:missing_prediction_cutoff" in v for v in result["violations"]))


    def test_precontract_coinbase_target_before_legacy_decision_is_quarantined(self):
        with tempfile.TemporaryDirectory() as td:
            created = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
            decision = datetime(2026, 9, 21, 12, 6, tzinfo=timezone.utc)
            scenario = {
                "decision_time_utc": decision.isoformat(),
                "provenance": {
                    "event_time": created.isoformat(),
                    "available_at": decision.isoformat(),
                    "retrieved_at": decision.isoformat(),
                    "prediction_cutoff": decision.isoformat(),
                    "sources": {
                        "coinbase_futures": {
                            "status": "ok",
                            "event_time": created.isoformat(),
                            "available_at": decision.isoformat(),
                            "retrieved_at": decision.isoformat(),
                            "prediction_cutoff": None,
                        }
                    },
                },
                "production_mode": "coinbase_fallback",
            }
            db = self.make_db(
                td,
                [(
                    1,
                    created.isoformat(),
                    "2026-09-21T12:05:00+00:00",
                    "2026-09-21T12:10:00+00:00",
                    "5m:coinbase_fallback.rf.v1|10m:coinbase_fallback.rf.v1",
                    json.dumps(scenario),
                )],
            )
            with patch.object(pit_oos_audit, "DB", db), patch.object(
                pit_oos_audit, "OUT", Path(td) / "audit.json"
            ):
                result = pit_oos_audit.audit()
                self.assertTrue(result["ok"], result)
                self.assertFalse(result["pit_verified"])
                self.assertEqual(result["legacy_unverified_count"], 1)
                self.assertTrue(any("5m_target_not_after_decision" in v for v in result["legacy_violations"]))

    def test_postcontract_coinbase_target_before_decision_remains_failure(self):
        with tempfile.TemporaryDirectory() as td:
            created = datetime(2026, 9, 22, 6, 0, tzinfo=timezone.utc)
            decision = datetime(2026, 9, 22, 6, 6, tzinfo=timezone.utc)
            scenario = {
                "decision_time_utc": decision.isoformat(),
                "provenance": {
                    "event_time": created.isoformat(),
                    "available_at": decision.isoformat(),
                    "retrieved_at": decision.isoformat(),
                    "prediction_cutoff": decision.isoformat(),
                    "sources": {
                        "coinbase_futures": {
                            "status": "ok",
                            "event_time": created.isoformat(),
                            "available_at": decision.isoformat(),
                            "retrieved_at": decision.isoformat(),
                            "prediction_cutoff": None,
                        }
                    },
                },
                "production_mode": "coinbase_fallback",
            }
            db = self.make_db(
                td,
                [(
                    1,
                    created.isoformat(),
                    "2026-09-22T06:05:00+00:00",
                    "2026-09-22T06:10:00+00:00",
                    "5m:coinbase_fallback.rf.v1|10m:coinbase_fallback.rf.v1",
                    json.dumps(scenario),
                )],
            )
            with patch.object(pit_oos_audit, "DB", db), patch.object(
                pit_oos_audit, "OUT", Path(td) / "audit.json"
            ):
                result = pit_oos_audit.audit()
                self.assertFalse(result["ok"])
                self.assertTrue(
                    any("5m_target_not_after_decision" in v for v in result["violations"])
                )

    def test_rejects_target_before_decision(self):
        with tempfile.TemporaryDirectory() as td:
            now = datetime.now(timezone.utc).replace(microsecond=0)
            db = self.make_db(td, [(1, now.isoformat(), (now - timedelta(minutes=1)).isoformat(), (now + timedelta(minutes=10)).isoformat(), "v1", "{}")])
            with patch.object(pit_oos_audit, "DB", db), patch.object(pit_oos_audit, "OUT", Path(td) / "audit.json"):
                result = pit_oos_audit.audit()
                self.assertFalse(result["ok"])
                self.assertTrue(any("5m_target_not_after_decision" in v for v in result["violations"]))
                self.assertEqual(result["violation_count"], 1)

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



    def test_tracks_strict_pit_readiness_for_adaptive_research(self):
        with tempfile.TemporaryDirectory() as td:
            created = datetime.now(timezone.utc).replace(microsecond=0)
            rows = []
            for i in range(2):
                at = (created + timedelta(seconds=i)).isoformat()
                scenario = {
                    "decision_time_utc": at,
                    "production_mode": "binance_primary",
                    "situation": {"market_state": "TREND_UP"},
                    "microstructure": {"vwap_distance_5m": 0.01},
                    "components": {"calibrated_5m": {"DOWN": 0.2, "FLAT": 0.2, "UP": 0.6}},
                    "provenance": {
                        "event_time": at,
                        "available_at": at,
                        "retrieved_at": at,
                        "prediction_cutoff": at,
                        "sources": {
                            "binance_futures": {
                                "status": "ok",
                                "event_time": at,
                                "available_at": at,
                                "retrieved_at": at,
                                "prediction_cutoff": at,
                            }
                        },
                    },
                }
                rows.append((
                    i + 1,
                    at,
                    (created + timedelta(minutes=5, seconds=i + 1)).isoformat(),
                    (created + timedelta(minutes=10, seconds=i + 1)).isoformat(),
                    "v1",
                    "UP",
                    "DOWN",
                    json.dumps(scenario),
                ))

            db = self.make_db(td, rows)
            with patch.object(pit_oos_audit, "DB", db), patch.object(
                pit_oos_audit, "OUT", Path(td) / "audit.json"
            ):
                result = pit_oos_audit.audit()

            self.assertTrue(result["ok"], result)
            self.assertEqual(result["coverage"]["5m"]["settled_predictions"], 2)
            self.assertEqual(result["coverage"]["5m"]["strict_primary_settled"], 2)
            self.assertEqual(result["coverage"]["5m"]["situation_meta_ready"], 2)
            self.assertEqual(result["coverage"]["10m"]["strict_primary_settled"], 2)
            self.assertEqual(result["coverage"]["10m"]["situation_meta_ready"], 2)
            self.assertEqual(result["coverage"]["5m"]["situation_meta_rows_needed"], 2998)
            self.assertEqual(result["coverage"]["10m"]["online_expert_rows_needed"], 138)


    def test_pit_history_records_once_per_15_minute_bucket(self):
        with tempfile.TemporaryDirectory() as td:
            history_dir = Path(td) / "pit_history"
            base = datetime(2026, 9, 26, 3, 8, tzinfo=timezone.utc)
            result = {
                "generated_at_utc": base.isoformat(),
                "status": "PASS",
                "pit_verified": True,
                "verified_primary_predictions": 400,
            }
            first = record_pit_history(result, history_dir)
            second = record_pit_history(
                {**result, "generated_at_utc": (base + timedelta(minutes=5)).isoformat()},
                history_dir,
            )
            self.assertEqual(first["status"], "RECORDED")
            self.assertEqual(second["status"], "SKIPPED_EXISTING_BUCKET")
            self.assertEqual(len(list(history_dir.glob("pit_*.json"))), 1)
            saved = json.loads(next(history_dir.glob("pit_*.json")).read_text())
            self.assertEqual(saved["history"]["bucket_start_utc"], "2026-09-26T03:00:00+00:00")


    def test_pit_history_retention_is_bounded(self):
        with tempfile.TemporaryDirectory() as td:
            history_dir = Path(td) / "pit_history"
            base = datetime(2026, 9, 20, 0, 0, tzinfo=timezone.utc)
            for i in range(HISTORY_RETENTION + 3):
                result = {
                    "generated_at_utc": (base + timedelta(minutes=15 * i)).isoformat(),
                    "status": "PASS",
                    "pit_verified": True,
                    "verified_primary_predictions": 400,
                }
                record_pit_history(result, history_dir)
            files = sorted(history_dir.glob("pit_*.json"))
            self.assertEqual(len(files), HISTORY_RETENTION)
            self.assertNotIn("pit_20260920T0000Z.json", {p.name for p in files})

if __name__ == "__main__":
    unittest.main()
