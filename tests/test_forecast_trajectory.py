import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src import forecast_trajectory

class ForecastTrajectoryTests(unittest.TestCase):
    def test_all_horizons_and_unknown_outcomes_are_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            db=Path(td)/"predictions.db"
            with sqlite3.connect(db) as con:
                cols=["prediction_id INTEGER PRIMARY KEY","created_at_utc TEXT","base_price REAL"]
                for h in forecast_trajectory.ALL_HORIZONS:
                    cols += [f"target_{h} TEXT",f"p_down_{h} REAL",f"p_flat_{h} REAL",f"p_up_{h} REAL",
                             f"actual_price_{h} REAL",f"actual_direction_{h} TEXT",f"correct_{h} INTEGER",f"settled_{h}_at_utc TEXT"]
                con.execute("CREATE TABLE predictions ("+",".join(cols)+")")
                vals=[1,"2026-10-03T04:10:00+00:00",100000.0]
                for h in forecast_trajectory.ALL_HORIZONS:
                    vals += ["2026-10-03T04:15:00+00:00",.2,.3,.5,None,None,None,None]
                con.execute("INSERT INTO predictions VALUES ("+(",".join(["?"]*len(vals)))+")",vals)
            with patch.object(forecast_trajectory,"DB",db):
                out=forecast_trajectory.build()
            self.assertEqual(out["horizon_order"],list(forecast_trajectory.ALL_HORIZONS))
            self.assertEqual(out["primary_horizons"],["5m","10m"])
            self.assertEqual(out["extended_research_horizons"],["15m","30m","1h","3h","6h","12h","24h"])
            for h in forecast_trajectory.ALL_HORIZONS:
                p=out["horizons"][h]["points"][0]
                self.assertEqual(p["p_up"],.5)
                self.assertIsNone(p["actual_price"])
                self.assertIsNone(p["actual_direction"])
                self.assertIsNone(p["correct"])
                self.assertFalse(p["settled"])
            self.assertTrue(out["horizons"]["24h"]["research_only"])
            self.assertFalse(out["horizons"]["5m"]["research_only"])

    def test_restored_legacy_schema_is_migrated_before_export(self):
        with tempfile.TemporaryDirectory() as td:
            db=Path(td)/"predictions.db"
            with sqlite3.connect(db) as con:
                con.execute(
                    """CREATE TABLE predictions (
                      prediction_id INTEGER PRIMARY KEY,
                      created_at_utc TEXT NOT NULL,
                      base_price REAL NOT NULL,
                      target_5m TEXT, p_down_5m REAL, p_flat_5m REAL, p_up_5m REAL,
                      actual_price_5m REAL, actual_direction_5m TEXT, correct_5m INTEGER,
                      settled_5m_at_utc TEXT,
                      target_10m TEXT, p_down_10m REAL, p_flat_10m REAL, p_up_10m REAL,
                      actual_price_10m REAL, actual_direction_10m TEXT, correct_10m INTEGER,
                      settled_10m_at_utc TEXT
                    )"""
                )
                con.execute(
                    "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (1,"2026-10-03T04:10:00+00:00",100000.0,
                     "2026-10-03T04:15:00+00:00",0.2,0.3,0.5,None,None,None,None,
                     "2026-10-03T04:20:00+00:00",0.2,0.3,0.5,None,None,None,None),
                )
            with patch.object(forecast_trajectory,"DB",db), patch.object(forecast_trajectory,"init_db",lambda: None):
                out=forecast_trajectory.build()
            self.assertEqual(out["horizon_order"],list(forecast_trajectory.ALL_HORIZONS))
            self.assertEqual(out["horizons"]["5m"]["point_count"],1)
            self.assertEqual(out["horizons"]["10m"]["point_count"],1)
            for h in forecast_trajectory.EXTENDED_RESEARCH_HORIZONS:
                self.assertEqual(out["horizons"][h]["point_count"],0)
                self.assertEqual(out["horizons"][h]["availability_status"],"PENDING_FIRST_LIVE_SAMPLE")
            with sqlite3.connect(db) as con:
                columns={row[1] for row in con.execute("PRAGMA table_info(predictions)").fetchall()}
            self.assertIn("settled_24h_at_utc",columns)

    def test_extended_horizons_can_be_pending_without_inventing_history(self):
        with tempfile.TemporaryDirectory() as td:
            db=Path(td)/"predictions.db"
            with sqlite3.connect(db) as con:
                cols=["prediction_id INTEGER PRIMARY KEY","created_at_utc TEXT","base_price REAL"]
                for h in forecast_trajectory.ALL_HORIZONS:
                    cols += [f"target_{h} TEXT",f"p_down_{h} REAL",f"p_flat_{h} REAL",
                             f"p_up_{h} REAL",f"actual_price_{h} REAL",
                             f"actual_direction_{h} TEXT",f"correct_{h} INTEGER",
                             f"settled_{h}_at_utc TEXT"]
                con.execute("CREATE TABLE predictions ("+",".join(cols)+")")
                vals=[1,"2026-10-03T04:10:00+00:00",100000.0]
                for h in forecast_trajectory.ALL_HORIZONS[:2]:
                    vals += ["2026-10-03T04:15:00+00:00",.2,.3,.5,None,None,None,None]
                for h in forecast_trajectory.ALL_HORIZONS[2:]:
                    vals += ["2026-10-03T04:20:00+00:00",None,None,None,None,None,None,None]
                con.execute("INSERT INTO predictions VALUES ("+(",".join(["?"]*len(vals)))+")",vals)
            with patch.object(forecast_trajectory,"DB",db):
                out=forecast_trajectory.build()
            self.assertEqual(out["horizons"]["5m"]["availability_status"],"AVAILABLE")
            self.assertEqual(out["horizons"]["10m"]["availability_status"],"AVAILABLE")
            for h in forecast_trajectory.EXTENDED_RESEARCH_HORIZONS:
                self.assertEqual(out["horizons"][h]["availability_status"],"PENDING_FIRST_LIVE_SAMPLE")
                self.assertEqual(out["horizons"][h]["point_count"],0)
                self.assertEqual(out["horizons"][h]["points"],[])

if __name__=="__main__": unittest.main()
