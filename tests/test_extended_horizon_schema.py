import sqlite3
import tempfile
import unittest
from pathlib import Path

from src import db


class ExtendedHorizonSchemaTests(unittest.TestCase):
    def test_init_db_adds_all_extended_horizon_columns(self):
        with tempfile.TemporaryDirectory() as td:
            old_db = db.DB
            try:
                db.DB = Path(td) / "predictions.db"
                db.init_db()
                with sqlite3.connect(db.DB) as con:
                    columns = {row[1] for row in con.execute("PRAGMA table_info(predictions)").fetchall()}
                for horizon in ("15m", "30m", "1h", "3h", "6h", "12h", "24h"):
                    for prefix in (
                        "target_", "p_up_", "p_down_", "p_flat_",
                        "actual_price_", "actual_direction_", "correct_",
                        "settled_", "settlement_source_",
                    ):
                        self.assertIn(prefix + horizon, columns)
            finally:
                db.DB = old_db


if __name__ == "__main__":
    unittest.main()
