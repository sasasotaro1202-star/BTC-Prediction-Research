import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import src.reconcile_fallback_settlements as reconcile
import db as db_module

class TestReconcileFallbackSettlements(unittest.TestCase):
    def test_reopens_only_unknown_or_wrong_fallback_rows(self):
        with tempfile.TemporaryDirectory() as td:
            db=Path(td)/'predictions.db'
            old=reconcile.DB
            old_db_module=db_module.DB
            try:
                reconcile.DB=db
                db_module.DB=db
                reconcile.init_db()
                base=('2026-09-24T00:00:00+00:00','2026-09-24T00:05:00+00:00','2026-09-24T00:10:00+00:00',100,.4,.3,.3,.4,.3,.3,'v','{}')
                def ins(s,src):
                    with sqlite3.connect(db) as con:
                        con.execute("INSERT INTO predictions(created_at_utc,target_5m,target_10m,base_price,p_up_5m,p_down_5m,p_flat_5m,p_up_10m,p_down_10m,p_flat_10m,model_version,feature_json,scenario_json,actual_price_5m,settlement_source_5m) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", base+(json.dumps(s),101,src))
                ins({'production_mode':'coinbase_fallback'},None)
                ins({'production_mode':'bybit_fallback'},'bybit_linear')
                reconcile.main()
                with sqlite3.connect(db) as con:
                    rows=con.execute('SELECT actual_price_5m,settlement_source_5m FROM predictions ORDER BY prediction_id').fetchall()
                self.assertEqual(rows[0],(None,None))
                self.assertEqual(rows[1],(101.0,'bybit_linear'))
            finally:
                reconcile.DB=old
                db_module.DB=old_db_module

if __name__ == '__main__':
    unittest.main()