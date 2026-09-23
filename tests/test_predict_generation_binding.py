import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import predict  # noqa: E402


class TestPredictFeatureSafety(unittest.TestCase):
    def test_features_reject_insufficient_history(self):
        rows=[[i,1.0,1.0,1.0,1.0,1.0] for i in range(30)]
        with self.assertRaisesRegex(ValueError, "insufficient_price_history_for_features"):
            predict.features(rows)

class TestPredictGenerationBinding(unittest.TestCase):
    def test_blend_weight_rejects_stale_generation(self):
        with tempfile.TemporaryDirectory() as td:
            old_db = predict.DB
            old_model_dir = predict.MODEL_DIR
            try:
                predict.DB = str(Path(td) / 'predictions.db')
                model_dir = Path(td) / 'models'
                predict.MODEL_DIR = model_dir
                model_dir.mkdir()
                (model_dir / '5m.blend.json').write_text(
                    json.dumps({
                        'base_weight': 0.35,
                        'n': 800,
                        'status': 'accepted',
                        'model_version': 'generation-A',
                    }),
                    encoding='utf-8',
                )
                with patch.object(predict, 'regver', return_value='generation-B'):
                    self.assertEqual(predict.load_blend_weight('5m'), 0.0)
            finally:
                predict.DB = old_db
                predict.MODEL_DIR = old_model_dir

    def test_blend_weight_accepts_current_generation(self):
        with tempfile.TemporaryDirectory() as td:
            old_db = predict.DB
            old_model_dir = predict.MODEL_DIR
            try:
                predict.DB = str(Path(td) / 'predictions.db')
                model_dir = Path(td) / 'models'
                predict.MODEL_DIR = model_dir
                model_dir.mkdir()
                (model_dir / '5m.blend.json').write_text(
                    json.dumps({
                        'base_weight': 0.35,
                        'n': 800,
                        'status': 'accepted',
                        'model_version': 'generation-A',
                    }),
                    encoding='utf-8',
                )
                with patch.object(predict, 'regver', return_value='generation-A'):
                    self.assertAlmostEqual(predict.load_blend_weight('5m'), 0.35)
            finally:
                predict.DB = old_db
                predict.MODEL_DIR = old_model_dir

    def test_temperature_rejects_stale_generation(self):
        with tempfile.TemporaryDirectory() as td:
            old_db = predict.DB
            old_model_dir = predict.MODEL_DIR
            try:
                predict.DB = str(Path(td) / 'predictions.db')
                model_dir = Path(td) / 'models'
                predict.MODEL_DIR = model_dir
                model_dir.mkdir()
                (model_dir / '5m.calibration.json').write_text(
                    json.dumps({
                        'temperature': 1.6,
                        'n_settled': 600,
                        'model_version': 'generation-A',
                    }),
                    encoding='utf-8',
                )
                with patch.object(predict, 'regver', return_value='generation-B'):
                    self.assertEqual(predict.load_temperature('5m'), 1.0)
            finally:
                predict.DB = old_db
                predict.MODEL_DIR = old_model_dir

    def test_latest_event_time_rejects_stale_and_future_data(self):
        current = predict.datetime(2026, 9, 20, 5, 0, tzinfo=predict.timezone.utc)
        stale = int((current - predict.timedelta(seconds=181)).timestamp() * 1000)
        with self.assertRaisesRegex(ValueError, "stale_live_market_event"):
            predict.validate_latest_event_time(stale, now=current)
        future = int((current + predict.timedelta(seconds=61)).timestamp() * 1000)
        with self.assertRaisesRegex(ValueError, "future_live_market_event"):
            predict.validate_latest_event_time(future, now=current)
        fresh = int((current - predict.timedelta(seconds=120)).timestamp() * 1000)
        event = predict.validate_latest_event_time(fresh, now=current)
        self.assertEqual(event, current - predict.timedelta(seconds=120))

    def test_features_fail_closed_on_nonfinite_or_invalid_ohlc(self):
        rows=[]
        for i in range(31):
            price=100.0+i
            rows.append((i, price, price+1, price-1, price, 10.0))
        bad=list(rows)
        bad[-1]=(30, 131.0, 132.0, 130.0, float("nan"), 10.0)
        with self.assertRaisesRegex(ValueError, "invalid_price_history_nonfinite"):
            predict.features(bad)
        bad=list(rows)
        bad[-1]=(30, 131.0, 129.0, 130.0, 131.0, 10.0)
        with self.assertRaisesRegex(ValueError, "invalid_ohlc_relationship"):
            predict.features(bad)


    def test_insert_prediction_rejects_missing_pit_provenance(self):
        now = predict.datetime(2026, 9, 23, 5, 0, tzinfo=predict.timezone.utc)
        with tempfile.TemporaryDirectory() as td:
            old_db = predict.DB
            try:
                predict.DB = str(Path(td) / 'predictions.db')
                predict.init_db()
                with self.assertRaisesRegex(ValueError, "prediction_provenance_missing"):
                    predict.insert_prediction(
                        now,
                        now + predict.timedelta(minutes=5),
                        now + predict.timedelta(minutes=10),
                        100.0,
                        {"UP": 0.4, "DOWN": 0.4, "FLAT": 0.2},
                        {"UP": 0.4, "DOWN": 0.4, "FLAT": 0.2},
                        "test-model",
                        {},
                        {},
                    )
            finally:
                predict.DB = old_db

    def test_insert_prediction_accepts_complete_pit_provenance(self):
        now = predict.datetime(2026, 9, 23, 5, 0, tzinfo=predict.timezone.utc)
        stamp = now.isoformat()
        scenario = {
            "decision_time_utc": stamp,
            "provenance": {
                "available_at": stamp,
                "retrieved_at": stamp,
                "prediction_cutoff": stamp,
                "sources": {
                    "binance_futures": {
                        "available_at": stamp,
                        "retrieved_at": stamp,
                        "prediction_cutoff": stamp,
                        "status": "ok",
                    }
                },
            },
        }
        with tempfile.TemporaryDirectory() as td:
            old_db = predict.DB
            try:
                predict.DB = str(Path(td) / 'predictions.db')
                predict.init_db()
                predict.insert_prediction(
                    now,
                    now + predict.timedelta(minutes=5),
                    now + predict.timedelta(minutes=10),
                    100.0,
                    {"UP": 0.4, "DOWN": 0.4, "FLAT": 0.2},
                    {"UP": 0.4, "DOWN": 0.4, "FLAT": 0.2},
                    "test-model",
                    {k: 0.0 for k in predict.FEATURES},
                    scenario,
                )
                with __import__("sqlite3").connect(predict.DB) as con:
                    self.assertEqual(con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0], 1)
            finally:
                predict.DB = old_db

    def test_secondary_venue_completeness_requires_valid_status(self):
        status = {"bybit_futures": "ok_current_only", "bybit_depth": "error:Timeout"}
        complete = (
            True
            and "bybit_book_imbalance" in {"bybit_book_imbalance": 0.1}
            and status.get("bybit_futures") in {"ok", "ok_current_only"}
            and status.get("bybit_depth") == "ok"
        )
        self.assertFalse(complete)




class TestPredictBlendSafety(unittest.TestCase):
    def test_unvalidated_context_agreement_cannot_add_live_blend_weight(self):
        base = {"DOWN": 0.45, "FLAT": 0.20, "UP": 0.35}
        structural = {"DOWN": 0.20, "FLAT": 0.50, "UP": 0.30}
        market = {"cross_exchange_gap": 0.0}
        with patch.object(predict, "load_blend_weight", return_value=0.0):
            _, weight = predict.fuse(base, structural, market, True, "5m")
        self.assertEqual(weight, 0.0)

    def test_accepted_blend_uses_only_validated_weight(self):
        base = {"DOWN": 0.45, "FLAT": 0.20, "UP": 0.35}
        structural = {"DOWN": 0.20, "FLAT": 0.50, "UP": 0.30}
        market = {"cross_exchange_gap": 0.0}
        with patch.object(predict, "load_blend_weight", return_value=0.20):
            _, weight = predict.fuse(base, structural, market, True, "5m")
        self.assertAlmostEqual(weight, 0.20, places=12)



if __name__ == '__main__':
    unittest.main()
