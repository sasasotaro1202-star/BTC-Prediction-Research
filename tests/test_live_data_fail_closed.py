import json
import os
import sqlite3
import unittest
from unittest.mock import patch

from src.live_data_fail_closed import validate_latest


class TestLiveDataFailClosed(unittest.TestCase):
    def test_deferred_cycle_bypasses_stale_prediction_check(self):
        with patch.dict(os.environ, {"BTC_LIVE_DEFERRED": "1"}, clear=False):
            result = validate_latest()
        self.assertEqual(result, {"ok": True, "mode": "deferred"})

    def test_accepts_explicit_coinbase_fallback_with_provenance_and_model_binding(self):
        scenario = {
            "production_mode": "coinbase_fallback",
            "policy": "coinbase_fallback_model+fallback_oos_calibration",
            "data_quality": {
                "binance_futures": "error:HTTPError:451",
                "binance_depth": "error:HTTPError",
                "binance_taker": "error:HTTPError",
                "binance_premium": "error:HTTPError",
            },
            "provenance": {
                "sources": {
                    "coinbase_futures": {
                        "status": "ok",
                    }
                }
            },
        }
        fake = (123, "coinbase_fallback.rf.v1", json.dumps(scenario))
        with patch("src.live_data_fail_closed.sqlite3.connect") as connect:
            con = connect.return_value.__enter__.return_value
            con.execute.return_value.fetchone.return_value = fake
            result = validate_latest()
        self.assertEqual(result, {"ok": True, "prediction_id": 123, "mode": "coinbase_fallback"})

    def test_rejects_failed_binance_inputs_for_normal_prediction(self):
        scenario = {
            "production_mode": "binance_primary",
            "data_quality": {
                "binance_futures": "error:HTTPError:451",
                "binance_depth": "error:HTTPError",
                "binance_taker": "error:HTTPError",
                "binance_premium": "error:HTTPError",
            },
        }
        fake = (124, "primary.v1", json.dumps(scenario))
        with patch("src.live_data_fail_closed.sqlite3.connect") as connect:
            con = connect.return_value.__enter__.return_value
            con.execute.return_value.fetchone.return_value = fake
            with self.assertRaisesRegex(SystemExit, "binance_futures"):
                validate_latest()


if __name__ == "__main__":
    unittest.main()
