import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from live_data_policy import is_valid_status, validate_live_inputs  # noqa: E402


class TestLiveDataPolicy(unittest.TestCase):
    def good(self):
        return {k: "ok" for k in (
            "binance_futures", "bybit_futures", "binance_depth",
            "bybit_depth", "binance_taker", "binance_premium",
        )} | {
            "binance_spot": "error:HTTPError",
            "bybit_funding": "error:HTTPError",
            "price_feature_fallback": "none",
        }

    def test_accepts_complete_production_inputs_without_unused_sources(self):
        validate_live_inputs(self.good(), fut_rows=40, spot_rows=0, bybit_rows=40)
        self.assertTrue(is_valid_status(self.good(), fut_rows=40, spot_rows=0, bybit_rows=40))

    def test_accepts_current_only_bybit_price_when_history_is_fragmented(self):
        status = self.good()
        status["bybit_futures"] = "ok_current_only"
        validate_live_inputs(status, fut_rows=40, spot_rows=0, bybit_rows=1)
        self.assertTrue(is_valid_status(status, fut_rows=40, spot_rows=0, bybit_rows=1))

    def test_rejects_any_failed_production_critical_source(self):
        status = self.good()
        status["binance_depth"] = "error:HTTPError"
        with self.assertRaisesRegex(ValueError, "binance_depth"):
            validate_live_inputs(status, fut_rows=40, spot_rows=0, bybit_rows=40)

    def test_rejects_fallback_even_if_rows_exist(self):
        status = self.good()
        status["price_feature_fallback"] = "coinbase"
        with self.assertRaisesRegex(ValueError, "price_feature_fallback"):
            validate_live_inputs(status, fut_rows=40, spot_rows=0, bybit_rows=40)

    def test_rejects_insufficient_primary_history(self):
        with self.assertRaisesRegex(ValueError, "contiguous_history_insufficient"):
            validate_live_inputs(self.good(), fut_rows=39, spot_rows=0, bybit_rows=40)

    def test_allows_bybit_outage_as_explicit_optional_secondary_data(self):
        status = self.good()
        status["bybit_futures"] = "error:missing_current_price"
        status["bybit_depth"] = "error:HTTPError"
        validate_live_inputs(status, fut_rows=40, spot_rows=0, bybit_rows=0)
        self.assertTrue(is_valid_status(status, fut_rows=40, spot_rows=0, bybit_rows=0))

    def test_rejects_inconsistent_optional_bybit_rows(self):
        status = self.good()
        status["bybit_futures"] = "error:HTTPError"
        with self.assertRaisesRegex(ValueError, "bybit_futures_status_inconsistent"):
            validate_live_inputs(status, fut_rows=40, spot_rows=0, bybit_rows=1)

    def test_unused_spot_outage_does_not_block_prediction(self):
        status = self.good()
        status["binance_spot"] = "error:HTTPError"
        status["bybit_funding"] = "error:HTTPError"
        validate_live_inputs(status, fut_rows=40, spot_rows=0, bybit_rows=40)


if __name__ == "__main__":
    unittest.main()
