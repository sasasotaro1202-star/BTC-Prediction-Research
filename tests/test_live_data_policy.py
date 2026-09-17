import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from live_data_policy import is_valid_status, validate_live_inputs  # noqa: E402


class TestLiveDataPolicy(unittest.TestCase):
    def good(self):
        return {k: "ok" for k in (
            "binance_futures", "binance_spot", "bybit_futures",
            "binance_depth", "bybit_depth", "binance_taker", "bybit_funding",
        )} | {"price_feature_fallback": "none"}

    def test_accepts_complete_primary_inputs(self):
        validate_live_inputs(self.good(), fut_rows=40, spot_rows=40, bybit_rows=40)
        self.assertTrue(is_valid_status(self.good(), fut_rows=40, spot_rows=40, bybit_rows=40))

    def test_rejects_any_failed_critical_source(self):
        status = self.good()
        status["binance_depth"] = "error:HTTPError"
        with self.assertRaisesRegex(ValueError, "binance_depth"):
            validate_live_inputs(status, fut_rows=40, spot_rows=40, bybit_rows=40)

    def test_rejects_fallback_even_if_rows_exist(self):
        status = self.good()
        status["price_feature_fallback"] = "coinbase"
        with self.assertRaisesRegex(ValueError, "price_feature_fallback"):
            validate_live_inputs(status, fut_rows=40, spot_rows=40, bybit_rows=40)

    def test_rejects_insufficient_history(self):
        with self.assertRaisesRegex(ValueError, "contiguous_history_insufficient"):
            validate_live_inputs(self.good(), fut_rows=39, spot_rows=40, bybit_rows=40)


if __name__ == "__main__":
    unittest.main()
