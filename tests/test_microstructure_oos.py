import unittest

from src.microstructure_oos import (
    BINANCE_MICRO,
    CROSS_VENUE,
    _micro_from_scenario,
    _strict_primary_sources_ok,
)


class MicrostructureOOSTests(unittest.TestCase):
    def _scenario(self):
        return {
            "microstructure": {
                "book_imbalance": 0.10,
                "taker_imbalance": -0.05,
                "funding_binance": 0.0001,
                "oi": 1000000.0,
                "cross_exchange_gap": 0.0002,
                "spot_futures_gap": -0.0001,
                "bybit_book_imbalance": 0.15,
            },
            "data_quality": {
                "bybit_depth": "ok",
                "bybit_futures": "ok_current_only",
            },
            "provenance": {
                "sources": {
                    name: {
                        "status": "ok",
                        "available_at": "2026-09-22T00:00:00+00:00",
                        "retrieved_at": "2026-09-22T00:00:00+00:00",
                        "prediction_cutoff": "2026-09-22T00:00:00+00:00",
                    }
                    for name in (
                        "binance_futures",
                        "binance_depth",
                        "binance_taker",
                        "binance_premium",
                    )
                }
            },
        }

    def test_binance_microstructure_is_complete_case_without_imputation(self):
        values = _micro_from_scenario(self._scenario(), cross_venue=False)
        self.assertEqual(set(values), set(BINANCE_MICRO))
        self.assertAlmostEqual(values["book_imbalance"], 0.10)
        self.assertAlmostEqual(values["taker_imbalance"], -0.05)
        self.assertAlmostEqual(values["funding_binance"], 0.0001)
        self.assertAlmostEqual(values["oi_log1p"], __import__("math").log1p(1000000.0))

    def test_cross_venue_requires_all_secondary_fields(self):
        values = _micro_from_scenario(self._scenario(), cross_venue=True)
        self.assertEqual(set(values), set(BINANCE_MICRO + CROSS_VENUE))
        scenario = self._scenario()
        scenario["data_quality"]["bybit_depth"] = "error:timeout"
        self.assertIsNone(_micro_from_scenario(scenario, cross_venue=True))

    def test_invalid_primary_field_fails_closed(self):
        scenario = self._scenario()
        scenario["microstructure"]["taker_imbalance"] = None
        self.assertIsNone(_micro_from_scenario(scenario, cross_venue=False))

    def test_primary_source_contract_requires_all_required_sources(self):
        self.assertTrue(_strict_primary_sources_ok(self._scenario()))
        scenario = self._scenario()
        del scenario["provenance"]["sources"]["binance_taker"]
        self.assertFalse(_strict_primary_sources_ok(scenario))


if __name__ == "__main__":
    unittest.main()
