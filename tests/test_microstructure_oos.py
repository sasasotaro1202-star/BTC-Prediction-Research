import unittest

from src.microstructure_oos import (
    BINANCE_MICRO,
    CROSS_VENUE,
    MARKET_FLOW_V2,
    _extended_from_feature_json,
    _market_flow_from_scenario,
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
                "binance_taker_window_transport": "websocket_closed_klines",
                "binance_taker_window_event_time_ms": 1_790_035_259_000,
                "binance_taker_window_retrieved_at_ms": 1_790_035_259_500,
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
                } | {
                    "binance_taker_window": {
                        "status": "ok",
                        "available_at": "2026-09-22T00:00:59+00:00",
                        "retrieved_at": "2026-09-22T00:00:59.500000+00:00",
                        "prediction_cutoff": "2026-09-22T00:00:59.500000+00:00",
                        "event_time": "2026-09-22T00:00:59+00:00",
                    }
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

    def test_extended_feature_parser_is_complete_case_without_imputation(self):
        payload = {
            "ret_15m": 0.001,
            "ret_30m": -0.002,
            "range_position_30m": 0.7,
            "trend_alignment": 0.0005,
        }
        values = _extended_from_feature_json(__import__("json").dumps(payload))
        self.assertEqual(set(values), {"ret_15m", "ret_30m", "range_position_30m", "trend_alignment"})
        self.assertAlmostEqual(values["ret_15m"], 0.001)

    def test_extended_feature_parser_rejects_missing_value(self):
        payload = {
            "ret_15m": 0.001,
            "ret_30m": -0.002,
            "range_position_30m": 0.7,
        }
        self.assertIsNone(_extended_from_feature_json(__import__("json").dumps(payload)))

    def test_market_flow_v2_is_complete_case_without_imputation(self):
        scenario = self._scenario()
        scenario["microstructure"].update({
            "vwap_distance_5m": 0.0001,
            "vwap_distance_15m": 0.0002,
            "vwap_distance_30m": 0.0003,
            "volume_burst_5m": 1.2,
            "volume_burst_15m": 1.1,
            "range_compression_5m": 0.8,
            "range_compression_15m": 0.9,
            "taker_imbalance_5m": 0.1,
            "taker_imbalance_15m": 0.05,
            "taker_imbalance_delta_5m_15m": 0.05,
        })
        values = _market_flow_from_scenario(scenario, created_at="2026-09-22T00:01:00+00:00")
        self.assertEqual(set(values), set(MARKET_FLOW_V2))
        self.assertAlmostEqual(values["vwap_distance_15m"], 0.0002)

    def test_market_flow_v2_fails_closed_when_any_feature_is_missing(self):
        scenario = self._scenario()
        scenario["microstructure"]["vwap_distance_15m"] = None
        self.assertIsNone(_market_flow_from_scenario(scenario))

    def test_market_flow_v2_rejects_missing_window_timing(self):
        scenario = self._scenario()
        scenario["microstructure"].update({
            "vwap_distance_5m": 0.0001,
            "vwap_distance_15m": 0.0002,
            "vwap_distance_30m": 0.0003,
            "volume_burst_5m": 1.2,
            "volume_burst_15m": 1.1,
            "range_compression_5m": 0.8,
            "range_compression_15m": 0.9,
            "taker_imbalance_5m": 0.1,
            "taker_imbalance_15m": 0.05,
            "taker_imbalance_delta_5m_15m": 0.05,
        })
        scenario["data_quality"].pop("binance_taker_window_transport", None)
        self.assertIsNone(_market_flow_from_scenario(scenario, created_at="2026-09-22T00:01:00+00:00"))

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
