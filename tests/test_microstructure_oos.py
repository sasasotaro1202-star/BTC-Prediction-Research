import unittest

from src.microstructure_oos import (
    BASE_FEATURES,
    BINANCE_MICRO,
    BINANCE_MICRO_CORE,
    BYBIT_OI,
    CROSS_VENUE,
    EXTENDED_FEATURES,
    MARKET_FLOW_V2,
    _bybit_oi_from_scenario,
    _extended_from_feature_json,
    _market_flow_from_scenario,
    _micro_from_scenario,
    _strict_primary_sources_ok,
    coverage_diagnostics,
)


class MicrostructureOOSTests(unittest.TestCase):
    def test_coverage_diagnostics_is_non_mutating_and_exposes_ratios(self):
        from unittest.mock import patch

        with patch(
            "src.microstructure_oos.load_primary_production_strict_rows",
            return_value=[],
        ):
            out = coverage_diagnostics("5m")
        self.assertEqual(out["strict_primary_rows"], 0)
        self.assertEqual(out["variant_coverage"]["binance_micro"]["complete_rows"], 0)
        self.assertEqual(out["variant_coverage"]["binance_micro"]["coverage_ratio"], 0.0)
        self.assertEqual(out["field_presence"]["book_imbalance"], 0)

    def test_oi_independent_variant_schema_and_feature_counts(self):
        self.assertEqual(BINANCE_MICRO_CORE, (
            "book_imbalance", "taker_imbalance", "funding_binance"
        ))
        self.assertEqual(len(BINANCE_MICRO), len(BINANCE_MICRO_CORE) + 1)

    def test_bybit_oi_variant_schema(self):
        self.assertEqual(BYBIT_OI, ("bybit_oi_log1p",))

    def test_feature_variant_manifest_includes_oi_independent_paths(self):
        variants = {
            "binance_core": list(BINANCE_MICRO_CORE),
            "market_flow_v2_no_oi": list(BINANCE_MICRO_CORE + MARKET_FLOW_V2),
            "cross_venue_no_oi": list(BINANCE_MICRO_CORE + CROSS_VENUE),
        }
        self.assertEqual(len(variants["binance_core"]), 3)
        self.assertNotIn("oi_log1p", variants["binance_core"])
        self.assertNotIn("oi_log1p", variants["market_flow_v2_no_oi"])
        self.assertNotIn("oi_log1p", variants["cross_venue_no_oi"])

    def test_feature_variant_manifest_matches_output_variants(self):
        expected = {
            "binance_micro": list(BASE_FEATURES + BINANCE_MICRO),
            "market_flow_v2": list(BASE_FEATURES + BINANCE_MICRO + MARKET_FLOW_V2),
            "full_stack": list(BASE_FEATURES + EXTENDED_FEATURES + BINANCE_MICRO + MARKET_FLOW_V2),
            "cross_venue": list(BASE_FEATURES + BINANCE_MICRO + CROSS_VENUE),
            "binance_core_bybit_oi": list(BASE_FEATURES + BINANCE_MICRO_CORE + BYBIT_OI),
            "market_flow_v2_no_oi_bybit_oi": list(BASE_FEATURES + BINANCE_MICRO_CORE + MARKET_FLOW_V2 + BYBIT_OI),
            "full_stack_no_oi_bybit_oi": list(BASE_FEATURES + EXTENDED_FEATURES + BINANCE_MICRO_CORE + MARKET_FLOW_V2 + BYBIT_OI),
        }
        self.assertEqual(
            set(expected),
            {
                "binance_micro",
                "market_flow_v2",
                "full_stack",
                "cross_venue",
                "binance_core_bybit_oi",
                "market_flow_v2_no_oi_bybit_oi",
                "full_stack_no_oi_bybit_oi",
            },
        )
        self.assertEqual(len(expected["full_stack"]), 33)
        self.assertEqual(len(expected["full_stack_no_oi_bybit_oi"]), 33)

    def _scenario(self):
        return {
            "microstructure": {
                "book_imbalance": 0.10,
                "taker_imbalance": -0.05,
                "funding_binance": 0.0001,
                "oi": 1000000.0,
                "bybit_oi": 1000000.0,
                "cross_exchange_gap": 0.0002,
                "spot_futures_gap": -0.0001,
                "bybit_book_imbalance": 0.15,
            },
            "data_quality": {
                "bybit_depth": "ok",
                "bybit_futures": "ok_current_only",
                "bybit_oi": "ok",
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
                    },
                    "bybit_ticker": {
                        "status": "ok",
                        "available_at": "2026-09-22T00:00:00+00:00",
                        "retrieved_at": "2026-09-22T00:00:00+00:00",
                        "prediction_cutoff": "2026-09-22T00:00:00+00:00",
                        "fields": ["openInterest"],
                    }
                }
            },
        }

    def test_bybit_oi_requires_own_source_provenance(self):
        values = _bybit_oi_from_scenario(self._scenario(), created_at="2026-09-22T00:00:00+00:00")
        self.assertEqual(set(values), set(BYBIT_OI))
        self.assertAlmostEqual(values["bybit_oi_log1p"], __import__("math").log1p(1000000.0))
        scenario = self._scenario()
        del scenario["provenance"]["sources"]["bybit_ticker"]
        self.assertIsNone(_bybit_oi_from_scenario(scenario, created_at="2026-09-22T00:00:00+00:00"))

    def test_bybit_oi_rejects_source_available_after_cutoff(self):
        scenario = self._scenario()
        scenario["provenance"]["sources"]["bybit_ticker"] = {
            "status": "ok",
            "available_at": "2026-09-22T00:00:02+00:00",
            "retrieved_at": "2026-09-22T00:00:01+00:00",
            "prediction_cutoff": "2026-09-22T00:00:01+00:00",
            "fields": ["openInterest"],
        }
        self.assertIsNone(_bybit_oi_from_scenario(scenario, created_at="2026-09-22T00:00:02+00:00"))

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
