import unittest

from btc_event_layer import (
    aggregate_event_window,
    build_situation_card,
    event_from_binance_depth,
    event_from_binance_kline,
    make_event,
)


class TestBTCEventLayer(unittest.TestCase):
    def test_binance_kline_becomes_provenance_carrying_event(self):
        row = {
            "open_time_ms": 1_000,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
            "taker_buy_base": 6.0,
            "event_time_ms": 2_000,
            "retrieved_at_ms": 3_000,
        }
        event = event_from_binance_kline(row)
        self.assertEqual(event["event_type"], "kline_1m")
        self.assertEqual(event["source"], "binance_ws")
        self.assertEqual(event["available_at_ms"], 3_000)
        self.assertEqual(event["payload"]["taker_sell_base"], 4.0)
        self.assertAlmostEqual(event["payload"]["taker_imbalance"], 0.2)

    def test_depth_event_derives_spread_and_imbalance(self):
        snapshot = {
            "bids": [[100.0, 3.0], [99.9, 2.0]],
            "asks": [[100.1, 1.0], [100.2, 1.0]],
            "event_time_ms": 5_000,
            "retrieved_at_ms": 6_000,
            "last_update_id": 7,
        }
        event = event_from_binance_depth(snapshot)
        self.assertEqual(event["event_type"], "depth20")
        self.assertAlmostEqual(event["payload"]["bid_qty_5"], 5.0)
        self.assertAlmostEqual(event["payload"]["ask_qty_5"], 2.0)
        self.assertGreater(event["payload"]["depth_imbalance_5"], 0.0)
        self.assertGreater(event["payload"]["spread_bps"], 0.0)

    def test_missing_availability_is_fail_closed(self):
        with self.assertRaises(ValueError):
            make_event(
                source="test",
                venue="test",
                event_type="x",
                event_time_ms=1_000,
                available_at_ms=0,
                retrieved_at_ms=1_000,
                payload={},
            )

    def test_event_with_availability_after_retrieval_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "available_at_after_retrieval"):
            make_event(
                source="test",
                venue="test",
                event_type="x",
                event_time_ms=1_000,
                available_at_ms=3_000,
                retrieved_at_ms=2_000,
                payload={},
            )

    def test_future_event_is_excluded_without_zero_imputation(self):
        events = [
            make_event(
                source="test",
                venue="test",
                event_type="kline_1m",
                event_time_ms=9_000,
                available_at_ms=9_500,
                retrieved_at_ms=10_000,
                payload={"volume": 10.0, "taker_buy_base": 4.0},
                event_id="past",
            ),
            make_event(
                source="test",
                venue="test",
                event_type="kline_1m",
                event_time_ms=11_000,
                available_at_ms=11_500,
                retrieved_at_ms=12_000,
                payload={"volume": 99.0, "taker_buy_base": 99.0},
                event_id="future",
            ),
        ]
        result = aggregate_event_window(
            events,
            prediction_time_ms=10_000,
            window_ms=5_000,
        )
        self.assertEqual(result["event_ids"], ["past"])
        self.assertEqual(result["volume"], 10.0)

    def test_available_after_cutoff_in_relevant_window_fails_closed(self):
        event = make_event(
            source="test",
            venue="test",
            event_type="kline_1m",
            event_time_ms=9_000,
            available_at_ms=11_000,
            retrieved_at_ms=11_500,
            payload={"volume": 10.0, "taker_buy_base": 5.0},
        )
        with self.assertRaisesRegex(ValueError, "event_available_after_prediction_cutoff"):
            aggregate_event_window(
                [event],
                prediction_time_ms=10_000,
                window_ms=5_000,
            )

    def test_aggregate_trade_imbalance_is_causal(self):
        events = [
            make_event(
                source="binance_ws",
                venue="binance_usdm_futures",
                event_type="kline_1m",
                event_time_ms=7_000,
                available_at_ms=7_500,
                retrieved_at_ms=8_000,
                payload={"volume": 10.0, "taker_buy_base": 8.0},
                event_id="a",
            ),
            make_event(
                source="binance_ws",
                venue="binance_usdm_futures",
                event_type="kline_1m",
                event_time_ms=8_000,
                available_at_ms=8_500,
                retrieved_at_ms=9_000,
                payload={"volume": 10.0, "taker_buy_base": 4.0},
                event_id="b",
            ),
        ]
        result = aggregate_event_window(
            events,
            prediction_time_ms=9_000,
            window_ms=3_000,
        )
        self.assertEqual(result["event_count"], 2)
        self.assertAlmostEqual(result["buy_volume"], 12.0)
        self.assertAlmostEqual(result["sell_volume"], 8.0)
        self.assertAlmostEqual(result["trade_imbalance"], 0.2)
        self.assertTrue(result["pit_verified"])

    def test_situation_card_keeps_traceability_across_windows(self):
        event = make_event(
            source="binance_ws",
            venue="binance_usdm_futures",
            event_type="kline_1m",
            event_time_ms=100_000,
            available_at_ms=101_000,
            retrieved_at_ms=101_000,
            payload={"volume": 3.0, "taker_buy_base": 2.0},
            event_id="trace-1",
        )
        card = build_situation_card(
            [event],
            prediction_time_ms=101_000,
            windows_ms=(60_000, 300_000),
        )
        self.assertEqual(card["event_ids"], ["trace-1"])
        self.assertEqual(card["event_count"], 1)
        self.assertEqual(card["windows"]["60000"]["event_ids"], ["trace-1"])
        self.assertEqual(card["windows"]["300000"]["event_ids"], ["trace-1"])

    def test_invalid_event_provenance_fails_closed_in_strict_mode(self):
        event = {
            "event_id": "broken",
            "event_time_ms": 1_000,
            "retrieved_at_ms": 2_000,
            "payload": {},
        }
        with self.assertRaisesRegex(ValueError, "event_provenance_missing:available_at_ms"):
            aggregate_event_window(
                [event],
                prediction_time_ms=2_000,
                window_ms=2_000,
                strict_pit=True,
            )

    def test_event_availability_cannot_precede_event_time(self):
        with self.assertRaisesRegex(ValueError, "event_time_after_availability"):
            make_event(
                source="test",
                venue="test",
                event_type="x",
                event_time_ms=3_000,
                available_at_ms=2_000,
                retrieved_at_ms=4_000,
                payload={},
            )


    def test_rest_seed_cache_is_not_mislabeled_as_websocket(self):
        import json
        import tempfile
        from pathlib import Path
        from scripts.build_btc_event_situation_card import load_events

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "kline.json").write_text(json.dumps({
                "schema_version": 1,
                "source": "Binance USD-M Futures WebSocket",
                "recovery_epoch_ms": 2_500,
                "rows": [{
                    "open_time_ms": 1_000, "open": 100, "high": 101, "low": 99,
                    "close": 100.5, "volume": 5, "taker_buy_base": 3,
                    "event_time_ms": 2_000, "retrieved_at_ms": 2_500,
                }],
            }), encoding="utf-8")
            events = load_events(root / "kline.json", root / "missing.json")
            self.assertEqual(events[0]["source"], "binance_futures_rest_seed")

if __name__ == "__main__":
    unittest.main()
