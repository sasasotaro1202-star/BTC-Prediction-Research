import asyncio
import json
import sys
import unittest
from unittest.mock import AsyncMock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import binance_ws


class TestBinanceWebSocket(unittest.TestCase):
    def test_documented_raw_endpoints_are_primary_with_legacy_fallbacks(self):
        self.assertEqual(binance_ws.KLINE_URL, "wss://fstream.binance.com/market/ws/btcusdt@kline_1m")
        self.assertEqual(binance_ws.DEPTH_URL, "wss://fstream.binance.com/market/ws/btcusdt@depth20@100ms")
        self.assertEqual(binance_ws.MARK_URL, "wss://fstream.binance.com/market/ws/btcusdt@markPrice@1s")
        self.assertIn("wss://fstream.binance.com/ws/btcusdt@kline_1m", binance_ws.KLINE_FALLBACK_URLS)
        self.assertIn("wss://fstream.binance.com/market/stream?streams=btcusdt@kline_1m", binance_ws.KLINE_FALLBACK_URLS)
        self.assertIn("wss://fstream.binance.com/ws/btcusdt@depth20@100ms", binance_ws.DEPTH_FALLBACK_URLS)
        self.assertIn("wss://fstream.binance.com/market/stream?streams=btcusdt@depth20@100ms", binance_ws.DEPTH_FALLBACK_URLS)
        self.assertIn("wss://fstream.binance.com/ws/btcusdt@markPrice@1s", binance_ws.MARK_FALLBACK_URLS)
        self.assertIn("wss://fstream.binance.com/market/stream?streams=btcusdt@markPrice@1s", binance_ws.MARK_FALLBACK_URLS)

    def test_collect_with_fallback_uses_legacy_when_primary_is_empty(self):
        primary = "wss://primary"
        legacy = "wss://legacy"
        fallback_rows = [{"open_time_ms": 1}]
        async_mock = AsyncMock(side_effect=[[], fallback_rows])
        with patch.object(binance_ws, "_collect_url", async_mock):
            rows, source = asyncio.run(
                binance_ws._collect_with_fallback(
                    (primary, legacy), 1.0, binance_ws.parse_kline_message
                )
            )
        self.assertEqual(rows, fallback_rows)
        self.assertEqual(source, legacy)
        self.assertEqual([call.args[0] for call in async_mock.await_args_list], [primary, legacy])

    def test_collect_with_fallback_stops_after_primary_success(self):
        primary = "wss://primary"
        legacy = "wss://legacy"
        primary_rows = [{"open_time_ms": 2}]
        async_mock = AsyncMock(return_value=primary_rows)
        with patch.object(binance_ws, "_collect_url", async_mock):
            rows, source = asyncio.run(
                binance_ws._collect_with_fallback(
                    (primary, legacy), 1.0, binance_ws.parse_kline_message
                )
            )
        self.assertEqual(rows, primary_rows)
        self.assertEqual(source, primary)
        async_mock.assert_awaited_once_with(primary, 1.0, binance_ws.parse_kline_message)

    def test_kline_parser_requires_closed_one_minute_bar(self):
        base = {
            "e": "kline",
            "E": 1_800_000_060_500,
            "k": {
                "t": 1_800_000_000_000, "o": "100", "h": "102", "l": "99",
                "c": "101", "v": "10", "V": "4", "i": "1m",
            },
        }
        self.assertIsNone(binance_ws.parse_kline_message({**base, "k": {**base["k"], "x": False}}, 1_800_000_061_000))
        row = binance_ws.parse_kline_message({**base, "k": {**base["k"], "x": True}}, 1_800_000_061_000)
        self.assertIsNotNone(row)
        self.assertEqual(row["open_time_ms"], 1_800_000_000_000)
        self.assertEqual(row["taker_buy_base"], 4.0)
        self.assertEqual(row["retrieved_at_ms"], 1_800_000_061_000)

    def test_kline_parser_accepts_combined_stream_wrapper(self):
        message = {
            "stream": "btcusdt@kline_1m",
            "data": {
                "e": "kline", "E": 1_800_000_060_500,
                "k": {
                    "t": 1_800_000_000_000, "o": "100", "h": "102", "l": "99",
                    "c": "101", "v": "10", "V": "4", "i": "1m", "x": True,
                },
            },
        }
        self.assertIsNotNone(binance_ws.parse_kline_message(message, 1_800_000_061_000))

    def test_depth_parser_is_bounded_and_validates_crossed_book(self):
        good = {"e": "depthUpdate", "E": 1_800_000_001_100, "bids": [["100", "5"], ["99", "3"]], "asks": [["101", "1"], ["102", "1"]]}
        bad = {"e": "depthUpdate", "bids": [["102", "5"]], "asks": [["101", "1"]]}
        parsed = binance_ws.parse_depth_message(good, 1_800_000_001_000)
        self.assertEqual(len(parsed["bids"]), 2)
        self.assertEqual(parsed["event_time_ms"], 1_800_000_001_100)
        self.assertEqual(parsed["retrieved_at_ms"], 1_800_000_001_000)
        self.assertIsNone(binance_ws.parse_depth_message(bad, 1_800_000_001_000))

    def test_mark_price_parser_keeps_exchange_event_time(self):
        message = {"e": "markPriceUpdate", "E": 1_800_000_060_100, "p": "100.5", "r": "0.0001"}
        parsed = binance_ws.parse_mark_price_message(message, 1_800_000_061_000)
        self.assertEqual(parsed["mark_price"], 100.5)
        self.assertEqual(parsed["funding_rate"], 0.0001)
        self.assertEqual(parsed["event_time_ms"], 1_800_000_060_100)

    def test_merge_cache_deduplicates_open_time_without_losing_newer_row(self):
        older = {"open_time_ms": 1_800_000_000_000, "open": 1, "high": 2, "low": 1, "close": 2,
                 "volume": 10, "taker_buy_base": 4, "event_time_ms": 1_800_000_060_000, "retrieved_at_ms": 1_800_000_061_000}
        newer = {**older, "close": 2.5, "event_time_ms": 1_800_000_060_500}
        merged = binance_ws.merge_cache([older], [newer])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["close"], 2.5)

    def test_taker_imbalance_requires_contiguous_recent_closed_bars(self):
        rows = []
        for i in range(5):
            rows.append({
                "open_time_ms": 1_800_000_000_000 + i * 60_000,
                "open": 100, "high": 101, "low": 99, "close": 100,
                "volume": 10, "taker_buy_base": 7,
                "event_time_ms": 1_800_000_060_000 + i * 60_000,
                "retrieved_at_ms": 1_800_000_061_000 + i * 60_000,
            })
        value, event_time = binance_ws.taker_imbalance(rows, 5)
        self.assertAlmostEqual(value, 0.4)
        self.assertEqual(event_time, rows[-1]["event_time_ms"])

    def test_continuous_capture_preserves_existing_rows_and_emits_checkpoint(self):
        initial = [{
            "open_time_ms": 1_800_000_000_000,
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
            "volume": 10.0, "taker_buy_base": 4.0,
            "event_time_ms": 1_800_000_060_000,
            "retrieved_at_ms": 1_800_000_061_000,
        }]
        incoming = {
            **initial[0],
            "open_time_ms": 1_800_000_060_000,
            "event_time_ms": 1_800_000_120_000,
            "retrieved_at_ms": 1_800_000_121_000,
            "close": 101.0,
        }
        checkpoints = []

        async def fake_stream(url, timeout_seconds, parser, on_row):
            await on_row(incoming)
            return 1, True

        async def checkpoint(rows):
            checkpoints.append(list(rows))

        with patch.object(binance_ws, "_stream_url", new=AsyncMock(side_effect=fake_stream)):
            rows = asyncio.run(
                binance_ws.capture_closed_klines_stream(
                    timeout_seconds=0.01,
                    checkpoint_seconds=0.0,
                    initial_rows=initial,
                    on_checkpoint=checkpoint,
                )
            )

        opens = [row["open_time_ms"] for row in rows]
        self.assertIn(initial[0]["open_time_ms"], opens)
        self.assertIn(incoming["open_time_ms"], opens)
        self.assertGreaterEqual(len(checkpoints), 1)

    def test_continuous_capture_reconnects_after_primary_socket_drop(self):
        incoming = {
            "open_time_ms": 1_800_000_000_000,
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
            "volume": 10.0, "taker_buy_base": 4.0,
            "event_time_ms": 1_800_000_060_000,
            "retrieved_at_ms": 1_800_000_061_000,
        }
        calls = []

        async def fake_stream(url, timeout_seconds, parser, on_row):
            calls.append(url)
            await on_row(incoming)
            return 1, True

        async def no_sleep(_seconds):
            return None

        with patch.object(binance_ws, "_stream_url", new=AsyncMock(side_effect=fake_stream)):
            with patch.object(binance_ws.asyncio, "sleep", new=AsyncMock(side_effect=no_sleep)):
                asyncio.run(
                    binance_ws.capture_closed_klines_stream(
                        timeout_seconds=0.02,
                        checkpoint_seconds=999.0,
                        initial_rows=[],
                        on_checkpoint=None,
                    )
                )

        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(all(url == binance_ws.KLINE_URL for url in calls))

    def test_direct_depth_snapshot_retries_after_transport_drop(self):
        incoming = {
            "bids": [["100.0", "2.0"], ["99.9", "1.0"]],
            "asks": [["100.1", "2.5"], ["100.2", "1.5"]],
            "last_update_id": 123,
            "event_time_ms": 1_800_000_001_100,
            "retrieved_at_ms": 1_800_000_001_000,
        }
        calls = []

        async def fake_stream(url, timeout_seconds, parser, on_row):
            calls.append(url)
            if len(calls) == 1:
                return 0, True
            await on_row(incoming)
            return 1, True

        async def no_sleep(_seconds):
            return None

        with patch.object(binance_ws, "_stream_url", new=AsyncMock(side_effect=fake_stream)):
            with patch.object(binance_ws.asyncio, "sleep", new=AsyncMock(side_effect=no_sleep)):
                result = asyncio.run(binance_ws.capture_depth_snapshot(0.02))

        self.assertEqual(result["event_time_ms"], incoming["event_time_ms"])
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(calls[0], binance_ws.DEPTH_URL)

    def test_depth_cache_round_trip_and_freshness_guard(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        now_ms = int(binance_ws.time.time() * 1000)
        snapshot = {
            "bids": [["100.0", "2.0"], ["99.9", "1.0"]],
            "asks": [["100.1", "2.5"], ["100.2", "1.5"]],
            "last_update_id": 123,
            "event_time_ms": now_ms - 1000,
            "retrieved_at_ms": now_ms,
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "depth.json"
            binance_ws.write_depth_cache(snapshot, path)
            loaded = binance_ws.load_depth_cache(path, max_age_ms=180_000)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["event_time_ms"], snapshot["event_time_ms"])
            self.assertEqual(loaded["retrieved_at_ms"], snapshot["retrieved_at_ms"])

            obj = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(obj["stream"], "btcusdt@depth20@100ms")
            with patch.object(
                binance_ws.time,
                "time",
                return_value=(snapshot["retrieved_at_ms"] + 180_001) / 1000,
            ):
                self.assertIsNone(
                    binance_ws.load_depth_cache(path, max_age_ms=180_000)
                )


    def test_cache_round_trip(self):
        row = {
            "open_time_ms": 1_800_000_000_000, "open": 100, "high": 101, "low": 99, "close": 100.5,
            "volume": 10, "taker_buy_base": 4, "event_time_ms": 1_800_000_060_000, "retrieved_at_ms": 1_800_000_061_000,
        }
        with TemporaryDirectory() as td:
            path = Path(td) / "cache.json"
            binance_ws.write_cache([row], path)
            obj = json.loads(path.read_text())
            self.assertEqual(obj["source"], "Binance USD-M Futures WebSocket")
            loaded = binance_ws.load_cache(path)
            self.assertEqual(loaded[0]["open_time_ms"], row["open_time_ms"])


if __name__ == "__main__":
    unittest.main()
