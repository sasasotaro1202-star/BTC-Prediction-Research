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
        self.assertEqual(binance_ws.DEPTH_URL, "wss://fstream.binance.com/public/ws/btcusdt@depth20@100ms")
        self.assertEqual(binance_ws.MARK_URL, "wss://fstream.binance.com/market/ws/btcusdt@markPrice@1s")
        self.assertEqual(
            binance_ws.KLINE_FALLBACK_URLS,
            ("wss://fstream.binance.com/market/stream?streams=btcusdt@kline_1m",),
        )
        self.assertEqual(
            binance_ws.DEPTH_FALLBACK_URLS,
            ("wss://fstream.binance.com/public/stream?streams=btcusdt@depth20@100ms",),
        )
        self.assertEqual(
            binance_ws.MARK_FALLBACK_URLS,
            ("wss://fstream.binance.com/market/stream?streams=btcusdt@markPrice@1s",),
        )
        self.assertFalse(any(
            "/fstream.binance.com/ws/" in url
            and "/market/" not in url
            and "/public/" not in url
            for urls in (
                binance_ws.KLINE_FALLBACK_URLS,
                binance_ws.DEPTH_FALLBACK_URLS,
                binance_ws.MARK_FALLBACK_URLS,
            )
            for url in urls
        ))

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

    def test_partial_depth_parser_accepts_current_binance_b_a_payload(self):
        received = 1_800_000_001_000
        message = {
            "e": "depthUpdate",
            "E": 1_800_000_000_900,
            "T": 1_800_000_000_899,
            "s": "BTCUSDT",
            "U": 390497796,
            "u": 390497878,
            "pu": 390497794,
            "ps": "BTCUSDT",
            "st": 1,
            "b": [["100.0", "5.0"], ["99.9", "3.0"]],
            "a": [["100.1", "1.0"], ["100.2", "1.0"]],
        }
        parsed = binance_ws.parse_depth_message(message, received)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["last_update_id"], 390497878)
        self.assertEqual(parsed["event_time_ms"], 1_800_000_000_900)
        self.assertEqual(parsed["retrieved_at_ms"], received)
        self.assertEqual(parsed["bids"][0], [100.0, 5.0])
        self.assertEqual(parsed["asks"][0], [100.1, 1.0])

    def test_partial_depth_parser_rejects_missing_exchange_event_time(self):
        message = {
            "e": "depthUpdate",
            "s": "BTCUSDT",
            "b": [["100.0", "5.0"]],
            "a": [["100.1", "1.0"]],
        }
        self.assertIsNone(binance_ws.parse_depth_message(message, 1_800_000_001_000))

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
            self.assertEqual(obj["snapshot"]["E"], snapshot["event_time_ms"])
            with patch.object(
                binance_ws.time,
                "time",
                return_value=(snapshot["retrieved_at_ms"] + 180_001) / 1000,
            ):
                self.assertIsNone(
                    binance_ws.load_depth_cache(path, max_age_ms=180_000)
                )


    def test_depth_publisher_skips_stale_remote_overlap_fail_closed(self):
        workflow = (ROOT / ".github" / "workflows" / "btc_binance_ws_collector.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("remote_retrieved >= local_retrieved", workflow)
        self.assertIn('if [ "$rc" -eq 11 ]; then', workflow)
        self.assertIn("return 0", workflow)
        guard_start = workflow.index("publish_depth_cache() {")
        guard_end = workflow.index("# Keep one WebSocket connection open", guard_start)
        guard = workflow[guard_start:guard_end]
        self.assertLess(guard.index("remote_retrieved >= local_retrieved"), guard.index("for attempt in 1 2 3 4; do"))
        self.assertIn("raise SystemExit(11)", guard)

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


    def test_collector_has_same_product_rest_cache_seed_for_stale_ws(self):
        workflow = (ROOT / ".github" / "workflows" / "btc_binance_ws_collector.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("Seed stale Binance Futures cache from closed REST klines", workflow)
        self.assertIn("https://fapi.binance.com/fapi/v1/klines?", workflow)
        self.assertIn('"symbol": "BTCUSDT"', workflow)
        self.assertIn('"interval": "1m"', workflow)
        self.assertIn('"event_time_ms": close_ms', workflow)
        self.assertIn('"retrieved_at_ms": retrieved_ms', workflow)
        self.assertIn("if suffix < 40:", workflow)
        self.assertIn("Continuing to WebSocket capture.", workflow)
        self.assertIn("local last_published=0", workflow)
        self.assertIn("semantic no-op publish", workflow)

    def test_collector_publisher_loop_has_single_function_definition(self):
        workflow = (ROOT / ".github" / "workflows" / "btc_binance_ws_collector.yml").read_text(encoding="utf-8")
        self.assertEqual(workflow.count("          publisher_loop() {"), 1)
        self.assertIn('publisher_loop > /tmp/btc_ws_publisher.log 2>&1 &', workflow)


    def test_collector_refuses_gappy_checkpoint_publication(self):
        workflow = (ROOT / ".github" / "workflows" / "btc_binance_ws_collector.yml").read_text(encoding="utf-8")
        start = workflow.index("          publish_cache() {")
        end = workflow.index("          publish_depth_cache() {", start)
        block = workflow[start:end]
        self.assertIn('contiguous_tail="$(python - "$local_file"', block)
        self.assertIn('if [ "$contiguous_tail" -lt 40 ]; then', block)
        self.assertIn('refusing to publish gappy Binance WS checkpoint', block)
        self.assertIn('return 0', block)


    def test_collector_rest_seed_repairs_internal_cache_gaps(self):
        workflow = (ROOT / ".github" / "workflows" / "btc_binance_ws_collector.yml").read_text(encoding="utf-8")
        self.assertIn("return len(ordered), suffix, age", workflow)
        self.assertIn("row_count, suffix, age = fresh_contiguous_suffix(existing)", workflow)
        self.assertIn("suffix == row_count", workflow)
        self.assertIn("Existing Binance WS cache requires REST healing", workflow)


    def test_collector_push_does_not_trigger_on_test_only_changes(self):
        workflow = (ROOT / ".github" / "workflows" / "btc_binance_ws_collector.yml").read_text(encoding="utf-8")
        self.assertNotIn("      - 'tests/test_binance_ws.py'", workflow)
        self.assertIn("  cancel-in-progress: false", workflow)
        self.assertIn("github.event_name == 'schedule' && github.run_id || github.sha", workflow)


    def test_collector_publishes_initial_cache_before_checkpoint_loop(self):
        workflow = (ROOT / ".github" / "workflows" / "btc_binance_ws_collector.yml").read_text(encoding="utf-8")
        start = workflow.index("          publisher_loop() {")
        end = workflow.index("          while kill -0", start)
        block = workflow[start:end]
        self.assertIn('if [ -s data/binance_ws_1m.json ]; then', block)
        self.assertIn('if publish_cache; then', block)
        self.assertIn('Initial Binance WS cache checkpoint published', block)
        self.assertIn('initial Binance WS cache checkpoint publication failed', block)

    def test_collector_uses_curl_fallback_for_rest_seed(self):
        workflow = (ROOT / ".github" / "workflows" / "btc_binance_ws_collector.yml").read_text(encoding="utf-8")
        start = workflow.index("      - name: Seed stale Binance Futures cache from closed REST klines")
        end = workflow.index("      - name: Record fixed collector base SHA", start)
        block = workflow[start:end]
        self.assertIn('curl', block)
        self.assertIn('--retry-all-errors', block)
        self.assertIn('fapi.binance.com/fapi/v1/klines?', block)
        self.assertIn('curl_exit_', block)

    def test_publisher_loop_avoids_heredoc_command_substitutions(self):
        workflow = (ROOT / ".github" / "workflows" / "btc_binance_ws_collector.yml").read_text(encoding="utf-8")
        start = workflow.index("          publisher_loop() {")
        end = workflow.index("          publisher_loop > /tmp/btc_ws_publisher.log", start)
        block = workflow[start:end]
        self.assertNotIn("<<'PY'", block)
        self.assertIn('last_depth_published="$(python -c', block)
        self.assertIn('current_latest="$(python -c', block)
        self.assertIn('current_depth="$(python -c', block)

    def test_collector_capture_step_avoids_heredoc_command_substitutions(self):
        workflow = (ROOT / ".github" / "workflows" / "btc_binance_ws_collector.yml").read_text(encoding="utf-8")
        start = workflow.index("          publish_cache() {")
        end = workflow.index("          publish_depth_cache() {", start)
        block = workflow[start:end]
        self.assertIn('contiguous_tail="$(python -c', block)
        self.assertNotIn('contiguous_tail="$(python - "$local_file" <<\'PY\'', block)
        self.assertIn('gappy Binance WS checkpoint', block)

    def test_collector_rest_seed_empty_cache_returns_full_health_tuple(self):
        workflow = (ROOT / ".github" / "workflows" / "btc_binance_ws_collector.yml").read_text(encoding="utf-8")
        start = workflow.index("      - name: Seed stale Binance Futures cache from closed REST klines")
        end = workflow.index("      - name: Record fixed collector base SHA", start)
        block = workflow[start:end]
        self.assertIn('if not isinstance(rows, list) or not rows:', block)
        self.assertIn('return 0, 0, None', block)
        self.assertIn('if not ordered:', block)
        self.assertEqual(block.count('return 0, 0, None'), 2)

    def test_collector_rest_seed_skips_malformed_open_time_rows(self):
        workflow = (ROOT / ".github" / "workflows" / "btc_binance_ws_collector.yml").read_text(encoding="utf-8")
        start = workflow.index("      - name: Seed stale Binance Futures cache from closed REST klines")
        end = workflow.index("      - name: Record fixed collector base SHA", start)
        block = workflow[start:end]
        self.assertIn('valid_rows = []', block)
        self.assertIn('open_ms = int(row["open_time_ms"])', block)
        self.assertIn('except (KeyError, TypeError, ValueError):', block)
        self.assertIn('ordered = [row for _, row in sorted(valid_rows, key=lambda item: item[0])]', block)
        self.assertIn('return 0, 0, None', block)

    def test_collector_blocks_old_generation_after_rest_repair(self):
        workflow = (ROOT / ".github" / "workflows" / "btc_binance_ws_collector.yml").read_text(encoding="utf-8")
        seed_start = workflow.index("      - name: Seed stale Binance Futures cache from closed REST klines")
        seed_end = workflow.index("      - name: Record fixed collector base SHA", seed_start)
        seed_block = workflow[seed_start:seed_end]
        self.assertIn('"recovery_epoch_ms": retrieved_ms', seed_block)
        start = workflow.index("          publish_cache() {")
        end = workflow.index("          publish_depth_cache() {", start)
        block = workflow[start:end]
        self.assertIn('recovery_epoch_ms', block)
        self.assertIn('remote_epoch" -gt "$local_epoch', block)
        self.assertIn('preserving repaired remote cache', block)

    def test_collector_rest_seed_tries_same_product_mirrors(self):
        workflow = (ROOT / ".github" / "workflows" / "btc_binance_ws_collector.yml").read_text(encoding="utf-8")
        start = workflow.index("      - name: Seed stale Binance Futures cache from closed REST klines")
        end = workflow.index("      - name: Record fixed collector base SHA", start)
        block = workflow[start:end]
        for host in ("fapi.binance.com", "fapi1.binance.com", "fapi2.binance.com", "fapi3.binance.com", "fapi4.binance.com"):
            self.assertIn(host, block)
        self.assertIn('Binance Futures REST seed succeeded via', block)
        self.assertIn('"symbol": "BTCUSDT"', block)
        self.assertIn('"interval": "1m"', block)

if __name__ == "__main__":
    unittest.main()
