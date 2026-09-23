import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import binance_ws


class TestBinanceWebSocket(unittest.TestCase):
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
