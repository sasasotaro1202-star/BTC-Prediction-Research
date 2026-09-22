from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
import unittest

from src import binance_history as bh


def _zip_bytes(rows):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        text = "\n".join(",".join(map(str, row)) for row in rows) + "\n"
        z.writestr("BTCUSDT-1m-test.csv", text)
    return buf.getvalue()


class TestBinanceHistory(unittest.TestCase):
    def test_rows_from_zip_ignores_headers_and_malformed_rows(self):
        raw = _zip_bytes([
            ["open_time", "open", "high", "low", "close", "volume"],
            [1000, 10, 11, 9, 10.5, 2],
            ["bad", "x", "x", "x", "x", "x"],
            [2000, 20, 21, 19, 20.5, 3],
        ])
        self.assertEqual(
            bh._rows_from_zip(raw, "unit"),
            [
                [1000, 10.0, 11.0, 9.0, 10.5, 2.0],
                [2000, 20.0, 21.0, 19.0, 20.5, 3.0],
            ],
        )

    def test_load_day_uses_cache_and_rejects_future_candles(self):
        old_cache = bh.CACHE_DIR
        old_time = bh.time.time
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                bh.CACHE_DIR = Path(tmp)
                bh.time.time = lambda: 2_000_000 / 1000
                path = bh._cached_day(datetime(1970, 1, 1))
                path.write_text(json.dumps([
                    [1_900_000, 1, 1, 1, 1, 1],
                    [1_950_000, 1, 1, 1, 1, 1],
                ]), encoding="utf-8")
                rows, source = bh._load_day(datetime(1970, 1, 1))
                self.assertEqual(source, "cache")
                self.assertEqual([r[0] for r in rows], [1_900_000])
        finally:
            bh.CACHE_DIR = old_cache
            bh.time.time = old_time


    def test_archive_rows_walks_back_across_multiple_months(self):
        old_month = bh._month_rows
        old_day = bh._load_day
        try:
            def month_rows(month):
                if month.month == 9:
                    raise RuntimeError("current month not published")
                if month.month == 8:
                    return [[i * 60_000, 1, 1, 1, 1, 1] for i in range(5)]
                return []
            bh._month_rows = month_rows
            bh._load_day = lambda day: ([], "mock")
            out = bh.binance_archive_rows(target=3)
            self.assertEqual(len(out), 3)
            self.assertEqual([r[0] for r in out], [120_000, 180_000, 240_000])
        finally:
            bh._month_rows = old_month
            bh._load_day = old_day

    def test_archive_rows_deduplicates_sorts_and_returns_recent_target(self):
        old_month = bh._month_rows
        try:
            bh._month_rows = lambda month: [
                [120_000, 3, 3, 3, 3, 3],
                [0, 1, 1, 1, 1, 1],
                [60_000, 2, 2, 2, 2, 2],
                [60_000, 22, 22, 22, 22, 22],
            ]
            rows = bh.binance_archive_rows(target=3)
            self.assertEqual([r[0] for r in rows], [0, 60_000, 120_000])
            self.assertEqual(rows[-1][1], 3.0)
        finally:
            bh._month_rows = old_month

    def test_archive_rows_fails_closed_when_not_enough_history(self):
        old_month = bh._month_rows
        old_day = bh._load_day
        try:
            bh._month_rows = lambda month: []
            bh._load_day = lambda day: ([], "mock")
            with self.assertRaisesRegex(RuntimeError, "need 5"):
                bh.binance_archive_rows(target=5)
        finally:
            bh._month_rows = old_month
            bh._load_day = old_day

    def test_contiguous_suffix_drops_older_rows_before_gap(self):
        rows = [
            [0, 1, 1, 1, 1, 1],
            [60_000, 1, 1, 1, 1, 1],
            [120_000, 1, 1, 1, 1, 1],
            [240_000, 1, 1, 1, 1, 1],
            [300_000, 1, 1, 1, 1, 1],
        ]
        out = bh._contiguous_suffix(rows)
        self.assertEqual([r[0] for r in out], [240_000, 300_000])

    def test_contiguous_suffix_is_deduplicated_and_sorted(self):
        rows = [
            [120_000, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [60_000, 1, 1, 1, 1, 1],
            [60_000, 2, 2, 2, 2, 2],
        ]
        out = bh._contiguous_suffix(rows)
        self.assertEqual([r[0] for r in out], [0, 60_000, 120_000])


if __name__ == "__main__":
    unittest.main()
