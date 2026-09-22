import unittest
from unittest.mock import patch

from src import model_compare


class ArchiveResearchFallbackTests(unittest.TestCase):
    def test_bybit_fallback_builds_research_rows_when_binance_archive_fails(self):
        raw = [
            [i * 60_000, 100.0, 101.0, 99.0, 100.0 + i * 0.01, 10.0]
            for i in range(100)
        ]
        with patch("src.binance_history.binance_archive_rows", side_effect=RuntimeError("archive unavailable")), \
             patch("binance_history.binance_archive_rows", side_effect=RuntimeError("archive unavailable")):
            with patch("src.bootstrap_train.fetch_bybit", return_value=raw), \
                 patch("bootstrap_train.fetch_bybit", return_value=raw):
                with patch("src.coinbase_fallback_train.fetch_coinbase", side_effect=RuntimeError("coinbase unavailable")), \
                     patch("coinbase_fallback_train.fetch_coinbase", side_effect=RuntimeError("coinbase unavailable")):
                    rows = model_compare.load_archive_research_rows("5m", max_rows=50)
        self.assertTrue(rows)
        self.assertEqual(len(rows), 50)
        self.assertTrue(all(r["data_source"] == "bybit" for r in rows))
        self.assertTrue(all(r["production_mode"] == "bybit_archive" for r in rows))
        self.assertTrue(all(r["target"].endswith("+00:00") for r in rows))
        self.assertTrue(all(
            rows[i]["target"] < rows[i + 1]["target"]
            for i in range(len(rows) - 1)
        ))
        self.assertTrue(all(
            rows[i]["data_source"] == "bybit"
            and rows[i]["production_mode"] == "bybit_archive"
            for i in range(len(rows))
        ))

    def test_source_row_identity_prevents_cross_venue_collision(self):
        raw = [
            [i * 60_000, 100.0, 101.0, 99.0, 100.0 + i * 0.01, 10.0]
            for i in range(40)
        ]
        a = model_compare._research_archive_rows_from_raw(raw, "5m", 20, "bybit")
        b = model_compare._research_archive_rows_from_raw(raw, "5m", 20, "coinbase")
        self.assertEqual(len(a), len(b))
        self.assertNotEqual(a[-1]["id"], b[-1]["id"])
        self.assertNotEqual(a[-1]["data_source"], b[-1]["data_source"])


if __name__ == "__main__":
    unittest.main()
