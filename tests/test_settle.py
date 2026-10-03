import unittest
from unittest.mock import patch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import settle


class TestSettle(unittest.TestCase):
    def test_direction_threshold(self):
        self.assertEqual(settle.direction(100.0, 100.03), 'UP')
        self.assertEqual(settle.direction(100.0, 99.97), 'DOWN')
        self.assertEqual(settle.direction(100.0, 100.01), 'FLAT')

    def test_resolve_targets_deduplicates_and_tolerates_failure(self):
        calls = []

        def fake_target(target, source):
            calls.append((target, source))
            if target == 'bad':
                raise RuntimeError('temporary public-source failure')
            return (123.0, 'test')

        with patch.object(settle, 'target_close_preferred', side_effect=fake_target):
            result = settle.resolve_targets([('a', 'binance_futures'), ('a', 'binance_futures'), ('bad', 'binance_futures')], max_workers=2)

        self.assertEqual(sorted(calls), [('a', 'binance_futures'), ('bad', 'binance_futures')])
        self.assertEqual(result[('a', 'binance_futures')], (123.0, 'test'))
        self.assertEqual(result[('bad', 'binance_futures')], (None, 'unavailable'))

    def test_extended_settlement_validates_finite_probabilities(self):
        import sqlite3
        import tempfile
        from pathlib import Path

        horizons = ("15m", "30m", "1h", "3h", "6h", "12h", "24h")
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "predictions.db"
            cols = [
                "prediction_id INTEGER PRIMARY KEY",
                "created_at_utc TEXT",
                "target_5m TEXT", "target_10m TEXT", "base_price REAL",
                "p_up_5m REAL", "p_down_5m REAL", "p_flat_5m REAL",
                "p_up_10m REAL", "p_down_10m REAL", "p_flat_10m REAL",
                "actual_price_5m REAL", "actual_price_10m REAL",
                "model_version TEXT", "scenario_json TEXT",
                "settlement_source_5m TEXT", "settlement_source_10m TEXT",
            ]
            for h in horizons:
                cols.extend([
                    f"target_{h} TEXT",
                    f"p_up_{h} REAL", f"p_down_{h} REAL", f"p_flat_{h} REAL",
                    f"actual_price_{h} REAL", f"actual_direction_{h} TEXT",
                    f"correct_{h} INTEGER", f"settled_{h}_at_utc TEXT",
                    f"settlement_source_{h} TEXT",
                ])
            with sqlite3.connect(db_path) as con:
                con.execute("CREATE TABLE predictions (" + ",".join(cols) + ")")
                values = [
                    1, "2000-01-01T00:00:00+00:00",
                    "2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", 100.0,
                    .2, .4, .4, .2, .4, .4, None, None,
                    "test_model", '{"production_mode":"binance_primary"}', None, None,
                ]
                for h in horizons:
                    values.extend([
                        "2000-01-01T00:00:00+00:00", .6, .2, .2,
                        None, None, None, None, None,
                    ])
                placeholders = ",".join("?" for _ in values)
                con.execute("INSERT INTO predictions VALUES (" + placeholders + ")", values)

            def fake_resolve(targets, max_workers=4):
                return {key: (110.0, "binance_futures") for key in set(targets)}

            with patch.object(settle, "DB", db_path),                  patch.object(settle, "init_db", lambda: None),                  patch.object(settle, "resolve_targets", side_effect=fake_resolve):
                settle.settle()

            with sqlite3.connect(db_path) as con:
                row = con.execute(
                    "SELECT actual_price_15m,actual_direction_15m,correct_15m,settlement_source_15m "
                    "FROM predictions WHERE prediction_id=1"
                ).fetchone()
            self.assertEqual(row, (110.0, "UP", 1, "binance_futures"))


if __name__ == '__main__':
    unittest.main()


    def test_scenario_source_uses_source_native_fallback(self):
        coinbase = {"production_mode":"coinbase_fallback","data_quality":{"price_feature_fallback":"coinbase"}}
        bybit = {"production_mode":"bybit_fallback","data_quality":{"price_feature_fallback":"bybit"}}
        self.assertEqual(settle.scenario_source(__import__("json").dumps(coinbase)), "coinbase_exchange")
        self.assertEqual(settle.scenario_source(__import__("json").dumps(bybit)), "bybit_linear")

    def test_known_fallback_never_silently_becomes_binance(self):
        row = {"production_mode":"coinbase_fallback","data_quality":{"price_feature_fallback":"coinbase"}}
        self.assertNotEqual(settle.scenario_source(__import__("json").dumps(row)), "binance_futures")
