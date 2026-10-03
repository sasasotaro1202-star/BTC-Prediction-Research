import unittest
import numpy as np
from tempfile import TemporaryDirectory
from src import historical_research as hr


from pathlib import Path


class HistoricalTargetTests(unittest.TestCase):
    def test_label_uses_exact_elapsed_minutes(self):
        rows = [
            (0, [1, 2], 100.0),
            (60_000, [1, 2], 101.0),
            (120_000, [1, 2], 102.0),
        ]
        _, y, ids, _ = hr.labels(rows, 2)
        self.assertEqual(ids.tolist(), [0])
        self.assertEqual(y.tolist(), ["UP"])

    def test_gap_does_not_stretch_target(self):
        rows = [
            (0, [1, 2], 100.0),
            (60_000, [1, 2], 101.0),
            (180_000, [1, 2], 103.0),
            (240_000, [1, 2], 104.0),
        ]
        _, y, ids, _ = hr.labels(rows, 2)
        self.assertEqual(ids.tolist(), [60_000])
        self.assertEqual(y.tolist(), ["UP"])

    def test_feature_window_must_be_contiguous(self):
        self.assertTrue(hr._window_is_contiguous([0, 60_000, 120_000]))
        self.assertFalse(hr._window_is_contiguous([0, 60_000, 180_000]))

    def test_feature_frontier_schema_is_distinct_from_production_schema(self):
        self.assertEqual(len(hr.BASE_FEATURES), 41)
        self.assertEqual(len(hr.FRONTIER_FEATURES), 51)
        self.assertEqual(len(hr.FEATURES), 92)
        self.assertEqual(len(set(hr.FEATURES)), len(hr.FEATURES))
        self.assertEqual(hr.FEATURES[:41], hr.BASE_FEATURES)
        self.assertTrue({"rsi14","bb_z60","ema_slope30","oi_x_return5","ret120","rv120","vol_of_vol20","trend_efficiency60","garman_klass_vol20"}.issubset(set(hr.FRONTIER_FEATURES)))

    def test_feature_frontier_helpers_are_finite_and_shape_safe(self):
        prices=np.linspace(100.0,110.0,121)
        returns=np.diff(prices)/prices[:-1]
        self.assertTrue(np.isfinite(hr._rsi_from_returns(returns,14)))
        self.assertTrue(np.isfinite(hr._zscore_current(prices,20)))
        self.assertTrue(np.isfinite(hr._moment_feature(returns[-20:],"skew")))
        self.assertTrue(np.isfinite(hr._moment_feature(returns[-20:],"kurt")))
        self.assertTrue(np.isfinite(hr._autocorr(returns,5)))


    def test_missing_historical_sources_fail_closed_instead_of_zero_imputation(self):
        source = Path("src/historical_research.py").read_text(encoding="utf-8")
        self.assertNotIn('np.zeros_like(b)', source)
        self.assertNotIn('funding.get(ft,0.0)', source)
        self.assertIn('if not exists(maps["btc_mark"],w) or not exists(maps["btc_premium"],w):', source)
        self.assertIn('raise RuntimeError("both BTC spot and futures historical data are unavailable")', source)
        self.assertNotIn('using BTC futures as explicit spot proxy', source)
        self.assertIn('if ft is None or prev_f is None or ot is None or prev_oi is None:', source)

    def test_spot_proxy_state_is_explicitly_disabled(self):
        source = Path("src/historical_research.py").read_text(encoding="utf-8")
        launcher = Path("scripts/historical_research_spot_fallback.py").read_text(encoding="utf-8")
        self.assertIn("spot_proxy=False", source)
        self.assertNotIn("if spot_proxy: basis=bd=0.0", source)
        self.assertIn("hr.spot_proxy=False", launcher)
        self.assertNotIn("hr.spot_proxy=not any(", launcher)




    def test_empty_or_corrupt_historical_cache_is_invalidated(self):
        with TemporaryDirectory() as tmp:
            empty=Path(tmp)/"empty.json"
            empty.write_text("[]",encoding="utf-8")
            self.assertIsNone(hr._load_nonempty_cached_rows(empty))
            self.assertFalse(empty.exists())

            corrupt=Path(tmp)/"corrupt.json"
            corrupt.write_text("{not-json",encoding="utf-8")
            self.assertIsNone(hr._load_nonempty_cached_rows(corrupt))
            self.assertFalse(corrupt.exists())

            valid=Path(tmp)/"valid.json"
            valid.write_text('[["x"]]',encoding="utf-8")
            self.assertEqual(hr._load_nonempty_cached_rows(valid), [["x"]])
            self.assertTrue(valid.exists())

    def test_invalid_base_price_is_skipped(self):
        rows = [
            (0, [1, 2], 0.0),
            (120_000, [1, 2], 101.0),
        ]
        _, y, ids, bases = hr.labels(rows, 2)
        self.assertEqual(ids.tolist(), [])
        self.assertEqual(y.tolist(), [])
        self.assertEqual(bases.tolist(), [])


if __name__ == "__main__":
    unittest.main()
