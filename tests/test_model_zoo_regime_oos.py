import unittest
import numpy as np
from src.model_zoo_regime_oos import _bounded_weights,_regime_thresholds,regime_key

class ModelZooRegimeTests(unittest.TestCase):
    def test_bounded_weights_sum_and_bounds(self):
        w=_bounded_weights([0.9,1.1,1.0,1.2],[0.5,0.7,0.6,0.8],[0.05,0.1,0.08,0.12])
        self.assertAlmostEqual(float(w.sum()),1.0,places=10)
        self.assertTrue(np.all(w>=0.10-1e-12)); self.assertTrue(np.all(w<=0.55+1e-12))
    def test_regime_thresholds_are_train_derived(self):
        rows=[{"x":[0,0,(-1 if i<5 else 1)*(0.002 if i<9 else 0.003),0,0,0,0.0002+i*0.00001,0,0,0,0,0,0,0,0]} for i in range(10)]
        t=_regime_thresholds(rows)
        self.assertGreater(t["trend_q"],0.0); self.assertGreater(t["vol_q"],0.0)
        self.assertEqual(regime_key(rows[-1],t),"TREND_UP|HIGH_VOL")
    def test_range_regime(self):
        rows=[{"x":[0,0,0.000001,0,0,0,0.00005,0,0,0,0,0,0,0,0]} for _ in range(10)]
        t=_regime_thresholds(rows); self.assertEqual(regime_key(rows[0],t),"RANGE|LOW_VOL")
    def test_uncertainty_guard_uses_prior_history_only(self):
        from src.model_zoo_regime_oos import _uncertainty_guard
        stable=np.asarray([[0.34,0.32,0.34],[0.90,0.05,0.05]],dtype=float)
        routed=np.asarray([[0.80,0.10,0.10],[0.34,0.32,0.34]],dtype=float)
        hist=[0.20]*400+[0.95]*10
        out,threshold,fraction=_uncertainty_guard(routed,stable,hist,quantile=0.80,min_history=300)
        self.assertIsNotNone(threshold)
        self.assertGreaterEqual(fraction,0.0)
        self.assertTrue(np.allclose(out[0],routed[0]))
