import unittest
import numpy as np
from src.model_zoo_regime_oos import _bounded_weights,_regime_thresholds,regime_key

class ModelZooRegimeTests(unittest.TestCase):
    def test_bounded_weights_sum_and_bounds(self):
        w=_bounded_weights([0.9,1.1,1.0,1.2],[0.5,0.7,0.6,0.8],[0.05,0.1,0.08,0.12])
        self.assertAlmostEqual(float(w.sum()),1.0,places=10)
        self.assertTrue(np.all(w>=0.10-1e-12)); self.assertTrue(np.all(w<=0.55+1e-12))
    def test_regime_thresholds_are_train_derived(self):
        rows=[{"x":[0,0,(-1 if i<5 else 1)*0.002,0,0,0,0.0002+i*0.00001,0,0,0,0,0,0,0,0]} for i in range(10)]
        t=_regime_thresholds(rows)
        self.assertGreater(t["trend_q"],0.0); self.assertGreater(t["vol_q"],0.0)
        self.assertEqual(regime_key(rows[-1],t),"TREND_UP|HIGH_VOL")
    def test_range_regime(self):
        rows=[{"x":[0,0,0.000001,0,0,0,0.0002,0,0,0,0,0,0,0,0]} for _ in range(10)]
        t=_regime_thresholds(rows); self.assertEqual(regime_key(rows[0],t),"RANGE|LOW_VOL")
