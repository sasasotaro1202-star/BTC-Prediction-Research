import unittest
import numpy as np
from src.recency_weighted_oos import decay_weights

class RecencyWeightTests(unittest.TestCase):
    def test_weight_normalization_and_monotonicity(self):
        w=decay_weights(100,20)
        self.assertAlmostEqual(float(w.mean()),1.0,places=10)
        self.assertGreater(float(w[-1]),float(w[0]))
        self.assertTrue(np.all(w>0))
    def test_half_life_changes_decay_strength(self):
        short=decay_weights(100,10)
        long=decay_weights(100,80)
        self.assertGreater(float(short[-1]/short[0]),float(long[-1]/long[0]))
