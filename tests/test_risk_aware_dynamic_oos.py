import unittest

from src.risk_aware_dynamic_oos import _weights_from_losses


class RiskAwareWeightTests(unittest.TestCase):
    def test_weights_are_normalized_and_bounded(self):
        w = _weights_from_losses([0.4, 0.8, 1.2])
        self.assertAlmostEqual(sum(w), 1.0, places=10)
        self.assertTrue(all(0.10 <= x <= 0.55 for x in w))

    def test_equal_losses_stay_near_uniform(self):
        w = _weights_from_losses([1.0, 1.0, 1.0])
        self.assertTrue(all(abs(x - 1.0 / 3.0) < 1e-9 for x in w))

    def test_invalid_losses_fail_closed(self):
        w = _weights_from_losses([float("nan"), 1.0, 1.0])
        self.assertTrue(all(abs(x - 1.0 / 3.0) < 1e-9 for x in w))
