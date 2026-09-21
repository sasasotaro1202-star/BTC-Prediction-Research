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


    def test_four_model_weights_are_normalized_and_bounded(self):
        w = _weights_from_losses([0.4, 0.6, 0.8, 1.0])
        self.assertEqual(len(w), 4)
        self.assertAlmostEqual(sum(w), 1.0, places=10)
        self.assertTrue(all(0.10 <= x <= 0.55 for x in w))

    def test_diversity_penalty_reduces_redundant_model_weight(self):
        baseline = _weights_from_losses([0.5, 0.5, 0.5, 0.5])
        diversified = _weights_from_losses(
            [0.5, 0.5, 0.5, 0.5],
            redundancy=[0.0, 1.0, 0.0, 0.0],
        )
        self.assertLess(diversified[1], baseline[1])
        self.assertAlmostEqual(sum(diversified), 1.0, places=10)
        self.assertTrue(all(w >= 0.10 for w in diversified))

    def test_invalid_redundancy_fails_closed_to_uniform(self):
        w = _weights_from_losses([0.4, 0.5, 0.6, 0.7], redundancy=[0.1])
        self.assertTrue(all(abs(x - 0.25) < 1e-9 for x in w))
