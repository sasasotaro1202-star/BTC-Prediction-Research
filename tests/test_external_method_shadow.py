import math
import unittest

from src.external_method_shadow import direction_probs, ece, metrics


class ExternalMethodShadowTests(unittest.TestCase):
    def test_direction_probs(self):
        result = direction_probs([101.0, 100.0, 100.01, 99.0], 100.0)
        self.assertAlmostEqual(sum(result.values()), 1.0)
        self.assertEqual(result["UP"], 0.25)
        self.assertEqual(result["DOWN"], 0.25)
        self.assertEqual(result["FLAT"], 0.5)

    def test_ece_empty(self):
        self.assertTrue(math.isnan(ece([], [])))

    def test_metrics_empty(self):
        result = metrics([], "5m")
        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["logloss"])
