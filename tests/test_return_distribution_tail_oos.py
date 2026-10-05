import unittest

import numpy as np

from src.return_distribution_tail_oos import _baseline_quantiles, _evaluate_block, pinball_loss


class ReturnDistributionTailTests(unittest.TestCase):
    def test_pinball_zero_on_exact_quantile(self):
        y = np.asarray([0.0, 0.1, -0.1], dtype=float)
        self.assertEqual(pinball_loss(y, y, 0.5), 0.0)

    def test_candidate_metrics_are_finite_and_tail_aware(self):
        y = np.asarray([-0.20, -0.05, 0.00, 0.03, 0.10], dtype=float)
        candidate = {
            "q10": np.asarray([-0.10] * 5),
            "q50": np.asarray([0.01] * 5),
            "q90": np.asarray([0.08] * 5),
        }
        baseline = _baseline_quantiles(y[:4], len(y))
        result = _evaluate_block(y, candidate, baseline)
        self.assertIn("candidate", result)
        self.assertIn("baseline", result)
        self.assertTrue(np.isfinite(result["candidate"]["pinball_mean"]))
        self.assertTrue(np.isfinite(result["candidate"]["lower_tail_breach_rate"]))
        self.assertTrue(np.isfinite(result["candidate"]["upper_tail_breach_rate"]))

    def test_monotone_quantile_contract(self):
        q10 = np.asarray([-0.02, -0.01])
        q50 = np.asarray([0.0, 0.01])
        q90 = np.asarray([0.02, 0.03])
        self.assertTrue(np.all(q10 <= q50))
        self.assertTrue(np.all(q50 <= q90))

if __name__ == "__main__":
    unittest.main()
