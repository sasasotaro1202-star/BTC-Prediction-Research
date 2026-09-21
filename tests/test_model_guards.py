import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from model_compare import CLASSES, EMBARGO_BARS, PURGE_BARS, metrics, normalize, prediction_precedes_target  # noqa: E402


class TestModelGuards(unittest.TestCase):
    def test_research_class_order_matches_db_storage_conversion(self):
        self.assertEqual(CLASSES, ['DOWN', 'FLAT', 'UP'])
        stored_up_down_flat = [0.70, 0.10, 0.20]
        research_down_flat_up = [stored_up_down_flat[1], stored_up_down_flat[2], stored_up_down_flat[0]]
        self.assertEqual(research_down_flat_up, [0.10, 0.20, 0.70])
        self.assertEqual(max(range(3), key=lambda i: research_down_flat_up[i]), CLASSES.index('UP'))

    def test_probability_normalization_is_finite_and_sums_to_one(self):
        p = normalize([[0.7, 0.2, 0.1], [10.0, 0.0, 0.0]])
        self.assertEqual(p.shape, (2, 3))
        self.assertTrue(all(math.isfinite(float(x)) for x in p.ravel()))
        self.assertTrue(all(abs(float(row.sum()) - 1.0) < 1e-9 for row in p))

    def test_metrics_respect_research_class_order(self):
        ys = ['UP', 'DOWN', 'FLAT']
        probs = [[0.05, 0.05, 0.90], [0.90, 0.05, 0.05], [0.05, 0.90, 0.05]]
        m = metrics(ys, probs)
        self.assertEqual(m['accuracy'], 1.0)
        self.assertLess(m['logloss'], 0.2)
        self.assertLess(m['brier'], 0.1)

    def test_horizon_purge_and_embargo_are_conservative(self):
        self.assertEqual(PURGE_BARS['5m'], 5)
        self.assertEqual(PURGE_BARS['10m'], 10)
        self.assertGreaterEqual(EMBARGO_BARS['5m'], 60)
        self.assertGreaterEqual(EMBARGO_BARS['10m'], 60)

    def test_prediction_must_precede_target_strictly(self):
        self.assertTrue(prediction_precedes_target(
            '2026-09-21T18:55:58+00:00',
            '2026-09-21T19:00:00+00:00',
        ))
        self.assertFalse(prediction_precedes_target(
            '2026-09-21T19:00:00+00:00',
            '2026-09-21T19:00:00+00:00',
        ))
        self.assertFalse(prediction_precedes_target(
            '2026-09-21T19:00:01+00:00',
            '2026-09-21T19:00:00+00:00',
        ))
        self.assertFalse(prediction_precedes_target('bad', '2026-09-21T19:00:00+00:00'))


if __name__ == '__main__':
    unittest.main()
