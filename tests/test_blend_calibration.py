import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import blend_calibration  # noqa: E402


class TestBlendCalibration(unittest.TestCase):
    def test_normalization(self):
        p = blend_calibration._norm([[2.0, 1.0, 1.0]])
        self.assertAlmostEqual(float(p.sum()), 1.0, places=10)

    def test_brier_prefers_correct_probability(self):
        # blend_calibration.CLASSES is [UP, DOWN, FLAT], so the rows below
        # must follow that exact probability-column order.
        y = ['UP', 'DOWN', 'FLAT']
        good = [[0.98, 0.01, 0.01], [0.01, 0.98, 0.01], [0.01, 0.01, 0.98]]
        bad = [[0.34, 0.33, 0.33]] * 3
        self.assertLess(blend_calibration._brier(y, good), blend_calibration._brier(y, bad))

    def test_grid_is_bounded(self):
        self.assertGreaterEqual(float(blend_calibration.GRID.min()), 0.0)
        self.assertLessEqual(float(blend_calibration.GRID.max()), 0.45)


if __name__ == '__main__':
    unittest.main()
