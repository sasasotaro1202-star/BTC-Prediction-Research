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
        y = ['UP', 'DOWN', 'FLAT']
        good = [[0.98, 0.01, 0.01], [0.01, 0.98, 0.01], [0.01, 0.01, 0.98]]
        bad = [[0.34, 0.33, 0.33]] * 3
        self.assertLess(blend_calibration._brier(y, good), blend_calibration._brier(y, bad))

    def test_horizon_column_mapping(self):
        self.assertEqual(blend_calibration._actual_column('5m'), 'actual_direction_5m')
        self.assertEqual(blend_calibration._actual_column('10m'), 'actual_direction_10m')
        with self.assertRaises(ValueError):
            blend_calibration._actual_column('5')

    def test_probability_suffix_does_not_double_append_m(self):
        self.assertEqual('p_up_5m', f'p_up_{"5m"}')
        self.assertEqual('p_up_10m', f'p_up_{"10m"}')
        self.assertNotEqual('p_up_5mm', f'p_up_{"5m"}')
        self.assertNotEqual('p_up_10mm', f'p_up_{"10m"}')

    def test_grid_is_bounded(self):
        self.assertGreaterEqual(float(blend_calibration.GRID.min()), 0.0)
        self.assertLessEqual(float(blend_calibration.GRID.max()), 0.45)


if __name__ == '__main__':
    unittest.main()
