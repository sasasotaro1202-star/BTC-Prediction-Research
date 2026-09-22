import unittest

import numpy as np

from src.adaptive_calibration_replay_oos import _quality, _fit_temperature


class TestAdaptiveCalibrationReplay(unittest.TestCase):
    def test_quality_reports_finite_metrics(self):
        y = ["DOWN", "FLAT", "UP"] * 40
        p = np.tile(
            np.asarray([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]]),
            (40, 1),
        )
        q = _quality(y, p)
        for key in ("accuracy", "logloss", "brier", "ece"):
            self.assertTrue(np.isfinite(float(q[key])))

    def test_temperature_falls_back_safely_on_small_or_single_class_history(self):
        y_small = ["UP"] * 20
        p_small = np.tile(np.asarray([[0.2, 0.2, 0.6]]), (20, 1))
        self.assertEqual(_fit_temperature(p_small, y_small), 1.0)

    def test_temperature_is_finite_on_balanced_history(self):
        y = ["DOWN", "FLAT", "UP"] * 400
        p = np.tile(
            np.asarray([[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]]),
            (400, 1),
        )
        t = _fit_temperature(p, y)
        self.assertTrue(np.isfinite(float(t)))
        self.assertGreaterEqual(t, 0.5)
        self.assertLessEqual(t, 3.0)


if __name__ == "__main__":
    unittest.main()
