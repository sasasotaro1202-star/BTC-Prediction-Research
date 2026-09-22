import unittest

import numpy as np

from src.calibration_frozen_replay_oos import (
    CALIBRATORS,
    _ece_mce,
    _quality,
    _post_champion_cutoff_ms,
)


class TestCalibrationFrozenReplay(unittest.TestCase):
    def test_champion_cutoff_is_resolved(self):
        cutoff = _post_champion_cutoff_ms()
        self.assertIsInstance(cutoff, int)
        self.assertGreater(cutoff, 0)

    def test_all_calibrators_are_available(self):
        self.assertEqual(
            set(CALIBRATORS),
            {"raw", "temperature", "isotonic", "vector_scaling"},
        )

    def test_calibrators_preserve_probability_simplex(self):
        y = ["DOWN", "FLAT", "UP"] * 30
        p = np.tile(np.asarray([[0.6, 0.3, 0.1],
                                [0.2, 0.6, 0.2],
                                [0.1, 0.2, 0.7]]), (30, 1))
        for ctor in CALIBRATORS.values():
            cal = ctor().fit(p, y)
            q = np.asarray(cal.transform(p), dtype=float)
            self.assertEqual(q.shape, p.shape)
            self.assertTrue(np.isfinite(q).all())
            np.testing.assert_allclose(q.sum(axis=1), 1.0, rtol=0.0, atol=1e-10)

    def test_ece_and_mce_are_bounded(self):
        y = ["DOWN", "UP"] * 50
        p = np.tile(np.asarray([[0.8, 0.1, 0.1],
                                [0.1, 0.1, 0.8]]), (50, 1))
        ece, mce = _ece_mce(y, p)
        self.assertGreaterEqual(ece, 0.0)
        self.assertGreaterEqual(mce, 0.0)
        self.assertLessEqual(ece, 1.0)
        self.assertLessEqual(mce, 1.0)

    def test_quality_contains_calibration_metrics(self):
        y = ["DOWN", "FLAT", "UP"] * 20
        p = np.tile(np.asarray([[0.6, 0.3, 0.1],
                                [0.2, 0.6, 0.2],
                                [0.1, 0.2, 0.7]]), (20, 1))
        q = _quality(y, p)
        self.assertIn("ece", q)
        self.assertIn("mce", q)
        self.assertIn("logloss", q)
        self.assertIn("brier", q)


if __name__ == "__main__":
    unittest.main()
