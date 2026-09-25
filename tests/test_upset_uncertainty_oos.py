import unittest
import numpy as np
from src.upset_uncertainty_oos import _adjust, _features, _metrics, CLASSES


def _row(y="UP"):
    p = {
        "rf": np.array([0.2, 0.2, 0.6]),
        "extra": np.array([0.25, 0.15, 0.6]),
        "hgb": np.array([0.2, 0.3, 0.5]),
        "ensemble": np.array([0.2166666667, 0.2166666667, 0.5666666667]),
        "actual": y,
        "ts": 1,
    }
    return p


class UpsetUncertaintyTests(unittest.TestCase):
    def test_features_are_finite_and_causal_shape(self):
        rows = [_row("UP") for _ in range(20)]
        x, errors, idx = _features(rows)
        self.assertEqual(x.shape, (20, 12))
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(len(errors), 20)
        self.assertTrue((idx == np.arange(20)).all())

    def test_current_error_is_not_in_rolling_features(self):
        good = _row("UP")
        bad = _row("DOWN")
        x1, _, _ = _features([good] * 10)
        x2, _, _ = _features([good] * 9 + [bad])
        np.testing.assert_allclose(x1[-1], x2[-1], rtol=0, atol=1e-12)

    def test_adjustment_is_probability_safe(self):
        rf = np.array([[0.1,0.2,0.7],[0.7,0.2,0.1]],float)
        nr = np.array([[0.2,0.2,0.6],[0.6,0.2,0.2]],float)
        risk = np.array([0.8,0.2],float)
        p = _adjust(rf,nr,risk,"risk_shrink")
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(np.allclose(p.sum(axis=1),1.0))
