import unittest
import numpy as np

from src.time_regime_oos import _timestamp_features, _augment


class TimeRegimeOOSTests(unittest.TestCase):
    def test_timestamp_features_are_bounded_and_periodic(self):
        a = _timestamp_features("2026-09-22T00:00:00+00:00")
        b = _timestamp_features("2026-09-23T00:00:00+00:00")
        self.assertEqual(len(a), 5)
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue(np.all(np.abs(np.asarray(a[:4])) <= 1.0))
        self.assertEqual(a[4], 0.0)
        self.assertEqual(b[4], 0.0)

    def test_weekend_flag_is_explicit(self):
        saturday = _timestamp_features("2026-09-19T12:00:00+00:00")
        monday = _timestamp_features("2026-09-21T12:00:00+00:00")
        self.assertEqual(saturday[4], 1.0)
        self.assertEqual(monday[4], 0.0)

    def test_augment_preserves_identity_and_adds_exactly_five_features(self):
        row = {
            "id": 1,
            "created": "2026-09-22T12:30:00+00:00",
            "x": [0.0] * 15,
            "y": "UP",
            "production": [0.2, 0.3, 0.5],
        }
        out = _augment([row])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["id"], 1)
        self.assertEqual(len(out[0]["x"]), 20)
        self.assertTrue(np.isfinite(np.asarray(out[0]["x"], dtype=float)).all())


if __name__ == "__main__":
    unittest.main()
