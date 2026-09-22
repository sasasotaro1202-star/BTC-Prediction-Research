import unittest

import numpy as np

from src.multiscale_frozen_replay_oos import (
    BASE_FEATURES,
    MULTISCALE_FEATURES,
    TIME_FEATURES,
    VARIANTS,
    _slice_partition,
    _timestamp_features,
    _vector,
)


def synthetic_rows(n=140):
    rows = []
    price = 100.0
    for i in range(n):
        price *= 1.0 + 0.0001 * np.sin(i / 7.0)
        ts = 1_725_000_000_000 + i * 60_000
        o = price * (1.0 - 0.0002)
        h = price * (1.0 + 0.0005)
        l = price * (1.0 - 0.0005)
        v = 1000.0 + 40.0 * np.cos(i / 5.0)
        rows.append([ts, o, h, l, price, v])
    return rows


class MultiScaleFrozenReplayTests(unittest.TestCase):
    def test_feature_vector_is_causal_and_finite(self):
        raw = synthetic_rows()
        x_a = np.asarray(_vector(raw, 100), dtype=float)
        x_b = np.asarray(_vector(raw, 100), dtype=float)
        self.assertEqual(x_a.shape[0], len(BASE_FEATURES) + len(MULTISCALE_FEATURES) + len(TIME_FEATURES))
        self.assertTrue(np.isfinite(x_a).all())
        np.testing.assert_allclose(x_a, x_b)

        # Future rows must not affect the feature vector at the earlier cutoff.
        raw_future_changed = synthetic_rows()
        raw_future_changed[120][4] *= 3.0
        raw_future_changed[120][2] *= 3.0
        raw_future_changed[120][3] *= 3.0
        np.testing.assert_allclose(
            x_a,
            np.asarray(_vector(raw_future_changed, 100), dtype=float),
        )

    def test_timestamp_features_have_expected_dimension_and_bounds(self):
        values = np.asarray(_timestamp_features(1_758_524_800_000), dtype=float)
        self.assertEqual(values.shape, (5,))
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(np.all(np.abs(values[:4]) <= 1.0))
        self.assertIn(values[4], (0.0, 1.0))

    def test_variant_feature_counts_are_consistent(self):
        self.assertEqual(len(VARIANTS["base"]), 15)
        self.assertEqual(len(VARIANTS["multiscale"]), 28)
        self.assertEqual(len(VARIANTS["time"]), 20)
        self.assertEqual(len(VARIANTS["full"]), 33)

    def test_partition_has_horizon_purge_between_slices(self):
        rows = list(range(10_000))
        train5, selection5, gate5, replay5 = _slice_partition(rows, "5m")
        train10, selection10, gate10, replay10 = _slice_partition(rows, "10m")

        self.assertEqual(train5[-1] + 5, 6000)
        self.assertEqual(selection5[-1] + 5, 7000)
        self.assertEqual(gate5[-1] + 5, 8000)

        self.assertEqual(train10[-1] + 10, 6000)
        self.assertEqual(selection10[-1] + 10, 7000)
        self.assertEqual(gate10[-1] + 10, 8000)
        self.assertEqual(replay5[0], 8000)
        self.assertEqual(replay10[0], 8000)


if __name__ == "__main__":
    unittest.main()
