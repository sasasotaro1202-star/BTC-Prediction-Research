import unittest

import numpy as np

from src.bootstrap_train import make_features
from src.predict import features


def synthetic_rows(n=80):
    rows = []
    price = 100.0
    for i in range(n):
        drift = 0.00015 * np.sin(i / 11.0) + 0.00005 * np.cos(i / 5.0)
        price *= 1.0 + drift
        open_price = price * (1.0 - 0.0003 * np.cos(i / 7.0))
        high = max(price, open_price) * (1.0 + 0.0004)
        low = min(price, open_price) * (1.0 - 0.0004)
        volume = 1000.0 + 80.0 * np.sin(i / 9.0)
        rows.append([
            1_750_000_000_000 + i * 60_000,
            open_price,
            high,
            low,
            price,
            max(1.0, volume),
        ])
    return rows


class TestCausalFeatureRegression(unittest.TestCase):
    def test_bootstrap_features_ignore_future_rows(self):
        raw = synthetic_rows()
        cutoff = 60
        baseline = np.asarray(make_features(raw[:cutoff + 1]), dtype=float)

        future_changed = [row[:] for row in raw]
        future_changed[cutoff + 5][4] *= 2.5
        future_changed[cutoff + 5][2] *= 2.5
        future_changed[cutoff + 5][3] *= 0.4
        future_changed[cutoff + 5][5] *= 10.0

        observed = np.asarray(make_features(future_changed[:cutoff + 1]), dtype=float)
        np.testing.assert_allclose(observed, baseline, rtol=0.0, atol=0.0)

    def test_runtime_features_ignore_future_rows(self):
        raw = synthetic_rows()
        cutoff = 60
        baseline = features(raw[:cutoff + 1])

        future_changed = [row[:] for row in raw]
        future_changed[cutoff + 3][4] *= 3.0
        future_changed[cutoff + 3][2] *= 3.0
        future_changed[cutoff + 3][3] *= 0.3
        future_changed[cutoff + 3][5] *= 20.0

        observed = features(future_changed[:cutoff + 1])
        self.assertEqual(tuple(baseline.keys()), tuple(observed.keys()))
        for key in baseline:
            self.assertEqual(observed[key], baseline[key])

    def test_current_row_does_affect_features(self):
        raw = synthetic_rows()
        cutoff = 60

        baseline = features(raw[:cutoff + 1])
        current_changed = [row[:] for row in raw]
        current_changed[cutoff][4] *= 1.001
        current_changed[cutoff][2] *= 1.001
        changed = features(current_changed[:cutoff + 1])

        self.assertTrue(
            any(changed[key] != baseline[key] for key in baseline),
            "causal feature path must still react to the current observation",
        )


if __name__ == "__main__":
    unittest.main()
