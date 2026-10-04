import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.label_policy import BINARY_CLASSES, BINARY_TARGET_VERSION, binary_direction_from_prices
from src.binary_target_oos import _apply_temperature, _temperature_fit, metrics


class TestBinaryTarget(unittest.TestCase):
    def test_binary_target_has_only_up_down(self):
        self.assertEqual(BINARY_CLASSES, ("DOWN", "UP"))
        self.assertEqual(BINARY_TARGET_VERSION, "binary_sign_v1")

    def test_zero_return_is_down(self):
        self.assertEqual(binary_direction_from_prices(100.0, 100.0), "DOWN")

    def test_positive_return_is_up(self):
        self.assertEqual(binary_direction_from_prices(100.0, 100.01), "UP")

    def test_negative_return_is_down(self):
        self.assertEqual(binary_direction_from_prices(100.0, 99.99), "DOWN")

    def test_probability_metrics_are_binary(self):
        y = np.asarray([0, 0, 1, 1])
        p = np.asarray([[0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.1, 0.9]])
        out = metrics(y, p)
        self.assertEqual(out["n"], 4)
        self.assertGreaterEqual(out["logloss"], 0.0)
        self.assertGreaterEqual(out["brier"], 0.0)

    def test_temperature_is_fail_closed_for_short_history(self):
        self.assertEqual(_temperature_fit([0, 1], [[0.7, 0.3], [0.3, 0.7]]), 1.0)

    def test_temperature_preserves_probability_contract(self):
        p = np.asarray([[0.7, 0.3], [0.4, 0.6]])
        q = _apply_temperature(p, 2.0)
        self.assertEqual(q.shape, (2, 2))
        self.assertTrue(np.allclose(q.sum(axis=1), 1.0))
        self.assertTrue(np.isfinite(q).all())


if __name__ == "__main__":
    unittest.main()
