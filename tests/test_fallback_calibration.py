import unittest
from src.fallback_calibration import _horizon_model_version


class FallbackCalibrationBindingTests(unittest.TestCase):
    def test_extracts_horizon_binding_from_combined_version(self):
        value = "5m:coinbase_fallback.rf.v1|10m:coinbase_fallback.rf.v1"
        self.assertEqual(
            _horizon_model_version(value, "coinbase", "5m"),
            "5m:coinbase_fallback.rf.v1",
        )
        self.assertEqual(
            _horizon_model_version(value, "coinbase", "10m"),
            "10m:coinbase_fallback.rf.v1",
        )

    def test_rejects_wrong_source_or_horizon(self):
        value = "5m:coinbase_fallback.rf.v1|10m:coinbase_fallback.rf.v1"
        self.assertIsNone(_horizon_model_version(value, "bybit", "5m"))
        self.assertIsNone(_horizon_model_version(value, "coinbase", "1h"))

    def test_probability_normalization_accepts_single_vector(self):
        import numpy as np
        from src.fallback_calibration import _norm
        got = _norm(np.array([0.8, 0.4, 0.4]))
        self.assertEqual(got.shape, (3,))
        self.assertTrue(np.allclose(got.sum(), 1.0))
        self.assertTrue(np.allclose(got, [0.5, 0.25, 0.25]))

    def test_probability_normalization_still_accepts_matrix(self):
        import numpy as np
        from src.fallback_calibration import _norm
        got = _norm(np.array([[0.8, 0.4, 0.4], [0.4, 0.4, 0.8]]))
        self.assertEqual(got.shape, (2, 3))
        self.assertTrue(np.allclose(got.sum(axis=1), [1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
