import unittest

from src.fallback_calibration import _horizon_model_version


class FallbackPrequentialCalibrationTests(unittest.TestCase):
    def test_exact_horizon_binding_is_extracted(self):
        value = "5m:coinbase_fallback.rf.v1|10m:coinbase_fallback.rf.v1"
        self.assertEqual(
            _horizon_model_version(value, "coinbase", "5m"),
            "5m:coinbase_fallback.rf.v1",
        )
        self.assertEqual(
            _horizon_model_version(value, "coinbase", "10m"),
            "10m:coinbase_fallback.rf.v1",
        )

    def test_wrong_source_is_not_accepted(self):
        value = "5m:coinbase_fallback.rf.v1|10m:coinbase_fallback.rf.v1"
        self.assertIsNone(_horizon_model_version(value, "bybit", "5m"))


if __name__ == "__main__":
    unittest.main()

    def test_source_probability_order_is_converted_to_canonical(self):
        import numpy as np
        from src.fallback_prequential_calibration_oos import _to_canonical_probability_matrix
        got = _to_canonical_probability_matrix([[0.70, 0.20, 0.10]])
        self.assertTrue(np.allclose(got, [[0.20, 0.10, 0.70]]))

    def test_loaded_probability_vectors_are_normalized(self):
        import numpy as np
        from src.fallback_prequential_calibration_oos import _norm_vector
        got = _norm_vector([7.0, 2.0, 1.0])
        self.assertTrue(np.allclose(got, [0.7, 0.2, 0.1]))
