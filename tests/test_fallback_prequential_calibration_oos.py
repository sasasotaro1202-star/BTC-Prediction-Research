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
