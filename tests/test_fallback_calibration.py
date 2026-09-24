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


if __name__ == "__main__":
    unittest.main()
