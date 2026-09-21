import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import extended_features_oos  # noqa: E402


class ExtendedFeaturesTests(unittest.TestCase):
    def test_extended_schema_is_append_only(self):
        self.assertEqual(
            list(extended_features_oos.ALL_FEATURES[: len(extended_features_oos.FEATURES)]),
            list(extended_features_oos.FEATURES),
        )
        self.assertEqual(
            set(extended_features_oos.EXTRA_FEATURES),
            {"ret_15m", "ret_30m", "range_position_30m", "trend_alignment"},
        )

    def test_candidate_factories_are_research_only(self):
        self.assertGreaterEqual(len(extended_features_oos.factories()), 3)
        self.assertTrue(all(name.endswith("_extended") for name in extended_features_oos.factories()))

    def test_insufficient_data_fails_closed(self):
        from unittest.mock import patch
        with patch.object(extended_features_oos, "load_rows", return_value=[]):
            result = extended_features_oos.evaluate("5m")
        self.assertEqual(result["status"], "DEFERRED")
        self.assertEqual(result["reason"], "insufficient_rows")


if __name__ == "__main__":
    unittest.main()
