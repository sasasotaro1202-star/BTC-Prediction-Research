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

    def test_no_production_flag_in_evaluation_contract(self):
        self.assertIn("production_changed", extended_features_oos.main.__code__.co_names)


if __name__ == "__main__":
    unittest.main()
