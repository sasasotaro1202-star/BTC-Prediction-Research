import unittest
from datetime import datetime, timezone

from src.horizon_registry import ALL_HORIZONS, EXTENDED_RESEARCH_HORIZONS, target_at


class HorizonRegistryTests(unittest.TestCase):
    def test_expected_horizons_are_registered(self):
        self.assertEqual(
            ALL_HORIZONS,
            ("5m", "10m", "15m", "30m", "1h", "3h", "6h", "12h", "24h"),
        )
        self.assertEqual(len(EXTENDED_RESEARCH_HORIZONS), 7)

    def test_target_grid_is_horizon_specific_and_future(self):
        cutoff = datetime(2026, 10, 3, 4, 7, tzinfo=timezone.utc)
        targets = [target_at(cutoff, h) for h in EXTENDED_RESEARCH_HORIZONS]
        self.assertTrue(all(target > cutoff for target in targets))
        self.assertEqual(targets, sorted(targets))


if __name__ == "__main__":
    unittest.main()
