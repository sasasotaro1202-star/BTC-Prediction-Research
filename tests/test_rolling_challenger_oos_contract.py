import unittest

from src.rolling_challenger_oos import factories


class TestRollingChallengerFactoryContract(unittest.TestCase):
    def test_all_registered_factories_can_be_constructed(self):
        expected = {
            "logreg_c0.1",
            "extra_trees_500",
            "rf_replay",
            "rf_balanced",
            "extra_trees_balanced",
            "hgb",
            "gaussian_nb",
            "adaptive_soft_ensemble",
        }
        registry = factories()
        self.assertEqual(set(registry), expected)
        for name, factory in registry.items():
            with self.subTest(factory=name):
                self.assertIsNotNone(factory())


if __name__ == "__main__":
    unittest.main()


def test_rolling_challenger_cancels_stale_duplicate_runs():
    from pathlib import Path
    workflow = Path(".github/workflows/btc_rolling_challenger.yml").read_text(encoding="utf-8")
    assert "cancel-in-progress: true" in workflow
