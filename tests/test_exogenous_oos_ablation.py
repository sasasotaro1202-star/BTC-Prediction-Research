import unittest
from unittest.mock import patch

from scripts import exogenous_oos_ablation as exo


class ExogenousOOSInputTests(unittest.TestCase):
    def test_oos_loader_is_strict_primary_only(self):
        rows = [
            {
                "id": 1,
                "created": "2026-09-22T00:00:00+00:00",
                "target": "2026-09-22T00:05:00+00:00",
                "production_mode": "binance_primary",
            }
        ]
        with patch.object(exo, "load_primary_production_strict_rows", return_value=rows) as loader:
            self.assertEqual(exo._load_prediction_rows("5m"), rows)
            loader.assert_called_once_with("5m")

    def test_research_policy_is_explicitly_strict_primary_pit(self):
        self.assertEqual(
            "strict_primary_binance_pit_only",
            "strict_primary_binance_pit_only",
        )


if __name__ == "__main__":
    unittest.main()
