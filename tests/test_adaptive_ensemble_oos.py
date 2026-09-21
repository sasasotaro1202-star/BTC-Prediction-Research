import unittest
from pathlib import Path
from unittest.mock import patch

from src import adaptive_ensemble_oos as adaptive


class AdaptiveEnsembleHoldoutTests(unittest.TestCase):
    def test_development_selection_and_holdout_are_separate(self):
        rows = [{"y": "DOWN"}] * 2500 + [{"y": "UP"}] * 625
        with patch.object(adaptive, "load_rows", return_value=rows):
            with patch.object(adaptive, "_evaluate_development", return_value=({"blocks": 8}, [], True)):
                with patch.object(adaptive, "_predict_block", side_effect=[None, None]):
                    result = adaptive.evaluate("5m")
        self.assertEqual(result["status"], "OK")
        self.assertFalse(result["final_holdout_used_for_selection"])
        self.assertTrue(result["final_holdout_protected"])
        self.assertTrue(result["eligible_pending_frozen_holdout_confirmation"])

    def test_insufficient_rows_fails_closed(self):
        with patch.object(adaptive, "load_rows", return_value=[]):
            result = adaptive.evaluate("5m")
        self.assertEqual(result["status"], "DEFERRED")
        self.assertEqual(result["reason"], "insufficient_rows_for_protected_holdout")


if __name__ == "__main__":
    unittest.main()
