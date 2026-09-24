import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import selective_prediction_oos as sp


class TestSelectivePredictionOOS(unittest.TestCase):
    def test_adaptive_threshold_uses_previous_scores_only(self):
        self.assertAlmostEqual(
            sp._adaptive_threshold([0.60, 0.70, 0.80, 0.90], 0.50),
            0.75,
            places=8,
        )
        self.assertEqual(sp._adaptive_threshold([0.70] * 10, 0.30), 0.70)

    def test_metrics_shape(self):
        probs = [[0.70, 0.20, 0.10], [0.10, 0.20, 0.70]]
        result = sp._metrics(["DOWN", "UP"], probs)
        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["accuracy"], 1.0)

    def test_policy_is_research_only(self):
        with patch.object(sp, "_evaluate_horizon", return_value={"status": "DEFERRED"}):
            with tempfile.TemporaryDirectory() as td:
                original = sp.OUT
                try:
                    sp.OUT = Path(td) / "selective.json"
                    sp.main()
                    obj = json.loads(sp.OUT.read_text(encoding="utf-8"))
                finally:
                    sp.OUT = original
        self.assertTrue(obj["research_only"])
        self.assertFalse(obj["production_changed"])


if __name__ == "__main__":
    unittest.main()
