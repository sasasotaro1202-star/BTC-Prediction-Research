import unittest
from pathlib import Path
import numpy as np

from src.binary_target_research import _metrics, build_rows
from src.label_policy import BINARY_CLASSES, BINARY_TARGET_VERSION, binary_direction_from_return

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "btc_binary_target_research.yml"


class TestBinaryTargetResearch(unittest.TestCase):
    def test_workflow_restores_prediction_state_before_oos(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("Require repository prediction state", text)
        self.assertIn("Restore repository prediction state", text)
        self.assertIn("data/predictions.db.gz", text)
        self.assertIn("restore_db", text)
        self.assertIn("test -s data/predictions.db", text)

    def test_binary_policy_has_no_flat(self):
        self.assertEqual(BINARY_CLASSES, ("DOWN", "UP"))
        self.assertEqual(binary_direction_from_return(-0.001), "DOWN")
        self.assertEqual(binary_direction_from_return(0.001), "UP")
        self.assertEqual(binary_direction_from_return(0.0), "DOWN")

    def test_binary_metrics_contract(self):
        out = _metrics(["DOWN", "UP", "UP", "DOWN"], [0.1, 0.9, 0.8, 0.2])
        self.assertEqual(out["n"], 4)
        self.assertEqual(out["up_rate"], 0.5)
        self.assertAlmostEqual(out["accuracy"], 1.0)
        self.assertTrue(np.isfinite(out["logloss"]))
        self.assertTrue(np.isfinite(out["brier"]))
        self.assertTrue(np.isfinite(out["ece"]))

    def test_build_rows_uses_future_close_and_emits_two_classes(self):
        raw = []
        base = 1_700_000_000_000
        for i in range(40):
            close = 100.0 + i
            raw.append([base + i * 60_000, close, close + 1, close - 1, close, 10.0])
        rows = build_rows(raw, "5m")
        self.assertTrue(rows)
        self.assertTrue(all(r["target_version"] == BINARY_TARGET_VERSION for r in rows))
        self.assertTrue(set(r["y"] for r in rows).issubset(set(BINARY_CLASSES)))
        self.assertNotIn("FLAT", {r["y"] for r in rows})
        first = rows[0]
        self.assertEqual(first["y"], "UP")


if __name__ == "__main__":
    unittest.main()
