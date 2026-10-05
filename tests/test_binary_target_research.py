import unittest
from pathlib import Path
import numpy as np

from src.binary_target_research import (
    _factories,
    _metrics,
    _training_rows_before_cutoff,
    _wfo_diagnostics,
    build_rows,
)
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

    def test_extra_trees_challenger_grid_contains_bounded_leaf_variants(self):
        factories = _factories()
        self.assertIn("extra_trees", factories)
        self.assertIn("extra_trees_leaf5", factories)
        self.assertIn("extra_trees_leaf20", factories)
        self.assertGreaterEqual(len(factories), 5)

    def test_binary_metrics_contract(self):
        out = _metrics(["DOWN", "UP", "UP", "DOWN"], [0.1, 0.9, 0.8, 0.2])
        self.assertEqual(out["n"], 4)
        self.assertEqual(out["up_rate"], 0.5)
        self.assertAlmostEqual(out["accuracy"], 1.0)
        self.assertEqual(len(out["accuracy_ci95"]), 2)
        self.assertGreaterEqual(out["accuracy_ci95"][0], 0.0)
        self.assertLessEqual(out["accuracy_ci95"][1], 1.0)
        self.assertGreaterEqual(out["effective_sample_size_accuracy"], 1.0)
        self.assertLessEqual(out["effective_sample_size_accuracy"], 4.0)
        self.assertTrue(np.isfinite(out["logloss"]))
        self.assertTrue(np.isfinite(out["brier"]))
        self.assertTrue(np.isfinite(out["ece"]))

    def test_knowledge_time_filter_excludes_unmatured_training_labels(self):
        rows = [
            {
                "created": "2026-09-23T17:55:00+00:00",
                "target": "2026-09-23T18:00:00+00:00",
            },
            {
                "created": "2026-09-23T17:59:00+00:00",
                "target": "2026-09-23T18:05:00+00:00",
            },
            {
                "created": "2026-09-23T18:01:00+00:00",
                "target": "2026-09-23T18:02:00+00:00",
            },
            {
                "created": "2026-09-23T17:50:00+00:00",
                "target": "not-a-timestamp",
            },
        ]
        out = _training_rows_before_cutoff(rows, "2026-09-23T18:01:00+00:00")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["target"], "2026-09-23T18:00:00+00:00")

    def test_wfo_diagnostics_capture_fold_stability_and_baseline_effect(self):
        block = {
            "train_end": "2026-01-01T00:00:00+00:00",
            "test_start": "2026-01-01T00:05:00+00:00",
            "test_end": "2026-01-01T08:20:00+00:00",
            "test_n": 500,
            "frequency_baseline": {
                "logloss": 0.70, "brier": 0.26, "accuracy": 0.50, "n": 500
            },
            "extra_trees": {
                "logloss": 0.69, "brier": 0.25, "accuracy": 0.52, "n": 500
            },
        }
        out = _wfo_diagnostics({"block_metrics": [block], "summary": {"extra_trees": {"status": "OK"}}})
        diag = out["models"]["extra_trees"]
        self.assertEqual(diag["n_blocks"], 1)
        self.assertEqual(diag["logloss_non_degraded_blocks"], 1)
        self.assertGreater(diag["mean_logloss_relative_improvement"], 0.0)
        self.assertEqual(diag["newest_block"]["test_start"], block["test_start"])
        self.assertEqual(diag["worst_logloss_block"]["test_end"], block["test_end"])

    def test_binary_source_binds_evidence_to_analysis_sha(self):
        text = (Path(__file__).resolve().parents[1] / "src" / "binary_target_research.py").read_text(encoding="utf-8")
        self.assertIn('"analysis_git_sha"', text)
        self.assertIn('GITHUB_SHA', text)

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
