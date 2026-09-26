import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from src.class_prior_recalibration_oos import (
    CLASSES,
    _history_ready,
    adjust_probs,
    empirical_prior,
    MIN_ROWS,
    MIN_OOS_BLOCKS,
    TEST_BLOCK,
    MIN_HISTORY,
    evaluate,
)


class TestClassPriorRecalibration(unittest.TestCase):
    def _row(self, minutes, y, created_offset=0):
        created = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minutes)
        target = created + timedelta(minutes=5 + created_offset)
        return {
            "created": created.isoformat(),
            "target": target.isoformat(),
            "y": y,
            "production": [0.49, 0.02, 0.49],
        }

    def test_adjustment_preserves_probability_contract_and_class_order(self):
        raw = [[0.49, 0.02, 0.49]]
        adjusted = adjust_probs(
            raw,
            target_prior=[0.33, 0.34, 0.33],
            model_prior=[0.49, 0.02, 0.49],
            gamma=1.0,
        )
        self.assertEqual(CLASSES, ["DOWN", "FLAT", "UP"])
        self.assertAlmostEqual(float(adjusted.sum()), 1.0, places=9)
        self.assertGreater(float(adjusted[1]), float(raw[0][1]))

    def test_empirical_prior_uses_smoothing(self):
        history = [
            {"y": "DOWN"},
            {"y": "DOWN"},
            {"y": "UP"},
        ]
        prior = empirical_prior(history)
        self.assertEqual(len(prior), 3)
        self.assertAlmostEqual(float(prior.sum()), 1.0, places=9)
        self.assertGreater(float(prior[1]), 0.0)


    def test_research_defers_before_small_sample_tuning(self):
        self.assertGreaterEqual(
            MIN_ROWS,
            ((MIN_HISTORY + (MIN_OOS_BLOCKS * TEST_BLOCK)) / (1.0 - 0.20)),
        )
        with patch(
            "src.class_prior_recalibration_oos.load_primary_production_strict_rows",
            return_value=[self._row(i, "DOWN") for i in range(MIN_ROWS - 1)],
        ):
            result = evaluate("5m")
        self.assertEqual(result["status"], "DEFERRED")
        self.assertEqual(result["promotion_evidence_eligible"], False)
        self.assertEqual(result["reason"], "insufficient_strict_binance_primary_rows")

    def test_history_is_strictly_causal(self):
        rows = [
            self._row(0, "DOWN"),
            self._row(10, "FLAT"),
            self._row(20, "UP"),
        ]
        # The second row's target equals 15m.  A history beginning at 15m must
        # not include it, and a prediction created at/after its target is invalid.
        rows[1]["target"] = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15)).isoformat()
        rows[2]["created"] = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=30)).isoformat()
        rows[2]["target"] = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=35)).isoformat()
        history = _history_ready(rows, "2026-01-01T00:15:00+00:00")
        self.assertEqual([r["y"] for r in history], ["DOWN"])

if __name__ == "__main__":
    unittest.main()
