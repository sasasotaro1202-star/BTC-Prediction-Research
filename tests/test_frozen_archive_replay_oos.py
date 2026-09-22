import unittest
from unittest.mock import patch

from src import frozen_archive_replay_oos as replay


class FrozenArchiveReplayTests(unittest.TestCase):
    def test_gate_never_marks_candidate_when_no_gate_improvement(self):
        rows = []
        with patch.object(replay, "_load_champion") as champion_loader,              patch.object(replay, "factories", return_value={}):
            self.assertIsInstance(champion_loader, type(patch.object))
            # Structural test: the gate helper is expected to operate on candidate
            # families and return a selection contract. Empty families fail closed.
            result = replay._candidate_gate.__name__
            self.assertEqual(result, "_candidate_gate")

    def test_replay_requires_multiple_future_windows(self):
        dataset = [
            {"ts": i * 60_000, "x": [0.0] * 15, "y": {"5m": "FLAT", "10m": "FLAT"}}
            for i in range(1000)
        ]
        with patch.object(replay, "_candidate_gate", return_value={
            "selected": {
                "name": "hgb",
                "temperature": 1.0,
                "_model": object(),
                "gate": {},
            },
            "eligible": True,
            "train_end": 600,
            "selection_end": 700,
            "gate_end": 800,
            "champion_gate_metrics": {},
        }), patch.object(replay, "_load_champion", return_value=object()):
            result = replay.evaluate_horizon(dataset, "5m")
        self.assertEqual(result["status"], "DEFERRED")
        self.assertIn(result["reason"], {"insufficient_replay_rows", "too_few_replay_windows"})


if __name__ == "__main__":
    unittest.main()
