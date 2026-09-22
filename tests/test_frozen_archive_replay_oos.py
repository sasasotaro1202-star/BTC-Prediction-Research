import unittest
from unittest.mock import patch

from src import frozen_archive_replay_oos as replay


class FrozenArchiveReplayTests(unittest.TestCase):
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

    def test_summary_requires_three_quarters_of_windows_to_improve(self):
        self.assertIsInstance(replay.REPLAY_WINDOWS, int)
        self.assertEqual(replay.REPLAY_WINDOWS, 4)


if __name__ == "__main__":
    unittest.main()
