import unittest
from unittest.mock import patch
import numpy as np

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

    def test_selection_prediction_precedes_selection_containing_fit(self):
        class SpyModel:
            events = []
            def fit(self, X, y):
                self.classes_ = np.asarray(["DOWN", "FLAT", "UP"])
                type(self).events.append(("fit", len(X)))
                return self
            def predict_proba(self, X):
                type(self).events.append(("predict", len(X)))
                return np.tile(np.asarray([[0.30, 0.40, 0.30]]), (len(X), 1))

        dataset = [
            {"ts": i * 60000, "x": [0.0] * 15, "y": {"5m": "FLAT", "10m": "FLAT"}}
            for i in range(20000)
        ]

        def factory():
            return SpyModel()

        with patch.object(replay, "factories", return_value={"spy": factory}):
            with patch.object(replay, "_load_champion", return_value=SpyModel()):
                SpyModel.events = []
                prep = replay._candidate_gate(dataset, "5m")

        train_n = int(len(dataset) * replay.TRAIN_FRAC)
        selection_n = int(len(dataset) * replay.SELECTION_FRAC)
        events = SpyModel.events
        selection_predict_idx = events.index(("predict", selection_n))
        preceding_fits = [size for kind, size in events[:selection_predict_idx] if kind == "fit"]
        self.assertIn(train_n, preceding_fits)
        self.assertNotIn(train_n + selection_n, preceding_fits)
        self.assertEqual(prep["selected"]["name"], "spy")


if __name__ == "__main__":
    unittest.main()
