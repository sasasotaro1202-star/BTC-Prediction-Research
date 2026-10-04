import unittest

import numpy as np

from src.time_state_trajectory import (
    STATE_COUNT,
    blend_probabilities,
    build_transition_matrix,
    forecast_probabilities,
    state_components,
    state_id,
    states_from_panel,
    trajectory_forecast_matrix,
)


class TestTimeStateTrajectory(unittest.TestCase):
    def _features(self, positive=True):
        row = np.zeros(15, dtype=float)
        row[0] = 1.0 if positive else -1.0
        row[1] = 1.0 if positive else -1.0
        row[2] = 1.0 if positive else -1.0
        row[3] = 1.0 if positive else -1.0
        row[4] = 1.0 if positive else -1.0
        row[13] = 1.0 if positive else -1.0
        row[14] = 1.0 if positive else -1.0
        return row

    def test_state_id_is_deterministic_and_decodable(self):
        panel = np.asarray(
            [
                [0.05, 0.10, 0.85],
                [0.10, 0.10, 0.80],
                [0.08, 0.12, 0.80],
                [0.06, 0.14, 0.80],
            ]
        )
        sid = state_id(panel, self._features(True))
        self.assertTrue(0 <= sid < STATE_COUNT)
        direction, momentum, disagreement = state_components(sid)
        self.assertEqual(direction, "UP")
        self.assertEqual(momentum, "POSITIVE")
        self.assertIn(disagreement, {"LOW", "HIGH"})

    def test_transition_matrix_is_row_stochastic_and_smoothed(self):
        matrix = build_transition_matrix([[0, 0, 0, 1], [1, 2, 2]])
        self.assertEqual(matrix.shape, (STATE_COUNT, STATE_COUNT))
        self.assertTrue(np.all(matrix > 0.0))
        self.assertTrue(np.allclose(matrix.sum(axis=1), 1.0))

    def test_forecast_direction_probabilities_sum_to_one(self):
        matrix = build_transition_matrix([[0, 0, 0, 0], [0, 0, 1, 1]])
        out = forecast_probabilities(0, matrix, 6)
        self.assertEqual(set(out), {str(i) for i in range(1, 7)})
        for probs in out.values():
            self.assertEqual(len(probs), 3)
            self.assertAlmostEqual(sum(probs), 1.0, places=10)

    def test_trajectory_uses_prior_sequences_only(self):
        panel = {
            "logreg": np.asarray(
                [[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.2, 0.6, 0.2]],
                dtype=float,
            ),
            "extra_trees": np.asarray(
                [[0.1, 0.1, 0.8], [0.7, 0.1, 0.2], [0.2, 0.7, 0.1]],
                dtype=float,
            ),
            "hgb": np.asarray(
                [[0.1, 0.2, 0.7], [0.75, 0.1, 0.15], [0.25, 0.55, 0.2]],
                dtype=float,
            ),
            "lightgbm": np.asarray(
                [[0.05, 0.15, 0.8], [0.72, 0.1, 0.18], [0.2, 0.65, 0.15]],
                dtype=float,
            ),
        }
        features = [self._features(True), self._features(False), self._features(True)]
        out = trajectory_forecast_matrix(
            panel,
            features,
            [[0, 0, 0, 0]],
            max_steps=3,
        )
        self.assertEqual(len(out["state_ids"]), 3)
        self.assertEqual(len(out["direction_probabilities_by_step"]["1"]), 3)
        self.assertGreaterEqual(out["transition_evidence"]["observed_transitions"], 1)

    def test_blend_respects_bounds_and_normalizes(self):
        base = np.asarray([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2]])
        trajectory = np.asarray([[0.2, 0.2, 0.6], [0.3, 0.2, 0.5]])
        out = blend_probabilities(base, trajectory, weight=0.10)
        self.assertTrue(np.all(out >= 0.0))
        self.assertTrue(np.allclose(out.sum(axis=1), 1.0))

    def test_states_from_panel_rejects_mismatched_lengths(self):
        panel = {"a": np.asarray([[1.0, 0.0, 0.0]])}
        with self.assertRaises(ValueError):
            states_from_panel(panel, [])

if __name__ == "__main__":
    unittest.main()
