import unittest
import numpy as np

from src.time_regime_oos import _timestamp_features, _augment


class TimeRegimeOOSTests(unittest.TestCase):
    def test_timestamp_features_are_bounded_and_periodic(self):
        a = _timestamp_features("2026-09-22T00:00:00+00:00")
        b = _timestamp_features("2026-09-23T00:00:00+00:00")
        self.assertEqual(len(a), 5)
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue(np.all(np.abs(np.asarray(a[:4])) <= 1.0))
        self.assertEqual(a[4], 0.0)
        self.assertEqual(b[4], 0.0)

    def test_weekend_flag_is_explicit(self):
        saturday = _timestamp_features("2026-09-19T12:00:00+00:00")
        monday = _timestamp_features("2026-09-21T12:00:00+00:00")
        self.assertEqual(saturday[4], 1.0)
        self.assertEqual(monday[4], 0.0)


    def test_archive_style_id_is_safe_for_chronological_sort(self):
        row = {
            "id": "archive:binance_vision:1789314900000:5m",
            "created": "2026-09-22T08:00:00+00:00",
            "x": [0.0] * 15,
            "y": "UP",
        }
        out = _augment([row])
        self.assertEqual(out[0]["id"], row["id"])
        self.assertEqual(len(out[0]["x"]), 20)

    def test_augment_preserves_identity_and_adds_exactly_five_features(self):
        row = {
            "id": 1,
            "created": "2026-09-22T12:30:00+00:00",
            "x": [0.0] * 15,
            "y": "UP",
            "production": [0.2, 0.3, 0.5],
        }
        out = _augment([row])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["id"], 1)
        self.assertEqual(len(out[0]["x"]), 20)
        self.assertTrue(np.isfinite(np.asarray(out[0]["x"], dtype=float)).all())



class StateTrajectoryOOSTests(unittest.TestCase):
    def test_direct_multi_horizon_trajectory_contract_exists(self):
        from src.state_trajectory_oos import build_trajectory_oos
        from datetime import datetime, timedelta, timezone
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = [
            {"id": i, "created": (start + timedelta(minutes=5 * i)).isoformat(), "x": [float(i % 4)] * 15}
            for i in range(220)
        ]
        out = build_trajectory_oos(
            rows, steps=(1, 2), n_clusters=4, min_train=80, test_block=20, min_oos=60
        )
        self.assertEqual(out["evaluation_mode"], "direct_multi_horizon")
        self.assertTrue(out["research_only"])
        self.assertFalse(out["production_changed"])

    def test_missing_intervals_do_not_create_false_trajectory_pairs(self):
        from src.state_trajectory_oos import _build_pairs
        from datetime import datetime, timedelta, timezone
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = [
            {"id": i, "created": (start + timedelta(minutes=5 * i)).isoformat(), "x": [float(i)] * 15}
            for i in range(12)
        ]
        rows.pop(5)
        pairs = _build_pairs(rows, 2)
        self.assertFalse(any(p["from_index"] == 4 for p in pairs))
        self.assertTrue(all(p["elapsed_minutes"] == 10 for p in pairs))


    def test_trajectory_state_vocabulary_is_frozen_for_cross_fold_comparability(self):
        from src.state_trajectory_oos import build_trajectory_oos
        from datetime import datetime, timedelta, timezone
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = [
            {
                "id": i,
                "created": (start + timedelta(minutes=5 * i)).isoformat(),
                "x": [float((i // 20) % 4)] * 15,
            }
            for i in range(220)
        ]
        out = build_trajectory_oos(
            rows, steps=(1, 2), n_clusters=4, min_train=80, test_block=20, min_oos=60
        )
        self.assertEqual(out["state_definition"], "frozen_initial_training_window")
        self.assertEqual(out["state_vocabulary_fit_rows"], 80)


if __name__ == "__main__":
    unittest.main()
