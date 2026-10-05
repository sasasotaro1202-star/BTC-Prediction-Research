import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from src import state_trajectory_oos as st


class StateTrajectoryTests(unittest.TestCase):
    @staticmethod
    def rows(n=80):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        return [
            {
                "id": str(i),
                "created": (base + timedelta(minutes=5 * i)).isoformat(),
                "x": [0.0 if (i // 5) % 2 == 0 else 5.0, float(i % 3)],
            }
            for i in range(n)
        ]

    def test_exact_interval_pairs_do_not_bridge_gaps(self):
        rows = self.rows(20)
        rows.pop(5)
        norm = st._normalize_rows(rows)
        pairs = st._build_pairs(norm, 2)
        self.assertFalse(any(p["created"] == rows[4]["created"] for p in pairs))

    def test_state_space_uses_only_supplied_training_rows(self):
        rows = st._normalize_rows(self.rows(40))
        prefix = rows[:20]
        s1 = st._fit_state_space(prefix, 2)
        future_appended_but_not_supplied = rows[:20] + [
            {"created": rows[-1]["created"], "x": np.asarray([999.0, 999.0]), "id": "future"}
        ]
        s2 = st._fit_state_space(future_appended_but_not_supplied[:20], 2)
        self.assertTrue(
            np.allclose(
                np.sort(s1[1].cluster_centers_, axis=0),
                np.sort(s2[1].cluster_centers_, axis=0),
            )
        )

    def test_pairs_respect_stop_before_boundary(self):
        rows = st._normalize_rows(self.rows(30))
        pairs = st._build_pairs(rows, 2, stop_before=10)
        self.assertTrue(all(p["to_index"] < 10 for p in pairs))

    def test_contract_is_research_only_and_no_future_fit(self):
        result = st.build_trajectory_oos(
            self.rows(60), steps=(1,), min_train=10, min_oos=10, test_block=5, n_clusters=2
        )
        self.assertTrue(result["research_only"])
        self.assertFalse(result["production_changed"])
        self.assertFalse(result["promotion_allowed"])
        item = result["horizons"]["5m"]
        if item["status"] == "OK":
            self.assertFalse(item["future_rows_used_for_state_space_fit"])
            self.assertFalse(item["final_holdout_used_for_selection"])

    def test_invalid_step_rejected(self):
        with self.assertRaises(ValueError):
            st.build_trajectory_oos(self.rows(20), steps=(4,))


if __name__ == "__main__":
    unittest.main()
