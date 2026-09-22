import unittest

from src.context_router_oos import _causal_purged_train_rows


class ContextRouterPITTests(unittest.TestCase):
    def test_unsettled_target_is_excluded_before_embargo_cutoff(self):
        rows = [
            {
                "created": "2026-09-21T23:50:00+00:00",
                "target": "2026-09-22T00:30:00+00:00",
                "x": [0.0] * 15,
                "y": "UP",
            },
            {
                "created": "2026-09-22T01:00:00+00:00",
                "target": "2026-09-22T01:30:00+00:00",
                "x": [0.0] * 15,
                "y": "DOWN",
            },
        ]
        # For a 02:00 test start and a 60-minute embargo, only labels settled
        # strictly before 01:00 are allowed into training.
        train = _causal_purged_train_rows(
            rows,
            "2026-09-22T02:00:00+00:00",
            "5m",
        )
        self.assertEqual(len(train), 1)
        self.assertEqual(train[0]["target"], "2026-09-22T00:30:00+00:00")

    def test_invalid_or_noncausal_timestamp_rows_fail_closed(self):
        rows = [
            {
                "created": "2026-09-22T01:00:00+00:00",
                "target": "2026-09-22T00:30:00+00:00",
                "x": [0.0] * 15,
                "y": "UP",
            },
            {
                "created": "not-a-time",
                "target": "2026-09-22T00:30:00+00:00",
                "x": [0.0] * 15,
                "y": "DOWN",
            },
        ]
        self.assertEqual(
            _causal_purged_train_rows(
                rows,
                "2026-09-22T02:00:00+00:00",
                "5m",
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
