import unittest

from src.model_compare import dedupe_exact_prediction_events, prediction_event_key


class PredictionEventIdentityTests(unittest.TestCase):
    def _row(self, model="m1", probability=0.7):
        return {
            "created": "2026-09-22T00:00:00+00:00",
            "target": "2026-09-22T00:05:00+00:00",
            "model_version": model,
            "x": [0.1, 0.2],
            "production": [0.1, probability, 0.2],
        }

    def test_exact_events_have_same_key(self):
        a = self._row()
        self.assertEqual(prediction_event_key(a), prediction_event_key(dict(a)))

    def test_model_or_probability_difference_changes_key(self):
        base = self._row()
        self.assertNotEqual(
            prediction_event_key(base),
            prediction_event_key(self._row(model="m2")),
        )
        self.assertNotEqual(
            prediction_event_key(base),
            prediction_event_key(self._row(probability=0.6)),
        )

    def test_dedupe_keeps_distinct_events(self):
        a = self._row()
        b = self._row(model="m2")
        c = dict(a)
        self.assertEqual(len(dedupe_exact_prediction_events([a, b, c])), 2)


if __name__ == "__main__":
    unittest.main()
