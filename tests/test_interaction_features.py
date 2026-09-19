import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interaction_features import add_interactions, interaction_features


class TestInteractionFeatures(unittest.TestCase):
    def base(self):
        return {
            "ret_1m": 0.001,
            "ret_5m": 0.002,
            "ret_15m": -0.001,
            "ret_30m": 0.003,
            "volatility_5m": 0.0008,
            "volatility_10m": 0.0012,
            "volume_ratio": 1.4,
            "volume_trend": 1.1,
            "ema_gap_5m": 0.0007,
            "ema_gap_10m": 0.0011,
        }

    def test_is_deterministic_and_finite(self):
        a = interaction_features(self.base())
        b = interaction_features(self.base())
        self.assertEqual(a, b)
        self.assertTrue(all(v == v and abs(v) < float("inf") for v in a.values()))

    def test_only_declared_interactions_are_added(self):
        f = self.base()
        out = add_interactions(f)
        self.assertEqual(set(out) - set(f), set(interaction_features(f)))

    def test_missing_input_fails_closed(self):
        f = self.base()
        del f["ret_15m"]
        with self.assertRaisesRegex(KeyError, "ret_15m"):
            interaction_features(f)

    def test_no_target_or_future_fields_are_created(self):
        out = interaction_features(self.base())
        self.assertFalse(any(k in out for k in ("target", "label", "future_return", "y")))


if __name__ == "__main__":
    unittest.main()
