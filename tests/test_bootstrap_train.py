import unittest

from src.bootstrap_train import development_gate_passes


class BootstrapGateTests(unittest.TestCase):
    def test_development_gate_uses_only_gate_and_baseline_metrics(self):
        baseline = {"accuracy": 0.50, "logloss": 0.90, "brier": 0.50}
        good_gate = {"accuracy": 0.50, "logloss": 0.85, "brier": 0.49}
        self.assertTrue(development_gate_passes(good_gate, baseline))

    def test_development_gate_rejects_small_or_non_robust_gain(self):
        baseline = {"accuracy": 0.50, "logloss": 0.90, "brier": 0.50}
        bad_gate = {"accuracy": 0.49, "logloss": 0.895, "brier": 0.499}
        self.assertFalse(development_gate_passes(bad_gate, baseline))

    def test_holdout_cannot_affect_development_gate_api(self):
        baseline = {"accuracy": 0.50, "logloss": 0.90, "brier": 0.50}
        gate = {"accuracy": 0.50, "logloss": 0.85, "brier": 0.49}
        self.assertTrue(development_gate_passes(gate, baseline))
        # The function intentionally accepts no holdout argument.
        self.assertEqual(development_gate_passes.__code__.co_argcount, 2)


if __name__ == "__main__":
    unittest.main()
