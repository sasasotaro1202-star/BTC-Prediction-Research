import json, tempfile, unittest
from pathlib import Path
from src.robustness_oos import _metrics, _regimes, robust_generation_prefix

class RobustnessTests(unittest.TestCase):
    def test_generation_prefix_is_horizon_specific(self):
        self.assertEqual(
            robust_generation_prefix("5m", "bootstrap.bootstrap_rf"),
            "5m:bootstrap.bootstrap_rf|%",
        )
        self.assertEqual(
            robust_generation_prefix("10m", "bootstrap.bootstrap_rf"),
            "10m:bootstrap.bootstrap_rf|%",
        )
    def test_metrics_normalize_probabilities(self):
        m=_metrics(["UP","DOWN","FLAT"],[[2,0,0],[0,3,0],[0,0,4]])
        self.assertEqual(m["n"],3)
        self.assertAlmostEqual(m["accuracy"],1.0)

    def test_regime_labels_use_current_and_past_only(self):
        rows=[
            {"ret":1.0,"vol":1.0},
            {"ret":-1.0,"vol":2.0},
            {"ret":1.0,"vol":1.0},
        ]
        regimes=_regimes(rows)
        self.assertEqual(regimes[0],"UP_MOMENTUM|LOW_VOL")
        self.assertEqual(regimes[1],"DOWN_MOMENTUM|HIGH_VOL")
        self.assertEqual(regimes[2],"UP_MOMENTUM|LOW_VOL")

if __name__=="__main__": unittest.main()
