import json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

from src import robustness_oos
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

    def test_archive_loader_excludes_rows_before_frozen_model_training(self):
        class FakeModel:
            classes_ = ["DOWN", "FLAT", "UP"]

            def predict_proba(self, X):
                import numpy as np
                return np.tile(np.asarray([[0.7, 0.2, 0.1]]), (len(X), 1))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "models"
            model_dir.mkdir(parents=True)
            meta = {
                "model_version": "bootstrap.bootstrap_rf.vtest",
                "trained_at_utc": "2026-09-22T01:00:00+00:00",
            }
            (model_dir / "5m.json").write_text(json.dumps(meta), encoding="utf-8")
            (model_dir / "5m.joblib").write_bytes(b"placeholder")
            archive_rows = [
                {
                    "id": "old",
                    "created": "2026-09-22T00:59:00+00:00",
                    "x": [0.0] * 15,
                    "y": "UP",
                },
                {
                    "id": "new",
                    "created": "2026-09-22T01:01:00+00:00",
                    "x": [0.0] * 15,
                    "y": "DOWN",
                },
            ]
            with patch.object(robustness_oos, "ROOT", root), \
                 patch.object(robustness_oos, "load_archive_research_rows", return_value=archive_rows), \
                 patch.object(robustness_oos.joblib, "load", return_value=FakeModel()):
                result = robustness_oos.load_research_archive("5m")
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["id"], "new")
            self.assertFalse(result[0]["promotion_evidence_eligible"])
            self.assertEqual(result[0]["p"], [0.1, 0.7, 0.2])

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
