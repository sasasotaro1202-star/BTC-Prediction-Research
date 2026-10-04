import unittest

from src.bootstrap_train import development_gate_passes


class BootstrapGateTests(unittest.TestCase):
    def test_development_gate_uses_only_gate_and_baseline_metrics(self):
        baseline = {"accuracy": 0.50, "logloss": 0.90, "brier": 0.50}
        good_gate = {"accuracy": 0.50, "logloss": 0.85, "brier": 0.49}
        self.assertTrue(development_gate_passes(good_gate, baseline))

    def test_development_gate_rejects_small_relative_gain(self):
        baseline = {"accuracy": 0.50, "logloss": 0.90, "brier": 0.50}
        bad_gate = {"accuracy": 0.50, "logloss": 0.88, "brier": 0.495}
        self.assertFalse(development_gate_passes(bad_gate, baseline))

    def test_development_gate_requires_both_relative_scoring_gains(self):
        baseline = {"accuracy": 0.50, "logloss": 0.90, "brier": 0.50}
        logloss_only = {"accuracy": 0.50, "logloss": 0.86, "brier": 0.497}
        self.assertFalse(development_gate_passes(logloss_only, baseline))

    def test_xgboost_adapter_preserves_canonical_string_labels_when_available(self):
        from src.bootstrap_train import XGBClassifier, candidate_factories

        if XGBClassifier is None:
            self.skipTest("xgboost unavailable")
        import numpy as np

        model = dict(candidate_factories())["xgboost"]
        X = np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.1],
                [1.0, 1.0],
                [1.1, 1.0],
                [1.0, 1.1],
                [-1.0, -1.0],
                [-1.1, -1.0],
                [-1.0, -1.1],
            ],
            dtype=float,
        )
        y = np.asarray(
            ["FLAT", "FLAT", "FLAT", "UP", "UP", "UP", "DOWN", "DOWN", "DOWN"]
        )
        model.fit(X, y)
        probs = model.predict_proba(X)
        pred = model.predict(X)
        self.assertEqual(model.classes_.tolist(), ["DOWN", "FLAT", "UP"])
        self.assertEqual(probs.shape, (len(y), 3))
        self.assertTrue(np.isfinite(probs).all())
        self.assertTrue(set(pred.tolist()) <= {"DOWN", "FLAT", "UP"})

    def test_holdout_cannot_affect_development_gate_api(self):
        baseline = {"accuracy": 0.50, "logloss": 0.90, "brier": 0.50}
        gate = {"accuracy": 0.50, "logloss": 0.85, "brier": 0.49}
        self.assertTrue(development_gate_passes(gate, baseline))
        self.assertEqual(development_gate_passes.__code__.co_argcount, 2)

    def test_development_gate_rejects_accuracy_regression_even_with_loss_gain(self):
        baseline = {"accuracy": 0.50, "logloss": 0.90, "brier": 0.50}
        candidate = {"accuracy": 0.499, "logloss": 0.84, "brier": 0.49}
        self.assertFalse(development_gate_passes(candidate, baseline))

    def test_candidate_factories_include_free_boosting_candidates_when_available(self):
        from src.bootstrap_train import LGBMClassifier, XGBClassifier, candidate_factories

        names = {name for name, _ in candidate_factories()}
        if LGBMClassifier is not None:
            self.assertIn("lightgbm", names)
        if XGBClassifier is not None:
            self.assertIn("xgboost", names)

    def test_publish_never_mutates_existing_production_registry_or_artifacts(self):
        import sqlite3
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from src import bootstrap_train

        class FakeModel:
            classes_ = ["DOWN", "FLAT", "UP"]

        result = {
            "model_name": "soft_ensemble",
            "model": FakeModel(),
            "promotion_allowed": True,
            "gate_score": {"accuracy": 0.6, "logloss": 0.8, "brier": 0.4},
            "baseline_gate": {"accuracy": 0.5, "logloss": 0.9, "brier": 0.5},
            "holdout_score": {"accuracy": 0.58, "logloss": 0.82, "brier": 0.42},
            "holdout_n": 1000,
            "validation_results": [],
            "validation_errors": [],
            "selection_method": "nested_chronological_selection_gate_holdout",
            "train_n": 1000,
            "selection_n": 500,
            "gate_n": 500,
        }

        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "predictions.db"
            model_dir = Path(td) / "models"
            model_dir.mkdir()
            existing = model_dir / "5m.joblib"
            existing.write_bytes(b"existing-production")
            with sqlite3.connect(db) as con:
                con.execute("CREATE TABLE model_registry (horizon TEXT PRIMARY KEY, production_version TEXT, updated_at_utc TEXT)")
                con.execute("INSERT INTO model_registry VALUES (?,?,?)", ("5m", "bootstrap.old.v5.4", "2026-10-04T00:00:00+00:00"))
                con.commit()
            with patch.object(bootstrap_train, "DB", db), patch.object(bootstrap_train, "MODEL_DIR", model_dir):
                ok, meta = bootstrap_train.publish("5m", result)
            self.assertFalse(ok)
            self.assertEqual(meta["status"], "production_publish_blocked_research_only")
            self.assertEqual(meta["existing_production_version"], "bootstrap.old.v5.4")
            self.assertEqual((model_dir / "5m.joblib").read_bytes(), b"existing-production")
            with sqlite3.connect(db) as con:
                version = con.execute("SELECT production_version FROM model_registry WHERE horizon=?", ("5m",)).fetchone()[0]
            self.assertEqual(version, "bootstrap.old.v5.4")

if __name__ == "__main__":
    unittest.main()