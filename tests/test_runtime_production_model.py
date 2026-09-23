import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src import runtime_production_model as rpm


class TestRuntimeProductionModel(unittest.TestCase):
    def _write_bundle(self, root: Path, horizon: str = "5m", version: str = "production.v1", candidate: bool = False):
        model_dir = root / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        x = np.array(
            [
                np.zeros(len(rpm.FEATURES)),
                np.ones(len(rpm.FEATURES)),
                np.full(len(rpm.FEATURES), 2.0),
                np.full(len(rpm.FEATURES), -1.0),
                np.full(len(rpm.FEATURES), -2.0),
                np.full(len(rpm.FEATURES), 0.5),
            ]
        )
        y = np.array(["DOWN", "FLAT", "UP", "DOWN", "FLAT", "UP"])
        model = LogisticRegression(max_iter=2000, random_state=42).fit(x, y)
        joblib.dump(model, model_dir / f"{horizon}.joblib")
        (model_dir / f"{horizon}.json").write_text(
            json.dumps(
                {
                    "model_version": version,
                    "horizon": horizon,
                    "classes": list(rpm.CLASSES),
                    "features": list(rpm.FEATURES),
                    "artifact": f"{horizon}.joblib",
                    "candidate": candidate,
                }
            ),
            encoding="utf-8",
        )

    def _db(self, root: Path, version: str | None):
        db = root / "predictions.db"
        with sqlite3.connect(db) as con:
            con.execute(
                "CREATE TABLE model_registry (horizon TEXT PRIMARY KEY, production_version TEXT NOT NULL, updated_at_utc TEXT NOT NULL)"
            )
            if version is not None:
                con.execute(
                    "INSERT INTO model_registry VALUES ('5m', ?, '2026-09-24T00:00:00+00:00')",
                    (version,),
                )
            con.commit()
        return db

    def test_resolves_current_production_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_bundle(root, version="production.v1")
            db = self._db(root, "production.v1")
            old_model_dir, old_db = rpm.MODEL_DIR, rpm.DB
            try:
                rpm.MODEL_DIR, rpm.DB = root / "models", db
                bundle = rpm.resolve_production_model("5m")
                self.assertEqual(bundle.model_version, "production.v1")
                self.assertTrue(bundle.artifact.exists())
                self.assertFalse(bundle.metadata["candidate"])
                self.assertEqual(len(bundle.sha256), 64)
            finally:
                rpm.MODEL_DIR, rpm.DB = old_model_dir, old_db

    def test_refuses_candidate_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_bundle(root, version="candidate.v2", candidate=True)
            db = self._db(root, None)
            old_model_dir, old_db = rpm.MODEL_DIR, rpm.DB
            try:
                rpm.MODEL_DIR, rpm.DB = root / "models", db
                with self.assertRaisesRegex(RuntimeError, "refused_candidate"):
                    rpm.resolve_production_model("5m")
            finally:
                rpm.MODEL_DIR, rpm.DB = old_model_dir, old_db

    def test_refuses_registry_metadata_generation_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_bundle(root, version="metadata.v2")
            db = self._db(root, "production.v1")
            old_model_dir, old_db = rpm.MODEL_DIR, rpm.DB
            try:
                rpm.MODEL_DIR, rpm.DB = root / "models", db
                with self.assertRaisesRegex(RuntimeError, "registry_metadata_mismatch"):
                    rpm.resolve_production_model("5m")
            finally:
                rpm.MODEL_DIR, rpm.DB = old_model_dir, old_db


if __name__ == "__main__":
    unittest.main()
