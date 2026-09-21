import json
import math
import unittest
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from feature_schema import FEATURE_COUNT, FEATURES
CLASSES = ["DOWN", "FLAT", "UP"]


class RuntimeModelGuards(unittest.TestCase):
    def test_published_model_artifacts_are_loadable(self):
        for horizon in ("5m", "10m"):
            model_path = ROOT / "models" / f"{horizon}.joblib"
            meta_path = ROOT / "models" / f"{horizon}.json"
            if not model_path.exists() and not meta_path.exists():
                continue
            self.assertTrue(model_path.exists(), f"{horizon} artifact missing")
            self.assertTrue(meta_path.exists(), f"{horizon} metadata missing")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta.get("horizon"), horizon)
            self.assertEqual(meta.get("artifact"), f"{horizon}.joblib")
            self.assertEqual(meta.get("classes"), CLASSES)
            self.assertEqual(meta.get("features"), list(FEATURES))
            self.assertEqual(len(meta.get("features", [])), FEATURE_COUNT)
            model = joblib.load(model_path)
            self.assertEqual(list(model.classes_), CLASSES)

    def test_published_models_return_finite_normalized_probabilities(self):
        x = np.zeros((3, FEATURE_COUNT), dtype=float)
        for horizon in ("5m", "10m"):
            model_path = ROOT / "models" / f"{horizon}.joblib"
            if not model_path.exists():
                continue
            model = joblib.load(model_path)
            p = np.asarray(model.predict_proba(x), dtype=float)
            self.assertEqual(p.shape, (3, 3))
            self.assertTrue(np.isfinite(p).all())
            self.assertTrue((p >= 0).all())
            self.assertTrue(np.allclose(p.sum(axis=1), 1.0, atol=1e-6))

    def test_metadata_temperature_is_safe_when_present(self):
        for horizon in ("5m", "10m"):
            meta_path = ROOT / "models" / f"{horizon}.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "temperature" in meta:
                t = float(meta["temperature"])
                self.assertTrue(math.isfinite(t))
                self.assertGreaterEqual(t, 0.5)
                self.assertLessEqual(t, 3.0)


if __name__ == "__main__":
    unittest.main()
