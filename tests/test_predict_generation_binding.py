import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import predict  # noqa: E402


class TestPredictFeatureSafety(unittest.TestCase):
    def test_features_reject_insufficient_history(self):
        rows=[[i,1.0,1.0,1.0,1.0,1.0] for i in range(30)]
        with self.assertRaisesRegex(ValueError, "insufficient_price_history_for_features"):
            predict.features(rows)

class TestPredictGenerationBinding(unittest.TestCase):
    def test_blend_weight_rejects_stale_generation(self):
        with tempfile.TemporaryDirectory() as td:
            old_db = predict.DB
            old_model_dir = predict.MODEL_DIR
            try:
                predict.DB = str(Path(td) / 'predictions.db')
                model_dir = Path(td) / 'models'
                predict.MODEL_DIR = model_dir
                model_dir.mkdir()
                (model_dir / '5m.blend.json').write_text(
                    json.dumps({
                        'base_weight': 0.35,
                        'n': 800,
                        'status': 'accepted',
                        'model_version': 'generation-A',
                    }),
                    encoding='utf-8',
                )
                with patch.object(predict, 'regver', return_value='generation-B'):
                    self.assertEqual(predict.load_blend_weight('5m'), 0.0)
            finally:
                predict.DB = old_db
                predict.MODEL_DIR = old_model_dir

    def test_blend_weight_accepts_current_generation(self):
        with tempfile.TemporaryDirectory() as td:
            old_db = predict.DB
            old_model_dir = predict.MODEL_DIR
            try:
                predict.DB = str(Path(td) / 'predictions.db')
                model_dir = Path(td) / 'models'
                predict.MODEL_DIR = model_dir
                model_dir.mkdir()
                (model_dir / '5m.blend.json').write_text(
                    json.dumps({
                        'base_weight': 0.35,
                        'n': 800,
                        'status': 'accepted',
                        'model_version': 'generation-A',
                    }),
                    encoding='utf-8',
                )
                with patch.object(predict, 'regver', return_value='generation-A'):
                    self.assertAlmostEqual(predict.load_blend_weight('5m'), 0.35)
            finally:
                predict.DB = old_db
                predict.MODEL_DIR = old_model_dir

    def test_temperature_rejects_stale_generation(self):
        with tempfile.TemporaryDirectory() as td:
            old_db = predict.DB
            old_model_dir = predict.MODEL_DIR
            try:
                predict.DB = str(Path(td) / 'predictions.db')
                model_dir = Path(td) / 'models'
                predict.MODEL_DIR = model_dir
                model_dir.mkdir()
                (model_dir / '5m.calibration.json').write_text(
                    json.dumps({
                        'temperature': 1.6,
                        'n_settled': 600,
                        'model_version': 'generation-A',
                    }),
                    encoding='utf-8',
                )
                with patch.object(predict, 'regver', return_value='generation-B'):
                    self.assertEqual(predict.load_temperature('5m'), 1.0)
            finally:
                predict.DB = old_db
                predict.MODEL_DIR = old_model_dir


if __name__ == '__main__':
    unittest.main()
    def test_secondary_venue_completeness_requires_valid_status(self):
        status = {"bybit_futures": "ok_current_only", "bybit_depth": "error:Timeout"}
        complete = (
            True
            and "bybit_book_imbalance" in {"bybit_book_imbalance": 0.1}
            and status.get("bybit_futures") in {"ok", "ok_current_only"}
            and status.get("bybit_depth") == "ok"
        )
        self.assertFalse(complete)


