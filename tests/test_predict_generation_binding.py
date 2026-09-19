import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import predict  # noqa: E402


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
                    self.assertEqual(predict.load_blend_weight('5m'), 0.20)
            finally:
                predict.DB = old_db
                predict.MODEL_DIR = old_model_dir

    def test_blend_weight_accepts_current_generation(self):
        with tempfile.TemporaryDirectory() as td:
            old_db = predict.DB
            try:
                predict.DB = str(Path(td) / 'predictions.db')
                model_dir = Path(td) / 'models'
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

    def test_temperature_rejects_stale_generation(self):
        with tempfile.TemporaryDirectory() as td:
            old_db = predict.DB
            try:
                predict.DB = str(Path(td) / 'predictions.db')
                model_dir = Path(td) / 'models'
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


if __name__ == '__main__':
    unittest.main()
