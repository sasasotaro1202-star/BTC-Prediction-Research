import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import predict  # noqa: E402
import db as db_module  # noqa: E402


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

def test_primary_production_source_rejects_venue_fallback():
    with __import__("pytest").raises(ValueError, match="production_fallback_disabled"):
        predict.enforce_primary_production_source({"price_feature_fallback": "coinbase"})


def test_primary_production_source_accepts_normal_binance_mode():
    predict.enforce_primary_production_source({"price_feature_fallback": "none"})
