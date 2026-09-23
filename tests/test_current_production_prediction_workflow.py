import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "btc_current_production_prediction.yml"


class TestCurrentProductionPredictionWorkflow(unittest.TestCase):
    def test_workflow_is_explicitly_current_main_and_production_bound(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("ref: main", text)
        self.assertIn("SELECT horizon, production_version FROM model_registry", text)
        self.assertIn("models/{horizon}.json", text)
        self.assertIn("model_version", text)
        self.assertIn("python src/predict.py", text)
        self.assertIn("5m", text)
        self.assertIn("10m", text)

    def test_workflow_does_not_enable_fallback_selection(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn("use_bybit_fallback = True", text)
        self.assertNotIn("use_coinbase_fallback = True", text)
        self.assertIn("candidate exposed as production", text)


if __name__ == "__main__":
    unittest.main()
