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

    def test_workflow_emits_auditable_evidence_envelope(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("Build auditable production prediction evidence envelope", text)
        self.assertIn("VERIFIED_RUNTIME_BINDING_AND_PERSISTENCE_CONTRACT", text)
        self.assertIn("NOT_COMPUTED_AT_PRODUCTION_RUNTIME", text)
        self.assertIn("forecast_lifetime_seconds", text)
        self.assertIn('payload["target_10m_jst"]', text)
        self.assertIn('horizons = {', text)
        self.assertIn("source_checks", text)
        self.assertIn("btc-current-production-evidence", text)

    def test_workflow_does_not_enable_fallback_selection(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn("use_bybit_fallback = True", text)
        self.assertNotIn("use_coinbase_fallback = True", text)
        self.assertIn("candidate exposed as production", text)


if __name__ == "__main__":
    unittest.main()
