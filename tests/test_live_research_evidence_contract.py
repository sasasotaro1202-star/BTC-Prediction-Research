from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LiveResearchEvidenceContractTests(unittest.TestCase):
    def test_live_cycle_generates_and_persists_research_evidence(self):
        text = (ROOT / ".github" / "workflows" / "btc_live_cycle.yml").read_text(encoding="utf-8")
        self.assertIn("python src/robustness_oos.py", text)
        self.assertIn("BTC_ROBUSTNESS_LIVE_ONLY: '1'", text)
        self.assertIn("python src/promotion_gate.py", text)
        self.assertGreaterEqual(text.count("data/historical_research/robustness_oos_report.json"), 4)
        self.assertGreaterEqual(text.count("data/historical_research/promotion_gate.json"), 4)
        self.assertIn("tests/test_robustness_oos.py", text)
        self.assertIn("expected_eligible", text)
        self.assertIn("n >= minimum", text)
        self.assertIn("immature robustness cohort must be insufficient_data", text)
        self.assertIn("extended_horizon_monitor.py", text)
        self.assertIn("extended_horizon_performance.json", text)
        for horizon in ("15m", "30m", "1h", "3h", "6h", "12h", "24h"):
            self.assertIn(horizon, text)
        self.assertIn("tests/test_live_research_evidence_contract.py", text)

    def test_robustness_is_explicitly_research_only(self):
        text = (ROOT / "src" / "robustness_oos.py").read_text(encoding="utf-8")
        self.assertIn("diagnostic_only_no_model_input_no_promotion_effect", text)
        self.assertIn("BTC_ROBUSTNESS_LIVE_ONLY", text)


    def test_live_cycle_expands_workspace_for_experience_ledger(self):
        text = (ROOT / ".github" / "workflows" / "btc_live_cycle.yml").read_text(encoding="utf-8")
        self.assertIn('PYTHONPATH="${GITHUB_WORKSPACE}/src:${GITHUB_WORKSPACE}/scripts" python src/experience_ledger.py', text)
        self.assertNotIn('PYTHONPATH="\\${GITHUB_WORKSPACE}/src:\\${GITHUB_WORKSPACE}/scripts" python src/experience_ledger.py', text)
    def test_bootstrap_is_not_a_production_refresh_path(self):
        text = (ROOT / ".github" / "workflows" / "btc_live_cycle.yml").read_text(encoding="utf-8")
        self.assertIn("bootstrap as research-only diagnosis", text)
        self.assertIn("Production generations remain stable", text)
        self.assertNotIn("max(ages, default=float(\"inf\")) >= 24 * 3600", text)

    def test_bootstrap_training_module_cannot_publish_to_production(self):
        text = (ROOT / "src" / "bootstrap_train.py").read_text(encoding="utf-8")
        self.assertIn("def _current_production_version", text)
        self.assertIn("production_publish_blocked_research_only", text)
        self.assertIn("never writes a Production model or model_registry entry", text)
        self.assertNotIn("joblib.dump(model, MODEL_DIR / f\"{horizon}.joblib\")", text)

if __name__ == "__main__":
    unittest.main()