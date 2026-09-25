import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestLiveCollectorSelfHeal(unittest.TestCase):
    def test_live_workflow_self_heals_stale_collector(self):
        workflow = (ROOT / '.github' / 'workflows' / 'btc_live_cycle.yml').read_text(encoding='utf-8')
        self.assertIn('actions: write', workflow)
        self.assertIn('GH_TOKEN: ${{ github.token }}', workflow)
        self.assertIn('gh run list --repo "$GITHUB_REPOSITORY"', workflow)
        self.assertIn('--workflow btc_binance_ws_collector.yml', workflow)
        self.assertIn('select(.status == "queued" or .status == "in_progress")', workflow)
        self.assertIn('gh workflow run btc_binance_ws_collector.yml --repo "$GITHUB_REPOSITORY" --ref main', workflow)
        self.assertIn('age_ms -gt 180000', workflow)
        self.assertIn('Live prediction remains fail-closed on stale data.', workflow)

    def test_self_heal_is_nonfatal_when_github_actions_api_is_unavailable(self):
        workflow = (ROOT / '.github' / 'workflows' / 'btc_live_cycle.yml').read_text(encoding='utf-8')
        start = workflow.index('      - name: Self-heal stale Binance WS collector')
        end = workflow.index('      - name: Load latest Binance depth cache', start)
        block = workflow[start:end]
        self.assertIn('if [ "$list_status" -ne 0 ]; then', block)
        self.assertIn('exit 0', block)
        self.assertIn('if [ "$dispatch_status" -ne 0 ]; then', block)

    def test_prediction_policy_is_not_relaxed_by_self_heal(self):
        workflow = (ROOT / '.github' / 'workflows' / 'btc_live_cycle.yml').read_text(encoding='utf-8')
        start = workflow.index('      - name: Self-heal stale Binance WS collector')
        end = workflow.index('      - name: Generate next BTC prediction', start)
        block = workflow[start:end]
        self.assertNotIn('python src/predict.py', block)
        self.assertNotIn('fallback_not_allowed', block)


if __name__ == '__main__':
    unittest.main()
