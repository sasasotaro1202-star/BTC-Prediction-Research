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
        self.assertIn('cache_health_status=$?', workflow)
        self.assertIn('if [ "$cache_health_status" -ne 0 ]; then', workflow)
        self.assertIn('stale=true', workflow)
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


    def test_self_heal_checks_event_age_and_contiguous_suffix(self):
        workflow = (ROOT / '.github' / 'workflows' / 'btc_live_cycle.yml').read_text(encoding='utf-8')
        start = workflow.index('      - name: Self-heal stale Binance WS collector')
        end = workflow.index('      - name: Load latest Binance depth cache', start)
        block = workflow[start:end]
        self.assertIn('contiguous_suffix=', block)
        self.assertIn('event_age_ms=', block)
        self.assertIn('if [ "$cache_health_status" -ne 0 ]; then', block)
        self.assertIn('stale=true', block)
        self.assertIn('if suffix < 40', block)
        self.assertIn('event_age > 180000', block)

    def test_prediction_boundary_rejects_stale_event_even_when_retrieval_is_fresh(self):
        workflow = (ROOT / '.github' / 'workflows' / 'btc_live_cycle.yml').read_text(encoding='utf-8')
        start = workflow.index('      - name: Refresh latest Binance WebSocket cache immediately before prediction')
        end = workflow.index('      - name: Refresh latest Binance depth cache immediately before prediction', start)
        block = workflow[start:end]
        self.assertIn('latest_event_age_ms=', block)
        self.assertIn("event_time_ms", block)
        self.assertIn('retrieved_age<0', block)
        self.assertIn('event_age<0', block)
        self.assertIn('event_age>180000', block)
        self.assertIn('stale_or_insufficient_at_prediction_boundary', block)

    def test_self_heal_allows_one_additional_collector_but_caps_at_two(self):
        workflow = (ROOT / '.github' / 'workflows' / 'btc_live_cycle.yml').read_text(encoding='utf-8')
        start = workflow.index('      - name: Self-heal stale Binance WS collector')
        end = workflow.index('      - name: Load latest Binance depth cache', start)
        block = workflow[start:end]
        self.assertIn('if [ "${active:-0}" -ge 2 ]; then', block)
        self.assertIn('gh api "repos/$GITHUB_REPOSITORY/contents/.github/workflows/btc_binance_ws_collector.yml?ref=main"', block)
        self.assertIn('gh api "repos/$GITHUB_REPOSITORY/contents/src/binance_ws.py?ref=main"', block)
        self.assertIn('gh run cancel "$run_id" --repo "$GITHUB_REPOSITORY"', block)
        self.assertIn('headSha', block)
        self.assertIn('current_workflow_blob', block)
        self.assertIn('current_source_blob', block)
        self.assertIn('self-heal dispatch capped to avoid a recovery storm', block)
        self.assertNotIn('if [ "${active:-0}" -gt 0 ]; then', block)

    def test_prediction_boundary_checks_depth_event_age(self):
        workflow = (ROOT / '.github' / 'workflows' / 'btc_live_cycle.yml').read_text(encoding='utf-8')
        start = workflow.index('      - name: Refresh latest Binance depth cache immediately before prediction')
        end = workflow.index('      - name: Generate next BTC prediction', start)
        block = workflow[start:end]
        self.assertIn('binance_depth_event_age_ms=', block)
        self.assertIn("event_age<0", block)
        self.assertIn("event_age>180000", block)
        self.assertIn("Binance depth cache stale_or_insufficient", block)
    def test_live_boundary_keeps_fresh_cache_and_bounds_rest_recovery(self):
        workflow = (ROOT / '.github' / 'workflows' / 'btc_live_cycle.yml').read_text(encoding='utf-8')
        start = workflow.index('      - name: Refresh latest Binance WebSocket cache immediately before prediction')
        end = workflow.index('      - name: Refresh latest Binance depth cache immediately before prediction', start)
        block = workflow[start:end]
        self.assertIn('Loaded fresh Binance Futures WS cache before direct REST recovery.', block)
        self.assertIn('timeout 30s python - "$tmp_cache"', block)
        self.assertIn('"transport": "binance_futures_rest"', block)
        self.assertIn('if [ "$cache_loaded" != true ]; then', block)

    def test_live_boundary_attempts_same_product_rest_recovery_before_polling(self):
        workflow = (ROOT / '.github' / 'workflows' / 'btc_live_cycle.yml').read_text(encoding='utf-8')
        start = workflow.index('      - name: Refresh latest Binance WebSocket cache immediately before prediction')
        end = workflow.index('      - name: Refresh latest Binance depth cache immediately before prediction', start)
        block = workflow[start:end]
        self.assertIn('rest_seed() {', block)
        self.assertIn('from src.market_data import binance_klines', block)
        self.assertIn('raw = binance_klines(False, 720)', block)
        self.assertIn('if rest_seed; then', block)
        self.assertIn('Loaded fresh Binance Futures cache directly via same-product REST failover.', block)
        self.assertIn('if [ "$cache_loaded" != true ]; then', block)
        self.assertIn('for attempt in 1 2 3 4 5 6; do', block)
        self.assertIn('raise SystemExit(f"Binance Futures REST recovery insufficient contiguous rows: {suffix}")', block)

    def test_self_heal_health_check_failure_is_captured_under_set_e(self):
        workflow = (ROOT / '.github' / 'workflows' / 'btc_live_cycle.yml').read_text(encoding='utf-8')
        start = workflow.index('      - name: Self-heal stale Binance WS collector')
        end = workflow.index('      - name: Load latest Binance depth cache', start)
        block = workflow[start:end]
        self.assertIn('set +e\n            cache_health=', block)
        self.assertIn('cache_health_status=$?\n            set -e', block)
        self.assertIn('if [ "$cache_health_status" -ne 0 ]; then', block)
        self.assertIn('stale=true', block)
        self.assertIn('raise SystemExit(2)', block)

if __name__ == '__main__':
    unittest.main()
