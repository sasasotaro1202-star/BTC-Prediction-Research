from pathlib import Path

WORKFLOW = Path(".github/workflows/btc_binance_flow_research.yml")


def _concurrency_block() -> str:
    text = WORKFLOW.read_text(encoding="utf-8")
    return text.split("\njobs:", 1)[0]


def test_flow_research_collector_is_not_cancelled_by_main_pushes_or_schedule():
    block = _concurrency_block()
    assert "group: btc-binance-flow-research-${{ github.event_name == 'pull_request' && github.head_ref || github.run_id }}" in block
    assert "cancel-in-progress: ${{ github.event_name == 'pull_request' }}" in block
    assert "cancel-in-progress: ${{ github.event_name != 'schedule' }}" not in block


def test_flow_research_pull_requests_remain_cancelable_when_superseded():
    block = _concurrency_block()
    assert "github.event_name == 'pull_request'" in block


def test_flow_cache_publisher_rechecks_remote_recency_after_push_conflicts():
    text = WORKFLOW.read_text(encoding="utf-8")
    retry_region = text.split("for attempt in 1 2 3 4; do", 1)[1].split("done", 1)[0]
    assert 'remote_latest="$(python - <<' not in retry_region or 'local_latest" -le "$remote_latest"' in retry_region
    assert 'Flow checkpoint became older than remote cache after conflict' in retry_region


def test_final_flow_cache_publish_refuses_stale_local_overwrite():
    text = WORKFLOW.read_text(encoding="utf-8")
    final_region = text.split("      - name: Publish dedicated flow cache", 1)[1]
    assert 'git show origin/binance-flow-cache:data/binance_flow_5s.json' in final_region
    assert 'Final flow cache is not newer than dedicated branch' in final_region
    assert 'Final flow cache became older than remote after conflict' in final_region
