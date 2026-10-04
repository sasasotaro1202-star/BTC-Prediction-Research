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
