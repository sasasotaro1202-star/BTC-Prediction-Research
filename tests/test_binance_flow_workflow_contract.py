from pathlib import Path

WORKFLOW = Path(".github/workflows/btc_binance_flow_research.yml")


def test_flow_collector_has_global_single_writer_concurrency():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    collector = workflow.split("\n  collector:", 1)[1]
    assert "concurrency:" in collector
    assert "group: btc-binance-flow-cache" in collector
    assert "cancel-in-progress: false" in collector


def test_flow_research_long_collector_is_schedule_or_manual_only():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    collector = workflow.split("\n  collector:", 1)[1]
    assert "github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'" in collector
    assert "github.event_name == 'push'" not in collector
    assert "capture-seconds 1500" in collector




def test_flow_cache_publisher_rechecks_remote_recency_after_push_conflicts():
    text = WORKFLOW.read_text(encoding="utf-8")
    retry_region = text.split("for attempt in 1 2 3 4; do", 1)[1].split("done", 1)[0]
    fetch_pos = retry_region.index("git fetch --no-tags --depth=1 origin binance-flow-cache")
    show_pos = retry_region.index("git show origin/binance-flow-cache:data/binance_flow_5s.json", fetch_pos)
    compare_pos = retry_region.index('if [ "$local_latest" -le "$remote_latest" ]; then', show_pos)
    assert fetch_pos < show_pos < compare_pos
    assert "Flow checkpoint became older than remote cache after conflict" in retry_region


def test_final_flow_cache_publish_refuses_stale_local_overwrite():
    text = WORKFLOW.read_text(encoding="utf-8")
    final_region = text.split("      - name: Publish dedicated flow cache", 1)[1]
    assert 'git show origin/binance-flow-cache:data/binance_flow_5s.json' in final_region
    assert 'Final flow cache is not newer than dedicated branch' in final_region
    assert 'Final flow cache became older than remote after conflict' in final_region


def test_flow_collector_accepts_intentional_publisher_shutdown():
    text = WORKFLOW.read_text(encoding="utf-8")
    capture_region = text.split("      - name: Capture rolling flow/liquidation window with resumable checkpoints", 1)[1]
    assert 'wait "$publisher_pid" || publisher_status=$?' in capture_region
    assert 'publisher_status" -ne 143' in capture_region
    assert 'publisher loop exited unexpectedly' in capture_region