from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "btc_live_cycle.yml"


def _commit_block() -> str:
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index("      - name: Commit BTC research state with conflict-safe merge retry")
    end = text.index("\n# Operational trigger:", start)
    return text[start:end]


def test_live_cycle_syncs_to_current_main_before_every_research_state_push():
    block = _commit_block()
    snapshot = block.index("snapshot_research_state")
    fetch = block.index("ci_git_fetch --prune origin main")
    reset = block.index("git reset --hard \"$remote_sha\"")
    first_commit = block.index("git commit -m 'Merge BTC research state safely'")
    push = block.index("CI_GIT_PUSH_ATTEMPTS=1 ci_git_push origin HEAD:main")

    assert snapshot < fetch < reset < first_commit < push
    assert "for attempt in 1 2 3 4 5; do" in block
    assert block.count("ci_git_fetch --prune origin main") >= 2


def test_depth_cache_sources_network_retry_helper_before_fetch():
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index("      - name: Refresh latest Binance depth cache immediately before prediction")
    end = text.index("\n      - name: Generate next BTC prediction", start)
    block = text[start:end]
    helper = block.index(". scripts/ci_network_retry.sh")
    fetch = block.index("ci_git_fetch --no-tags --depth=1 origin binance-ws-cache")
    assert helper < fetch
    assert "git push --force" not in block
    assert "git push -f" not in block


def test_prediction_boundary_fetches_source_network_retry_helper():
    text = WORKFLOW.read_text(encoding="utf-8")
    sections = [
        (
            "Load latest Binance WebSocket cache from dedicated state branch",
            "Load latest Binance depth cache from dedicated state branch",
        ),
        (
            "Refresh latest Binance WebSocket cache immediately before prediction",
            "Refresh latest Binance depth cache immediately before prediction",
        ),
        (
            "Refresh latest Binance depth cache immediately before prediction",
            "Generate next BTC prediction",
        ),
    ]
    for start_label, end_label in sections:
        start = text.index(f"      - name: {start_label}")
        end = text.index(f"\n      - name: {end_label}", start)
        section = text[start:end]
        if "ci_git_fetch --no-tags --depth=1 origin binance-ws-cache" not in section:
            continue
        assert ". scripts/ci_network_retry.sh" in section
        assert section.index(". scripts/ci_network_retry.sh") < section.index(
            "ci_git_fetch --no-tags --depth=1 origin binance-ws-cache"
        )


def test_live_cycle_rebuild_preserves_state_and_merges_predictions_without_force_push():
    block = _commit_block()
    assert "STATE_ROOT=/tmp/btc_research_commit_state" in block
    assert "scripts/merge_prediction_state.py" in block
    assert "restore_db(local_archive" in block
    assert "git push --force" not in block
    assert "git push -f" not in block
    assert "failed to publish BTC research state after 5 synchronized attempts" in block
    assert "return 1" in block


def test_live_cycle_persists_performance_monitor_artifacts_across_conflict_rebuild():
    block = _commit_block()
    for path in (
        "data/historical_research/performance_snapshot.json",
        "data/historical_research/performance_change.json",
    ):
        assert block.count(path) >= 3


def test_live_cycle_pit_history_restore_is_flat_and_prunes_nested_state():
    block = _commit_block()
    assert "prune_pit_history()" in block
    assert 'find "$root" -mindepth 1 -maxdepth 1 -type d -exec rm -rf -- {} +' in block
    assert "rm -rf data/historical_research/pit_history" in block
    assert 'cp -a data/historical_research/pit_history "$STATE_ROOT/data/historical_research/pit_history"' not in block
    assert 'find "$STATE_ROOT/data/historical_research/pit_history" -maxdepth 1 -type f -name \'pit_*.json\'' in block


def test_live_cycle_does_not_cancel_overlapping_state_writers():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "group: btc-state-main" in workflow
    assert "cancel-in-progress: false" in workflow


def test_live_cycle_refuses_stale_workflow_state_publication():
    block = _commit_block()
    assert "assert_current_workflow_is_not_stale()" in block
    assert 'git ls-tree -r "$remote_sha" -- .github/workflows/btc_live_cycle.yml' in block
    assert 'git hash-object .github/workflows/btc_live_cycle.yml' in block
    assert "stale BTC Live Cycle workflow detected" in block
    assert 'if [ "$assert_current_workflow_is_not_stale_rc" -eq 2 ]; then' in block

