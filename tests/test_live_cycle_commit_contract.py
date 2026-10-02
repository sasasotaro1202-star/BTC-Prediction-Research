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
    assert block.count("ci_git_fetch --prune origin main") == 1


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


def test_live_cycle_rebuild_preserves_state_and_merges_predictions_without_force_push():
    block = _commit_block()
    assert "STATE_ROOT=/tmp/btc_research_commit_state" in block
    assert "scripts/merge_prediction_state.py" in block
    assert "restore_db(local_archive" in block
    assert "git push --force" not in block
    assert "git push -f" not in block
    assert "failed to publish BTC research state after 5 synchronized attempts" in block
    assert "return 1" in block
