from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "btc_production_sentinel.yml"


def test_production_sentinel_has_real_shell_variable_expansion():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert '<<<"$live_json"' in text
    assert '<<<"$latest"' in text
    assert '[ -z "$updated" ]' in text
    assert 'age=$((now-epoch))' in text
    assert '[ "$active" -eq 0 ]' in text
    assert '[ "$age" -ge 300 ]' in text
    assert '[ "$head" != "$main_sha" ]' in text
    for token in ("\\$live_json", "\\$latest", "\\$updated", "\\$active", "\\$age", "\\$head", "\\$main_sha", "age=\\$((now-epoch))"):
        assert token not in text


def test_production_sentinel_uses_shared_bounded_retry_and_fails_closed():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert ". scripts/ci_network_retry.sh" in text
    assert "ci_gh_api_get" in text
    assert "ci_gh run cancel" in text
    assert "after 3 attempts" in text
    assert "unable to read current main SHA after 3 attempts" in text
    assert "unable to read Live workflow runs after 3 attempts" in text
    assert "ERROR: Live Cycle recovery dispatch failed." in text
    assert "ERROR: next sentinel generation dispatch failed." in text
    assert "set -euo pipefail" in text



def test_production_sentinel_cancels_only_old_obsolete_active_runs():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'stale_ids=' in text
    assert '[ "$active" -eq 0 ]' in text
    assert 'select(.head_sha != $main_sha)' in text
    assert '>= 720' in text
    assert 'ci_gh run cancel "$run_id" --repo "$REPO"' in text
    assert 'ERROR: failed to cancel stale obsolete Live Cycle run_id=' in text
    assert 'Production heartbeat stale; dispatching one Live Cycle from current main.' in text


def test_production_sentinel_deferred_recovery_requires_post_defer_fresh_cache_event():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "data/live_cycle_status.json?ref=main" in text
    assert "data/binance_ws_1m.json?ref=binance-ws-cache" in text
    assert '[ "$cycle_mode" = "deferred" ]' in text
    assert "deferred_completed_ms=0" in text
    assert "cache_age_ms=$((now*1000-cache_event_ms))" in text
    assert '[ "$cache_age_ms" -le 180000 ]' in text
    assert '[ "$cache_event_ms" -gt "$deferred_completed_ms" ]' in text
    assert '[ "$deferred_cache_recovery" -eq 1 ]' in text


def test_production_sentinel_fails_closed_for_deferred_cache_recovery_read_failures():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "deferred-cache recovery disabled." in text
    assert "cache_fresh=0" in text
    assert "cache_event_ms=0" in text
