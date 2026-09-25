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


def test_production_sentinel_retries_control_plane_reads_and_fails_closed():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "retry_api()" in text
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
    assert 'gh run cancel "$run_id" --repo "$REPO"' in text
    assert 'ERROR: failed to cancel stale obsolete Live Cycle run_id=' in text
    assert 'Production heartbeat stale; dispatching one Live Cycle from current main.' in text
