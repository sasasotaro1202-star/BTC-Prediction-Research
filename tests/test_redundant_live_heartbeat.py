from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "btc_redundant_live_heartbeat.yml"


def test_redundant_live_heartbeat_fails_closed_on_control_plane_errors():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "read_heartbeat()" in text
    assert "read_active_runs()" in text
    assert "after 3 attempts; no recovery dispatch." in text
    assert 'echo "ERROR: Live heartbeat state is missing completed_at_utc; no recovery dispatch."' in text
    assert 'echo "ERROR: Live heartbeat completed_at_utc is invalid; no recovery dispatch."' in text
    assert 'echo "ERROR: Live heartbeat completed_at_utc is in the future; no recovery dispatch."' in text
    assert 'exit 1' in text
    assert "degraded=0" not in text
    assert 'if [ "$age" -ge 300 ]; then' in text
    assert 'if [ "$completed_epoch" -eq 0 ]' not in text


def test_redundant_live_heartbeat_dispatch_has_explicit_fallback_failure():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch API" in text
    assert "gh workflow run btc_live_cycle.yml --ref main" in text
    assert 'echo "ERROR: Live recovery dispatch failed through REST and CLI."' in text
    assert text.count("exit 1") >= 4
