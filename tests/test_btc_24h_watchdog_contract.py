from pathlib import Path


WORKFLOW = Path(".github/workflows/btc_24h_watchdog.yml")


def test_24h_watchdog_binds_dispatch_to_current_main_sha():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert 'git/ref/heads/main' in workflow
    assert 'r.get("head_sha") == main_sha' in workflow
    assert 'gh workflow run btc_24h_autonomous_research.yml --repo "$REPO" --ref main' in workflow


def test_24h_watchdog_verifies_dispatch_created_a_run():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert 'dispatch_started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"' in workflow
    assert 'select(.head_sha == "' in workflow
    assert 'BTC 24H marathon dispatch verified:' in workflow


def test_24h_watchdog_has_failure_circuit_breaker():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "FAILURE_CIRCUIT_BREAKER" in workflow
    assert 'print("ACTION=HOLD")' in workflow
    assert '[ "$action" = "HOLD" ]' in workflow
