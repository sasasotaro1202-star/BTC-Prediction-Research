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

def test_24h_watchdog_self_enables_marathon_before_dispatch():
    text = Path('.github/workflows/btc_24h_watchdog.yml').read_text(encoding='utf-8')
    assert 'ci_gh workflow enable btc_24h_autonomous_research.yml --repo "$REPO"' in text
    assert '24H marathon workflow enable check completed.' in text
    assert 'BTC 24H marathon dispatch verified:' in text

def test_24h_watchdog_cancels_obsolete_main_sha_runs_immediately():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert 'if r.get("head_sha") != main_sha:' in workflow
    assert 'stale_active.append((r, age_minutes, "obsolete_main_sha"))' in workflow
    assert 'ACTION=RECOVER_STALE' in workflow

def test_24h_watchdog_wakes_on_recovery_chain_changes():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "  push:" in workflow
    assert "      - '.github/workflows/btc_24h_watchdog.yml'" in workflow
    assert "      - '.github/workflows/btc_24h_autonomous_research.yml'" in workflow

def test_24h_watchdog_wakes_on_policy_source_changes():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "      - 'PROJECT_INSTRUCTIONS.md'" in workflow
    assert "      - 'docs/PROJECT_SOURCE.md'" in workflow
