from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _workflow(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def test_continuous_supervisor_retries_failed_generations_after_bounded_cooldown():
    text = _workflow("btc_continuous_supervisor.yml")
    assert "latest_conclusion" in text
    assert "failure|cancelled|timed_out|action_required|stale" in text
    assert 'if [ "${age}" -ge "${failure_cooldown}" ]; then failure_retry_due=true; fi' in text
    assert '[ "${failure_retry_due}" = true ]' in text


def test_watchdog_recovers_failed_generations_on_current_main():
    text = _workflow("btc_watchdog.yml")
    assert "latest_conclusion" in text
    assert "latest_head_sha" in text
    assert "latest_attempt" in text
    assert "failure|cancelled|timed_out|action_required|stale" in text
    assert 'if [ "$age" -ge "$failure_cooldown" ]; then failure_retry_due=true; fi' in text
    assert '[ "$failure_retry_due" = true ]' in text


def test_24h_research_recovery_is_owned_by_dedicated_watchdog():
    supervisor = _workflow("btc_continuous_supervisor.yml")
    watchdog = _workflow("btc_watchdog.yml")
    dedicated = _workflow("btc_24h_watchdog.yml")

    assert '"BTC 24H Autonomous Research"' in supervisor
    assert '"BTC 24H Research Watchdog"' in supervisor
    assert "dispatch_if_stale btc_24h_autonomous_research.yml 86400" not in supervisor
    assert "btc_24h_watchdog.yml" in watchdog
    assert "recover_if_stale btc_24h_watchdog.yml 900 1200 300" in watchdog
    assert "btc_24h_autonomous_research.yml" in dedicated
    assert "FAILURE_CIRCUIT_BREAKER" in dedicated
    assert "ACTION=HOLD" in dedicated


def test_return_tail_lane_has_fail_closed_current_main_and_research_only_boundaries():
    text = _workflow("btc_return_distribution_tail_oos.yml")
    assert "Bind immutable analysis snapshot" in text
    assert "RETURN_TAIL_WORKFLOW_MUST_RUN_FROM_MAIN" in text
    assert "Current main may advance while this immutable research snapshot runs" in text
    assert 'obj.get("research_only") is not True' in text
    assert 'obj.get("production_changed") is not False' in text
    assert 'obj.get("analysis_git_sha") != __import__("os").environ.get("GITHUB_SHA")' in text
    assert 'obj.get("prediction_db_sha256")' in text
    assert 'result.get("promotion_evidence_eligible") is True' in text

def test_watchdog_checks_out_repository_before_using_versioned_recovery_helpers():
    text = _workflow("btc_watchdog.yml")
    assert "actions/checkout@v7" in text
    assert "scripts/ci_failure_streak.sh" in text

def test_watchdog_runs_immediately_when_its_recovery_workflow_changes():
    text = _workflow("btc_watchdog.yml")
    assert "  push:" in text
    assert "      - '.github/workflows/btc_watchdog.yml'" in text
def test_research_workflow_keeps_runtime_production_audit_out_of_git_state():
    text = _workflow("btc_research.yml")
    assert "data/historical_research/production_artifact_audit.json" in text
    assert "actions/upload-artifact@v6" in text
    assert "git add -f data/historical_research/production_artifact_audit.json" not in text
    for line in text.splitlines():
        if line.strip().startswith("git add "):
            assert "production_artifact_audit.json" not in line
    assert "/tmp/btc_research_artifact_audit.json" not in text
def test_experience_policy_allows_state_only_main_drift_but_rejects_code_drift():
    text = _workflow("btc_experience_policy_oos.yml")
    assert "Validate scheduled research snapshot against current main" in text
    assert 'compare/${GITHUB_SHA}...${remote_sha}' in text
    assert 'compare_status' in text
    assert "changed_file_count=\"$(jq '.files | length' <<<\"${compare_json}\")\"" in text
    assert 'compare_file_limit_reached' in text
    assert 'jq -r ".files[]?.filename // empty"' in text
    assert "data/*|models/*.json|models/*.joblib)" in text
    assert 'STALE_MAIN_RUN incompatible_change=' in text
    assert "STATE_ONLY_MAIN_DRIFT allowed_between_snapshot_and_main:" in text
    assert "git add" not in text


def test_research_pr_automerge_does_not_cancel_inflight_merge_checks():
    text = _workflow("btc_research_pr_automerge.yml")
    assert "group: btc-research-pr-automerge" in text
    assert "cancel-in-progress: false" in text


def test_binance_flow_collector_143_requires_fresh_advancing_checkpoint():
    text = _workflow("btc_binance_flow_research.yml")
    assert 'capture_started_at_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"' in text
    assert "initial_latest=" in text
    assert "rows[-1].get('end_time_ms'" in text
    assert "updated_at_utc" in text
    assert 'if [ "$collector_status" -eq 143 ]; then' in text
    assert "latest > initial_latest" in text
    assert 'fresh advancing checkpoint' in text
    assert 'exited with 143 without a fresh advancing checkpoint' in text
