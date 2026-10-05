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
