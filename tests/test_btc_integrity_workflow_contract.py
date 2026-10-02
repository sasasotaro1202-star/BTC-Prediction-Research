from pathlib import Path

WORKFLOW = Path(".github/workflows/btc_integrity.yml")


def test_integrity_job_timeout_has_observed_headroom():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "timeout-minutes: 30" in workflow
    assert "timeout-minutes: 15" not in workflow


def test_archive_integrity_does_not_gate_on_repository_snapshot_freshness():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "BTC_INTEGRITY_REQUIRE_FRESH: 'false'" in workflow
    assert "production_freshness_observation.json" in workflow
    assert "freshness_policy" in workflow
    assert "Production freshness is enforced by Live Cycle/Sentinel" in workflow
