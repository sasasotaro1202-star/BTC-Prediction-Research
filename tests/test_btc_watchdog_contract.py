from pathlib import Path


WORKFLOW = Path(".github/workflows/btc_watchdog.yml")


def test_integrity_watchdog_age_matches_integrity_job_timeout():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "timeout-minutes: 30" in Path(".github/workflows/btc_integrity.yml").read_text(encoding="utf-8")
    assert "recover_if_stale btc_integrity.yml 3600 1800 900" in workflow
    assert "recover_if_stale btc_integrity.yml 3600 1200 900" not in workflow
