from pathlib import Path

WORKFLOW = Path(".github/workflows/btc_integrity.yml")


def test_integrity_job_timeout_has_observed_headroom():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "timeout-minutes: 30" in workflow
    assert "timeout-minutes: 15" not in workflow
