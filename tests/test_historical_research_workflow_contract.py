from pathlib import Path

WORKFLOW = Path(".github/workflows/btc_research.yml")


def test_conflict_retry_preserves_all_research_json_artifacts():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    for name in (
        "report.json",
        "feature_frontier_oos.json",
        "feature_pattern_exhaustive.json",
    ):
        assert f"cp data/historical_research/{name} /tmp/btc_local_historical_{name}" in workflow
        assert f"cp /tmp/btc_local_historical_{name} data/historical_research/{name}" in workflow


def test_conflict_retry_stages_all_research_json_artifacts():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    expected = (
        "git add data/historical_research/report.json "
        "data/historical_research/feature_frontier_oos.json "
        "data/historical_research/feature_pattern_exhaustive.json "
        "data/historical_research/oos_*.csv"
    )
    assert expected in workflow


def test_runtime_only_production_artifact_audit_is_not_staged_for_git_persistence():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    staging_block = workflow.split("git add data/predictions.db.gz", 1)[1].split("if git ls-files", 1)[0]
    assert "production_artifact_audit.json" not in staging_block
