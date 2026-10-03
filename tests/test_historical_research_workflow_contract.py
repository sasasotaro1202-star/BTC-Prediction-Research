from pathlib import Path

WORKFLOW = Path(".github/workflows/btc_historical_research.yml")


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
