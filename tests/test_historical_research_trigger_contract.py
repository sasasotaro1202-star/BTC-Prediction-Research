from pathlib import Path

WORKFLOW = Path(".github/workflows/btc_historical_research.yml")


def test_test_only_and_workflow_only_changes_do_not_trigger_expensive_historical_research():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert ".github/workflows/btc_historical_research.yml" not in workflow
    assert "tests/test_historical_research.py" not in workflow
    assert "tests/test_feature_pattern_exhaustive.py" not in workflow


def test_research_trigger_still_includes_research_sources_and_workflow():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    for path in (
        "src/historical_research.py",
        "src/historical_research_runner.py",
        "src/label_policy.py",
        "src/http_resilience.py",
        "scripts/historical_research_spot_fallback.py",
        "src/feature_pattern_exhaustive.py",
    ):
        assert path in workflow
