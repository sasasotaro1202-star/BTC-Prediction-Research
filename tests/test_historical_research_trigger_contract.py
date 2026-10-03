from pathlib import Path

WORKFLOW = Path(".github/workflows/btc_historical_research.yml")


def test_test_only_changes_do_not_trigger_expensive_historical_research():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "tests/test_historical_research.py" not in workflow
    assert "tests/test_feature_pattern_exhaustive.py" not in workflow
