from pathlib import Path

WORKFLOW = Path(".github/workflows/btc_research.yml")


def test_research_state_commit_does_not_stage_ignored_runtime_audit():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "git add data/predictions.db.gz models/*.json models/*.joblib" in text
    assert "git add data/predictions.db.gz data/historical_research/production_artifact_audit.json" not in text
    assert ".gitignore" not in text
