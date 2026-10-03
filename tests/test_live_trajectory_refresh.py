from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "btc_live_cycle.yml"


def test_live_trajectory_is_refreshed_after_prediction_generation():
    text = WORKFLOW.read_text(encoding="utf-8")
    prediction = text.index("      - name: Generate next BTC prediction")
    refresh = text.index("      - name: Refresh chart-ready BTC trajectory after prediction")
    fail_closed = text.index("      - name: Fail closed on incomplete live prediction inputs")
    assert prediction < refresh < fail_closed

    end = text.index("      - name:", refresh + 10)
    block = text[refresh:end]
    assert "python src/forecast_trajectory.py --limit 2880" in block
    assert "latest_prediction_id" in block
    assert "newest != latest_id" in block
    assert "post-prediction trajectory lag detected" in block
