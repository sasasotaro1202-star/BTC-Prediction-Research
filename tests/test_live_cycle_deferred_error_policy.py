from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "btc_live_cycle.yml"


def test_stale_or_future_market_event_is_deferred():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'grep -Eq "stale_live_market_event:|future_live_market_event:" "$err"' in text


def test_generic_prediction_failures_remain_red():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'rm -f "$out" "$err"\n          exit 1' in text
