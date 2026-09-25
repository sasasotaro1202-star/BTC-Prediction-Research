from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "btc_live_cycle.yml"


def test_live_cycle_defers_stale_and_future_market_events():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "stale_live_market_event:" in text
    assert "future_live_market_event:" in text
    assert 'echo "BTC_LIVE_DEFERRED=1"' in text
