from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_watchdog_never_source_cancels_rolling_collector():
    watchdog = (ROOT / ".github" / "workflows" / "btc_watchdog.yml").read_text(encoding="utf-8")
    guarded = 'if [ "$workflow" != "btc_binance_ws_collector.yml" ] && [ "$active_age" -ge "$stale_head_grace" ]'
    assert guarded in watchdog


def test_collector_schedule_does_not_cancel_existing_stream():
    workflow = (ROOT / ".github" / "workflows" / "btc_binance_ws_collector.yml").read_text(encoding="utf-8")
    assert "cancel-in-progress: ${{ github.event_name != 'schedule' }}" in workflow
    assert "capture_seconds=2700" in workflow
    assert "checkpoint_seconds=60" in workflow