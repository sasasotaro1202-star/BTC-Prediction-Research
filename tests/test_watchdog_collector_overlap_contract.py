from pathlib import Path


def test_watchdog_staggers_binance_collectors_after_30_minutes():
    workflow = Path(".github/workflows/btc_watchdog.yml").read_text(encoding="utf-8")
    assert "in_progress_active_count=0" in workflow
    assert "oldest_in_progress_age=0" in workflow
    assert 'active_count" -eq 1' in workflow
    assert 'in_progress_active_count" -eq 1' in workflow
    assert 'oldest_in_progress_age" -ge 1800' in workflow
    assert "staggered overlap collector" in workflow
def test_collector_active_grace_matches_full_capture_window():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    collector = Path(".github/workflows/btc_binance_ws_collector.yml").read_text(encoding="utf-8")
    assert "timeout-minutes: 90" in collector
    assert "recover_if_stale btc_binance_ws_collector.yml 3900 300 300" in workflow
    assert "recover_if_stale btc_binance_ws_collector.yml 600 3900 300" not in workflow
