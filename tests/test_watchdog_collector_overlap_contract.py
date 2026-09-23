from pathlib import Path


def test_watchdog_staggers_binance_collectors_after_30_minutes():
    workflow = Path(".github/workflows/btc_watchdog.yml").read_text(encoding="utf-8")
    assert "in_progress_active_count=0" in workflow
    assert "oldest_in_progress_age=0" in workflow
    assert 'active_count" -eq 1' in workflow
    assert 'in_progress_active_count" -eq 1' in workflow
    assert 'oldest_in_progress_age" -ge 1800' in workflow
    assert "staggered overlap collector" in workflow
