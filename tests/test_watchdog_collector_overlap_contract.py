from pathlib import Path

def test_watchdog_serializes_binance_collector_recovery():
    workflow = Path(".github/workflows/btc_watchdog.yml").read_text(encoding="utf-8")
    collector = Path(".github/workflows/btc_binance_ws_collector.yml").read_text(encoding="utf-8")
    assert "recover_if_stale btc_binance_ws_collector.yml 3000 300 300" in workflow
    assert "staggered overlap collector" not in workflow
    assert "group: btc-binance-ws-cache" in collector
    assert "cancel-in-progress: false" in collector

def test_collector_active_grace_matches_full_capture_window():
    workflow = Path(".github/workflows/btc_watchdog.yml").read_text(encoding="utf-8")
    collector = Path(".github/workflows/btc_binance_ws_collector.yml").read_text(encoding="utf-8")
    assert "timeout-minutes: 55" in collector
    assert "recover_if_stale btc_binance_ws_collector.yml 3000 300 300" in workflow
    assert "recover_if_stale btc_binance_ws_collector.yml 3900 300 300" not in workflow
