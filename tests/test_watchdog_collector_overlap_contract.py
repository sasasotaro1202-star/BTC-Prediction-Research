from pathlib import Path

def test_watchdog_serializes_binance_collector_recovery():
    workflow = Path(".github/workflows/btc_watchdog.yml").read_text(encoding="utf-8")
    collector = Path(".github/workflows/btc_binance_ws_collector.yml").read_text(encoding="utf-8")
    assert "recover_if_stale btc_binance_ws_collector.yml 3000 3300 300" in workflow
    assert "staggered overlap collector" not in workflow
    assert "group: btc-binance-ws-cache" in collector
    assert "cancel-in-progress: false" in collector

def test_collector_active_grace_matches_full_capture_window():
    workflow = Path(".github/workflows/btc_watchdog.yml").read_text(encoding="utf-8")
    collector = Path(".github/workflows/btc_binance_ws_collector.yml").read_text(encoding="utf-8")
    assert "timeout-minutes: 55" in collector
    assert "recover_if_stale btc_binance_ws_collector.yml 3000 3300 300" in workflow
    assert "recover_if_stale btc_binance_ws_collector.yml 3000 300 300" not in workflow
    assert "recover_if_stale btc_binance_ws_collector.yml 3900 300 300" not in workflow


def test_stale_janitor_collapses_existing_collector_duplicates():
    janitor = Path('.github/workflows/btc_stale_run_janitor.yml').read_text(encoding='utf-8')
    assert 'collector_runs=' in janitor
    assert 'collapsing to newest active generation' in janitor
    assert '.[1:][]?.id' in janitor


def test_janitor_self_validation_trigger():
    janitor = Path('.github/workflows/btc_stale_run_janitor.yml').read_text(encoding='utf-8')
    assert 'push:' in janitor
    assert 'branches: [main]' in janitor
    assert "'.github/workflows/btc_stale_run_janitor.yml'" in janitor



def test_watchdog_self_change_push_can_recover_live_but_not_research():
    workflow = Path(".github/workflows/btc_watchdog.yml").read_text(encoding="utf-8")
    assert "push)" in workflow
    assert 'if [ "$workflow" != "btc_live_cycle.yml" ]; then' in workflow
    assert "dispatch_allowed=false" in workflow
