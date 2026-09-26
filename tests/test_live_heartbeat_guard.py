from pathlib import Path


def test_live_heartbeat_guard_uses_rest_dispatch():
    text = Path('.github/workflows/btc_live_heartbeat_guard.yml').read_text(encoding='utf-8')
    assert 'actions/workflows/btc_live_cycle.yml/dispatches' in text
    assert 'Live recovery dispatch succeeded via REST' in text


def test_live_heartbeat_guard_does_not_dispatch_when_active():
    text = Path('.github/workflows/btc_live_heartbeat_guard.yml').read_text(encoding='utf-8')
    assert 'if [ "$active" -gt 0 ]; then' in text
    assert 'Live already active/queued; no dispatch.' in text

def test_watchdog_uses_heartbeat_age_not_execution_head_for_staleness():
    text = Path('.github/workflows/btc_watchdog.yml').read_text(encoding='utf-8')
    old = 'if [ "$status_epoch" -gt 0 ] && [ "$status_age" -ge 300 ] && [ "$status_head" != "$main_sha" ]; then'
    assert old not in text
    assert 'if [ "$status_epoch" -gt 0 ] && [ "$status_age" -ge 300 ]; then' in text
    assert 'expected_after_state_merge' in text