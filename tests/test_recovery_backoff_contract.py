from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / '.github' / 'workflows' / name).read_text(encoding='utf-8')


def test_supervisor_has_bounded_failure_backoff():
    text = _read('btc_continuous_supervisor.yml')
    assert 'failure_streak' in text
    assert 'failure_cooldown' in text
    assert '2 ** backoff_power' in text
    assert '[ "${failure_cooldown}" -gt 4800 ] && failure_cooldown=4800' in text


def test_watchdog_has_bounded_failure_backoff():
    text = _read('btc_watchdog.yml')
    assert 'failure_streak' in text
    assert 'failure_cooldown' in text
    assert '2 ** backoff_power' in text
    assert '[ "$failure_cooldown" -gt 4800 ] && failure_cooldown=4800' in text


def test_stale_janitor_is_not_a_second_dispatcher():
    text = _read('btc_stale_run_janitor.yml')
    assert 'dispatch_if_stale()' not in text
    assert 'cancellation-only' in text
