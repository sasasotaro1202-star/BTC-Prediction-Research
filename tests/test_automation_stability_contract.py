from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / '.github' / 'workflows' / name).read_text(encoding='utf-8')


def test_supervisor_does_not_wake_on_unit_test_completion():
    text = _read('btc_continuous_supervisor.yml')
    assert '- "BTC Unit Tests"' not in text


def test_supervisor_failure_streak_stops_at_latest_success():
    text = _read('btc_continuous_supervisor.yml')
    assert 'ci_failure_streak_from_stdin' in text


def test_watchdog_failure_streak_stops_at_latest_success():
    text = _read('btc_watchdog.yml')
    assert 'ci_failure_streak_from_stdin' in text


def test_stale_janitor_is_not_a_dispatcher():
    text = _read('btc_stale_run_janitor.yml')
    assert 'dispatch_if_stale()' not in text
    assert 'recovery dispatch is centralized' in text
