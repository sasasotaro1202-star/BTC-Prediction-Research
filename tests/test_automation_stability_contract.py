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
    assert 'Recovery dispatch is centralized' in text


def test_ops_preflight_does_not_wake_on_live_prediction_db_commit():
    text = _read('btc_ops_preflight.yml')
    assert "      - 'data/predictions.db.gz'" not in text


def test_unit_tests_have_periodic_full_suite_safety_net():
    text = _read('btc_unit_tests.yml')
    assert "cron: '47 */6 * * *'" in text


def test_collector_janitor_collapses_duplicate_active_generations():
    janitor = _read('btc_stale_run_janitor.yml')
    assert 'collapsing to newest active generation' in janitor
    assert 'btc_binance_ws_collector.yml/runs' in janitor
    assert '.[1:][]?.id' in janitor


def test_stale_janitor_releases_obsolete_supervisor_generations_promptly():
    janitor = _read('btc_stale_run_janitor.yml')
    assert '[btc_continuous_supervisor.yml]=600' in janitor


def test_supervisor_initializes_route_candidate_count_before_production_gate():
    text = _read('btc_continuous_supervisor.yml')
    init = 'route_candidate_count=0'
    gate = 'if [ "${production_active}" -eq 0 ]; then'
    assert init in text
    assert text.index(init) < text.index(gate)
