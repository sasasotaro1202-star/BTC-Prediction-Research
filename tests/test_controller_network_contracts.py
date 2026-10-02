import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _workflow(path: str) -> str:
    return (ROOT / path).read_text(encoding='utf-8')


def test_continuous_supervisor_uses_bounded_github_wrapper():
    text = _workflow('.github/workflows/btc_continuous_supervisor.yml')
    assert '. scripts/ci_network_retry.sh' in text
    assert 'ci_gh_api_get' in text
    assert 'ci_gh workflow run' in text
    assert not re.search(r'(?<![\w-])gh api\s', text)
    assert not re.search(r'(?<![\w-])gh workflow run\s', text)


def test_24h_watchdog_uses_bounded_github_wrapper():
    text = _workflow('.github/workflows/btc_24h_watchdog.yml')
    assert '. scripts/ci_network_retry.sh' in text
    assert 'ci_gh_api_get' in text
    assert 'ci_gh workflow run' in text
    assert 'ci_gh run cancel' in text
    assert not re.search(r'(?<![\w-])gh api\s', text)
    assert not re.search(r'(?<![\w-])gh workflow run\s', text)
    assert not re.search(r'(?<![\w-])gh run cancel\s', text)


def test_controller_job_deadlines_are_finite():
    supervisor = _workflow('.github/workflows/btc_continuous_supervisor.yml')
    watchdog = _workflow('.github/workflows/btc_24h_watchdog.yml')
    assert 'timeout-minutes: 8' in supervisor
    assert 'timeout-minutes: 5' in watchdog
    for text in (supervisor, watchdog):
        assert "CI_GH_ATTEMPTS: '2'" in text
        assert "CI_GH_TIMEOUT_SECONDS: '12'" in text
        assert "CI_GH_BACKOFF_SECONDS: '1'" in text
