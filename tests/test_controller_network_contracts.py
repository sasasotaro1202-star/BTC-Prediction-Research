from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _workflow(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_continuous_supervisor_sources_bounded_network_helper():
    text = _workflow(".github/workflows/btc_continuous_supervisor.yml")
    assert "actions/checkout@v7" in text
    assert ". scripts/ci_network_retry.sh" in text
    assert "ci_gh_api_get" in text
    assert "ci_gh api --method POST" in text
    assert "gh api " not in text
    assert "gh workflow run" not in text


def test_24h_watchdog_sources_bounded_network_helper():
    text = _workflow(".github/workflows/btc_24h_watchdog.yml")
    assert "actions/checkout@v7" in text
    assert ". scripts/ci_network_retry.sh" in text
    assert "ci_gh_api_get" in text
    assert "ci_gh workflow run" in text
    assert "gh api " not in text
    assert "gh workflow run" not in text.replace("ci_gh workflow run", "")
    assert "gh run cancel" not in text


def test_controller_job_deadlines_are_finite():
    supervisor = _workflow(".github/workflows/btc_continuous_supervisor.yml")
    watchdog = _workflow(".github/workflows/btc_24h_watchdog.yml")
    assert "timeout-minutes: 8" in supervisor
    assert "timeout-minutes: 5" in watchdog
    for text in (supervisor, watchdog):
        assert "CI_GH_ATTEMPTS: '2'" in text
        assert "CI_GH_TIMEOUT_SECONDS: '12'" in text
        assert "CI_GH_BACKOFF_SECONDS: '1'" in text
