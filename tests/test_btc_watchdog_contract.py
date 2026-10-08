from pathlib import Path


WORKFLOW = Path(".github/workflows/btc_watchdog.yml")


def test_integrity_watchdog_age_matches_integrity_job_timeout():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "timeout-minutes: 30" in Path(".github/workflows/btc_integrity.yml").read_text(encoding="utf-8")
    assert "recover_if_stale btc_integrity.yml 3600 1800 900" in workflow
    assert "recover_if_stale btc_integrity.yml 3600 1200 900" not in workflow


def test_integrity_watchdog_grace_matches_30m_integrity_timeout():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "timeout-minutes: 30" in Path(".github/workflows/btc_integrity.yml").read_text(encoding="utf-8")
    assert "recover_if_stale btc_integrity.yml 3600 1800 900" in workflow


def test_watchdog_does_not_cancel_itself_on_next_five_minute_tick():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "group: btc-workflow-watchdog" in workflow
    assert "cancel-in-progress: false" in workflow


def test_watchdog_recovers_recency_challenger():
    text = Path('.github/workflows/btc_watchdog.yml').read_text(encoding='utf-8')
    assert 'btc_recency_challenger.yml' in text
    assert 'recover_if_stale btc_recency_challenger.yml 86400 14400 900' in text

def test_watchdog_recovers_pattern_matrix():
    text = Path('.github/workflows/btc_watchdog.yml').read_text(encoding='utf-8')
    assert 'btc_pattern_matrix_research.yml' in text
    assert 'recover_if_stale btc_pattern_matrix_research.yml 86400 7200 300' in text
