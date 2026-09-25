from pathlib import Path


def test_supervisor_uses_rest_dispatch_first():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert 'actions/workflows/${workflow}/dispatches' in text
    assert 'Dispatch succeeded via workflow_dispatch API' in text


def test_supervisor_degrades_explicitly_on_dispatch_failure():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert 'degraded=1' in text
    assert 'ERROR: dispatch failed through both REST and gh CLI' in text
