from pathlib import Path


def test_supervisor_uses_rest_dispatch_first():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert 'actions/workflows/${workflow}/dispatches' in text
    assert 'Dispatch succeeded via workflow_dispatch API' in text


def test_supervisor_degrades_explicitly_on_dispatch_failure():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert 'degraded=1' in text
    assert 'ERROR: dispatch failed through both REST and gh CLI' in text


def test_supervisor_monitors_case_adaptation_research_lanes():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    for workflow in (
        'btc_uncertainty_layer_oos.yml',
        'btc_multiscale_frozen_replay.yml',
        'btc_time_regime_research.yml',
        'btc_adaptive_calibration_replay.yml',
        'btc_selective_prediction_oos.yml',
    ):
        assert f'dispatch_if_stale {workflow}' in text
