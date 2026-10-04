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
        'btc_selective_prediction_oos.yml',
    ):
        assert f'dispatch_if_stale {workflow}' in text


def test_supervisor_routes_one_additional_evidence_driven_lane():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert 'python src/autonomous_research_router.py' in text
    assert 'autonomous_route.json' in text
    assert 'Dispatched routed research candidate:' in text
    for workflow in (
        'btc_research_readiness.yml',
        'btc_autonomous_data_frontier.yml',
        'btc_adaptive_calibration_replay.yml',
        'btc_experience_policy_oos.yml',
        'btc_rich_production_challenger.yml',
        'btc_ultimate_final_v13_e2e.yml',
    ):
        assert workflow in text


def test_supervisor_does_not_use_fail_open_shell_suppression():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert '| true' not in text


def test_supervisor_fail_closes_router_failure_to_readiness():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert "route_workflow='btc_research_readiness.yml'" in text
    assert "route_reason='router_failed_fail_closed'" in text




def test_ops_preflight_triggers_on_supervisor_and_router_changes():
    text = Path('.github/workflows/btc_ops_preflight.yml').read_text(encoding='utf-8')
    for path in (
        ".github/workflows/btc_continuous_supervisor.yml",
        "src/autonomous_research_router.py",
        "tests/test_continuous_supervisor.py",
        "tests/test_autonomous_research_router.py",
    ):
        assert f"      - '{path}'" in text



def test_ops_preflight_validates_autonomous_router_output():
    text = Path('.github/workflows/btc_ops_preflight.yml').read_text(encoding='utf-8')
    assert 'python src/autonomous_research_router.py > /tmp/autonomous_route.json' in text
    assert 'router produced workflow outside allowlist' in text
    assert 'router production_impact must be false' in text
    assert 'router threshold is outside safe bounds' in text


def test_supervisor_publishes_route_priority_and_evidence_signals():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert '"route_workflow":' in text
    assert '"route_threshold_seconds":' in text
    assert '"route_priority":' in text
    assert '"route_reason":' in text
    assert '"route_evidence_state":' in text
    assert '"route_signals":' in text


def test_supervisor_routes_through_ordered_candidate_list():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert 'route_candidate_count="$(jq' in text
    assert 'for ((i=0; i<route_candidate_count; i++)); do' in text
    assert 'dispatch_if_stale "${candidate_workflow}" "${candidate_threshold}" 1' in text
    assert 'if [ "${candidate_rc}" -eq 10 ]; then' in text
    assert 'non-stale lanes remain eligible for a later heartbeat' in text


def test_supervisor_preserves_readiness_candidate_on_router_failure():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert '"reason": "router_failed_fail_closed"' in text
    assert '"candidates": [' in text
    assert '"workflow": "btc_research_readiness.yml"' in text


def test_supervisor_route_errors_are_explicitly_degraded():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert 'if [ "${route_mode}" -eq 1 ]; then return 20; fi' in text
    assert 'elif [ "${candidate_rc}" -eq 20 ]; then' in text



def test_frontier_push_trigger_excludes_test_only_changes():
    text = Path('.github/workflows/btc_autonomous_data_frontier.yml').read_text(encoding='utf-8')
    trigger = text.split('permissions:', 1)[0]
    assert 'src/btc_data_frontier_autoselect.py' in trigger
    assert 'tests/test_btc_data_frontier_autoselect.py' not in trigger
    assert '.github/workflows/btc_autonomous_data_frontier.yml' not in trigger


def test_supervisor_has_event_driven_recovery_triggers():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    for workflow in (
        'BTC Live Cycle',
        'BTC Binance WS Collector',
        'BTC PIT OOS Audit',
        'BTC Archive Refresh Research',
        'BTC Research Readiness Audit',
        'BTC Adaptive Calibration Replay Research',
        'BTC Experience Policy OOS Learning',
        'BTC Selective Prediction OOS',
        'BTC Rich Production Challenger',
        'BTC Ultimate Final V13 — Maximum Future-Generalization E2E',
        'BTC Autonomous Data Frontier',
        'BTC Unit Tests',
        'BTC Ops Preflight',
    ):
        assert f'- "{workflow}"' in text
    assert 'types: [completed]' in text


def test_supervisor_records_trigger_event():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert '"trigger_event": "${GITHUB_EVENT_NAME}"' in text
