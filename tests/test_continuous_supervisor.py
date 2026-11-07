from pathlib import Path


def test_supervisor_uses_rest_dispatch_first():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert 'actions/workflows/${workflow}/dispatches' in text
    assert 'Dispatch request accepted via workflow_dispatch API' in text


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
        'btc_uncertainty_layer_oos.yml',
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
        'btc_external_method_research.yml',
        'btc_external_method_shadow.yml',
        'btc_ultimate_final_v13_e2e.yml',
        'btc_recency_challenger.yml',
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
    assert 'if [ "${route_dispatch_outcome}" -eq 1 ]; then' in text
    assert 'candidate_rc=$?' in text
    assert 'if [ "${candidate_rc}" -ne 0 ]; then' in text
    assert 'non-stale lanes remain eligible for a later heartbeat' in text


def test_supervisor_preserves_readiness_candidate_on_router_failure():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert '"reason": "router_failed_fail_closed"' in text
    assert '"candidates": [' in text
    assert '"workflow": "btc_research_readiness.yml"' in text


def test_supervisor_route_errors_are_explicitly_degraded():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert 'route_dispatch_outcome=2' in text
    assert 'route_dispatch_outcome=1' in text
    assert 'return 10' not in text
    assert 'return 20' not in text
    assert 'elif [ "${route_dispatch_outcome}" -eq 2 ]; then' in text



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
        'BTC Ops Preflight',
        'BTC 24H Autonomous Research',
        'BTC 24H Research Watchdog',
        'BTC Recency Challenger',
    ):
        assert f'- "{workflow}"' in text
    assert 'types: [completed]' in text


def test_supervisor_records_trigger_event():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert '"trigger_event": "${GITHUB_EVENT_NAME}"' in text


def test_supervisor_prioritizes_production_capacity_over_stale_research_dispatch():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert 'production_active=0' in text
    assert 'production_response="$(ci_gh_api_get' in text
    assert 'if [ "${production_active}" -eq 0 ]; then' in text
    assert 'suppressing avoidable research stale-dispatches' in text

def test_supervisor_reenables_intended_workflow_before_recovery_dispatch():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert 'ci_gh workflow enable "${workflow}" --repo "${REPO}"' in text
    assert 'Workflow enable check completed: ${workflow}' in text


def test_supervisor_verifies_dispatch_created_on_target_sha():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert 'dispatch_target_sha="$(ci_gh_api_get "/repos/${REPO}/git/ref/heads/main"' in text
    assert 'Dispatch creation verified: ${workflow} sha=${dispatch_target_sha}' in text
    assert 'no matching workflow_dispatch run was created within the verification window' in text
    assert 'select(.event=="workflow_dispatch")' in text

def test_supervisor_recovers_only_the_24h_watchdog_not_the_marathon():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert 'dispatch_if_stale btc_24h_watchdog.yml 900' in text
    assert 'dispatch_if_stale btc_24h_autonomous_research.yml' not in text
    assert '24H marathon remains exclusively owned by its dedicated watchdog.' in text

def test_supervisor_initializes_route_candidate_count_before_backpressure_branch():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    production_backpressure_guard = '          if [ "${production_active}" -eq 0 ]; then\n            # Try ordered candidates'
    initialization = '          route_candidate_count=0'
    marker = '          route_dispatched=0'
    assert initialization in text
    assert production_backpressure_guard in text
    assert text.index(initialization) > text.index(marker)
    assert text.index(initialization) < text.index(production_backpressure_guard)


def test_supervisor_defers_routed_research_when_production_spine_is_active():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert 'route_deferred_by_production=0' in text
    assert 'route_deferred_by_production=1' in text
    assert 'Production spine active; deferring evidence-driven routed research until a later heartbeat.' in text
    assert '"route_deferred_by_production": ${route_deferred_by_production}' in text


def test_supervisor_and_preflight_accept_all_allowlisted_router_candidates():
    supervisor = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    preflight = Path('.github/workflows/btc_ops_preflight.yml').read_text(encoding='utf-8')
    assert 'length >= 1 and length <= 12' in supervisor
    assert '1 <= len(candidates) <= len(allowed)' in preflight


def test_supervisor_keeps_pit_audit_fresh_during_production_backpressure():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert 'dispatch_if_stale btc_pit_oos_audit.yml 900' in text
    assert text.index('dispatch_if_stale btc_pit_oos_audit.yml 900') < text.index('production_active=0')
    assert 'PIT OOS Audit is a production-safety prerequisite' in text

def test_supervisor_uses_bounded_60_second_dispatch_verification_window():
    text = Path('.github/workflows/btc_continuous_supervisor.yml').read_text(encoding='utf-8')
    assert 'for check in 1 2 3 4 5 6 7 8 9 10 11 12; do' in text
    assert 'sleep 5' in text
    assert 'bounded 60-second poll' in text

