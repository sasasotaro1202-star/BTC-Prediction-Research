from pathlib import Path

WORKFLOW = Path('.github/workflows/btc_24h_autonomous_research.yml')

def test_24h_has_success_only_stage_checkpoints():
    workflow = WORKFLOW.read_text(encoding='utf-8')
    for stage in ('stage1_success_checkpoint.json', 'stage2_success_checkpoint.json', 'stage3_success_checkpoint.json', 'stage4_success_checkpoint.json'):
        assert stage in workflow
    assert workflow.count('if: success()') >= 4

def test_24h_reconcile_requires_all_success_checkpoints():
    workflow = WORKFLOW.read_text(encoding='utf-8')
    for stage in ('stage1_success_checkpoint.json', 'stage2_success_checkpoint.json', 'stage3_success_checkpoint.json', 'stage4_success_checkpoint.json'):
        assert f'"{stage}",' in workflow
    assert 'status = "COMPLETED" if (' in workflow
    assert 'all(v == "success" for v in stages.values())' in workflow
    assert 'and not missing' in workflow
    assert 'and not checkpoint_integrity_failures' in workflow
    assert 'and not safety_failures' in workflow
    assert 'else "PARTIAL_OR_BLOCKED"' in workflow

def test_24h_success_checkpoints_bind_actual_run_metadata():
    workflow = WORKFLOW.read_text(encoding='utf-8')
    assert workflow.count('RUN_ID: ${{ github.run_id }}') == 4
    assert workflow.count('SHA: ${{ github.sha }}') == 4
    assert workflow.count('"run_id": int(os.environ["RUN_ID"])') == 4
    assert workflow.count('"sha": os.environ["SHA"]') == 4
    assert "printf '%s\n' '{" not in workflow

def test_24h_checkpoint_creation_paths_match_reconciliation_contract():
    workflow = WORKFLOW.read_text(encoding='utf-8')
    for name in (
        'stage1_success_checkpoint.json',
        'stage2_success_checkpoint.json',
        'stage3_success_checkpoint.json',
        'stage4_success_checkpoint.json',
    ):
        assert f'Path("data/historical_research/{name}")' in workflow
        assert f'"{name}",' in workflow

def test_24h_removes_stale_research_outputs_before_each_stage():
    workflow = WORKFLOW.read_text(encoding='utf-8')
    for name in (
        'maximum_future_generalization_v6_registry.json',
        'report.json',
        'rolling_challenger_oos.json',
        'adaptive_ensemble_oos.json',
        'calibration_frozen_replay_oos.json',
        'uncertainty_layer_oos.json',
        'microstructure_oos.json',
    ):
        assert f'rm -f data/historical_research/{name}' in workflow


def test_24h_reconcile_validates_checkpoint_identity():
    workflow = WORKFLOW.read_text(encoding='utf-8')
    assert 'checkpoint_integrity_failures' in workflow
    assert 'current_run_id = int(os.environ["GITHUB_RUN_ID"])' in workflow
    assert 'current_sha = os.environ["GITHUB_SHA"]' in workflow
    assert 'obj.get("status") != "SUCCESS"' in workflow
    assert 'obj.get("research_only") is not True' in workflow
    assert 'obj.get("production_changed") is not False' in workflow
    assert 'obj.get("sha") != current_sha' in workflow

def test_24h_reconcile_binds_checkpoint_stage_to_filename():
    workflow = WORKFLOW.read_text(encoding='utf-8')
    assert '"stage1_success_checkpoint.json": "stage1_maximum"' in workflow
    assert '"stage2_success_checkpoint.json": "stage2_historical"' in workflow
    assert '"stage3_success_checkpoint.json": "stage3_challenger"' in workflow
    assert '"stage4_success_checkpoint.json": "stage4_robustness"' in workflow
    assert 'obj.get("stage") != expected_stage' in workflow

def test_24h_stage1_uses_actual_maximum_future_generalization_registry():
    workflow = WORKFLOW.read_text(encoding='utf-8')
    assert 'maximum_future_generalization_v6_registry.json' in workflow
    assert 'test -s data/historical_research/maximum_future_generalization_v6_registry.json' in workflow
    assert 'test -s data/historical_research/maximum_future_generalization_v6.json' not in workflow


def test_24h_stage3_runtime_budget_matches_standalone_lane():
    workflow = WORKFLOW.read_text(encoding='utf-8')
    assert '220m python src/rolling_challenger_oos.py' in workflow
    assert '100m python src/adaptive_ensemble_oos.py' in workflow
    assert '180m python src/rolling_challenger_oos.py' not in workflow
    assert '140m python src/adaptive_ensemble_oos.py' not in workflow
    assert "ROLLING_MAX_ROWS: '3000'" in workflow
    assert "ROLLING_TEST_BLOCK: '100'" in workflow

def test_24h_stage_results_are_enforced_by_finalize_job():
    workflow = WORKFLOW.read_text(encoding='utf-8')
    assert 'Enforce complete marathon evidence' in workflow
    for result_var in ('S1', 'S2', 'S3', 'S4'):
        assert f'test "${result_var}" = success' in workflow

def test_24h_rolling_config_is_recorded_by_research_script():
    script = Path('src/rolling_challenger_oos.py').read_text(encoding='utf-8')
    assert 'ROLLING_MAX_ROWS' in script
    assert 'ROLLING_TEST_BLOCK' in script
    assert '"evaluation_config"' in script
    assert '"max_rows": MAX_ROWS' in script
    assert '"test_block": TEST_BLOCK' in script

def test_24h_stage4_does_not_self_reference_its_own_needs_result():
    workflow = WORKFLOW.read_text(encoding='utf-8')
    stage4_block = workflow.split('\n  finalize:', 1)[0].split('\n  stage4_robustness:', 1)[1]
    assert 'needs.stage4_robustness.result' not in stage4_block


def test_24h_stage1_rejects_superseded_workflow_sha_before_expensive_research():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "Verify workflow SHA is current main before expensive research" in workflow
    assert "ci_git_fetch origin main --depth=1" in workflow
    assert "STALE_WORKFLOW_SHA current_run=$GITHUB_SHA current_main=$main_sha" in workflow
    assert "WORKFLOW_SHA_CURRENT_MAIN=$GITHUB_SHA" in workflow

def test_24h_all_jobs_fail_closed_when_main_moves():
    workflow = WORKFLOW.read_text(encoding='utf-8')
    assert workflow.count('name: Fail closed if run is not latest main') == 5
    assert workflow.count('git/ref/heads/main') == 5
    assert workflow.count('STALE_MAIN_RUN expected=') == 5

def test_actions_cleanup_cancels_stale_24h_research_runs():
    cleanup = Path('.github/workflows/btc_actions_cleanup.yml').read_text(encoding='utf-8')
    assert 'btc_24h_autonomous_research.yml' in cleanup
    assert '[ "${head}" != "${main_sha}" ]' in cleanup
    assert '[ "${age}" -ge 1800 ]' in cleanup
