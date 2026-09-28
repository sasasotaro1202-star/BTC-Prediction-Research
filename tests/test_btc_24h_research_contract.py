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
    assert 'status = "COMPLETED" if all(v == "success" for v in stages.values()) and not missing and not safety_failures else "PARTIAL_OR_BLOCKED"' in workflow

def test_24h_success_checkpoints_bind_actual_run_metadata():
    workflow = WORKFLOW.read_text(encoding='utf-8')
    assert workflow.count('RUN_ID: ${{ github.run_id }}') == 4
    assert workflow.count('SHA: ${{ github.sha }}') == 4
    assert workflow.count('"run_id": int(os.environ["RUN_ID"])') == 4
    assert workflow.count('"sha": os.environ["SHA"]') == 4
    assert "printf '%s\n' '{" not in workflow
