from pathlib import Path

WORKFLOW = Path('.github/workflows/btc_pattern_matrix_research.yml')


def test_pattern_matrix_persists_current_production_audit():
    workflow = WORKFLOW.read_text(encoding='utf-8')
    assert 'data/historical_research/production_artifact_audit.json' in workflow
    assert 'git add data/historical_research/pattern_matrix_research.json data/historical_research/production_artifact_audit.json' in workflow


def test_pattern_matrix_uploads_both_research_and_production_audit_evidence():
    workflow = WORKFLOW.read_text(encoding='utf-8')
    assert 'name: btc-pattern-matrix-${{ github.sha }}' in workflow
    block = workflow.split('      - name: Upload research evidence', 1)[1].split('      - name: Persist evidence', 1)[0]
    assert 'data/historical_research/pattern_matrix_research.json' in block
    assert 'data/historical_research/production_artifact_audit.json' in block