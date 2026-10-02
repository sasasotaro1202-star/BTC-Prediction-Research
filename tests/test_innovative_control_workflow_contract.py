from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github' / 'workflows' / 'btc_innovative_prediction_control_v2.yml'


def test_research_gate_artifact_writes_real_newline_not_literal_backslash_n():
    text = WORKFLOW.read_text(encoding='utf-8')
    assert "json.dumps(payload,indent=2,sort_keys=True)+'\\\\n'" not in text
    assert "json.dumps(payload,indent=2,sort_keys=True)+'\n'" in text
