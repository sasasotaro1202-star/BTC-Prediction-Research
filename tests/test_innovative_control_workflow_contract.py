from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github' / 'workflows' / 'btc_innovative_prediction_control_v2.yml'


def _run_start_section() -> str:
    text = WORKFLOW.read_text(encoding='utf-8')
    start = text.index("      - name: Record v2 run start")
    end = text.index("      - name: Compile and unit test", start)
    return text[start:end]


def test_research_gate_artifact_writes_real_newline_not_literal_backslash_n():
    text = WORKFLOW.read_text(encoding='utf-8')
    assert "json.dumps(payload,indent=2,sort_keys=True)+'\\\\n'" not in text
    assert "json.dumps(payload,indent=2,sort_keys=True)+'\\n'" in text


def test_v2_running_state_is_workspace_local_and_not_committed_to_main():
    section = _run_start_section()
    assert "'status':'RUNNING'" in section
    assert "git add data/historical_research/innovative_control_v2_run.json" not in section
    assert "git commit -m 'Record BTC v2 research run start'" not in section
    assert "git push origin HEAD:main" not in section
    assert "git rebase origin/main" not in section
