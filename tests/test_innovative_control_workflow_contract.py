from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github' / 'workflows' / 'btc_innovative_prediction_control_v2.yml'


def test_research_gate_artifact_writes_real_newline_not_literal_backslash_n():
    text = WORKFLOW.read_text(encoding='utf-8')
    assert "json.dumps(payload,indent=2,sort_keys=True)+'\\\\n'" not in text
    assert "json.dumps(payload,indent=2,sort_keys=True)+'\\n'" in text


def test_v2_run_start_journal_is_local_only_until_completion():
    s=WORKFLOW.read_text(encoding='utf-8')
    start=s.index("Initialize v2 run journal (no main write)")
    compile_step=s.index("Compile and unit test", start)
    segment=s[start:compile_step]
    assert "git add data/historical_research/innovative_control_v2_run.json" not in segment
    assert "git push origin HEAD:main" not in segment
    assert "git commit -m 'Record BTC v2 research run start'" not in segment
    completion=s.index("Record v2 run completion")
    completion_segment=s[completion:]
    assert "Record BTC v2 research run completion" in completion_segment
    assert "git push origin HEAD:main" in completion_segment
