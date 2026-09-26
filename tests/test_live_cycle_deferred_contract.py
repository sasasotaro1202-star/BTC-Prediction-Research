from pathlib import Path


def test_deferred_live_cycle_does_not_fail_on_missing_pit_audit():
    text = Path(".github/workflows/btc_live_cycle.yml").read_text(encoding="utf-8")
    assert 'if [ "${BTC_LIVE_DEFERRED:-0}" = "1" ]; then' in text
    assert "PIT/OOS audit skipped because live prediction was safely DEFERRED" in text
    assert 'raise SystemExit("PIT audit output missing; refusing to record history")' in text

def test_live_cycle_runs_pit_audit_before_recording_history():
    text = Path(".github/workflows/btc_live_cycle.yml").read_text(encoding="utf-8")
    audit = text.index("      - name: Run strict PIT/OOS audit")
    record = text.index("      - name: Record bounded PIT audit history")
    block = text[audit:record]
    assert audit < record
    assert "python src/pit_oos_audit.py" in block
    assert "test -s data/historical_research/pit_oos_audit.json" in block
