from pathlib import Path


def test_deferred_live_cycle_refreshes_stale_pit_audit_and_fails_closed_on_missing_output():
    text = Path(".github/workflows/btc_live_cycle.yml").read_text(encoding="utf-8")
    assert 'if [ "${BTC_LIVE_DEFERRED:-0}" = "1" ]; then' in text
    assert "python src/pit_deferred_audit_policy.py data/historical_research/pit_oos_audit.json --max-age-seconds 900" in text
    assert "Live prediction is safely DEFERRED, but the PIT/OOS audit is missing, malformed, future-dated, or older than 15 minutes" in text
    assert 'raise SystemExit("PIT audit output missing; refusing to record history")' in text


def test_live_cycle_runs_pit_audit_before_recording_history():
    text = Path(".github/workflows/btc_live_cycle.yml").read_text(encoding="utf-8")
    audit = text.index("      - name: Run strict PIT/OOS audit")
    record = text.index("      - name: Record bounded PIT audit history")
    block = text[audit:record]
    assert audit < record
    assert "python src/pit_oos_audit.py" in block
    assert "test -s data/historical_research/pit_oos_audit.json" in block
