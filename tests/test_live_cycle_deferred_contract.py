from pathlib import Path


def test_deferred_live_cycle_does_not_fail_on_missing_pit_audit():
    text = Path(".github/workflows/btc_live_cycle.yml").read_text(encoding="utf-8")
    assert 'if os.environ.get("BTC_LIVE_DEFERRED") == "1":' in text
    assert "PIT audit unavailable because live prediction was safely DEFERRED" in text
    assert 'raise SystemExit("PIT audit output missing; refusing to record history")' in text
