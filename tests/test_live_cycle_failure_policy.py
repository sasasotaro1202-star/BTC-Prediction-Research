from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "btc_live_cycle.yml"


def test_live_cycle_defers_stale_future_and_predictor_fail_closed():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "stale_live_market_event:" in text
    assert "future_live_market_event:" in text
    assert "live_prediction_fail_closed:" in text
    assert 'echo "BTC_LIVE_DEFERRED=1"' in text


def test_live_cycle_canonicalizes_prediction_state_before_settlement():
    text = WORKFLOW.read_text(encoding="utf-8")
    canonical = text.index("      - name: Canonicalize prediction state before settlement")
    settle = text.index("      - name: Settle due BTC predictions", canonical)
    block = text[canonical:settle]
    assert "python scripts/merge_prediction_state.py --compact data/predictions.db" in block
    assert "canonical_compaction_snapshot" in block
    assert canonical < settle
