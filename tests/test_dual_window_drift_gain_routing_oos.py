from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src" / "dual_window_drift_gain_routing_oos.py"


def test_dual_window_gain_router_uses_recent_and_long_windows():
    text = SRC.read_text(encoding="utf-8")
    assert "SHORT_META_BLOCK = 350" in text
    assert "META_BLOCK = 700" in text
    assert "_fit_gain_models" in text
    assert "_predict_gain_views" in text
    assert "DISAGREEMENT_QUANTILES" in text
    assert "short_weight" in text
    assert "disagreement_cap" in text


def test_dual_window_gain_router_keeps_temporal_gap_and_holdout_protection():
    text = SRC.read_text(encoding="utf-8")
    assert "GAP_BARS" in text
    assert "META_BLOCK + TUNE_BLOCK + GAP_BARS[horizon]" in text
    assert "final_meta_idx" in text
    assert "final_tune_idx" in text
    assert "final_holdout_idx" in text
    assert "final_holdout_used_for_selection" in text
    assert "final_holdout_protected" in text
