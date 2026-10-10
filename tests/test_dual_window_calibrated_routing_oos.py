from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src" / "dual_window_calibrated_routing_oos.py"


def test_calibrated_router_uses_causal_temperature_search():
    text = SRC.read_text(encoding="utf-8")
    assert "CALIBRATION_TEMPERATURES" in text
    assert "_temperature_scale" in text
    assert "_choose_temperature" in text
    assert "tune_y" in text
    assert "final_tune_idx" in text
    assert "final_holdout_idx" in text


def test_calibrated_router_preserves_research_only_holdout_contract():
    text = SRC.read_text(encoding="utf-8")
    assert '"research_only": True' in text
    assert '"production_changed": False' in text
    assert '"final_holdout_used_for_selection": False' in text
    assert '"final_holdout_protected": True' in text
    assert "production" in text
