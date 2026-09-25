from pathlib import Path
import numpy as np

from src.rich_production_research import (
    CLASSES,
    LEGACY_FEATURES,
    LEGACY_INDICES,
    RICH_FEATURES,
    _block_bootstrap_ci,
    _metrics,
    _relative_gain,
)


def test_rich_schema_contains_legacy_features_exactly():
    assert len(LEGACY_FEATURES) == 15
    assert len(RICH_FEATURES) == 53
    assert len(LEGACY_INDICES) == 15
    assert [RICH_FEATURES[i] for i in LEGACY_INDICES] == [
        "ret1","ret3","ret5","ret10","accel","rv5","rv10",
        "rangepos10","body","upper","lower","volratio","voltrend",
        "ema_gap_5m","ema_gap_10m",
    ]


def test_metrics_three_class_known_case():
    y = ["DOWN", "FLAT", "UP"]
    p = np.eye(3, dtype=float)
    m = _metrics(y, p)
    assert m["n"] == 3
    assert m["accuracy"] == 1.0
    assert m["brier"] < 1e-12
    assert m["logloss"] < 1e-6


def test_relative_gain_direction():
    assert abs(_relative_gain(0.50, 0.55, True) - 0.10) < 1e-12
    assert abs(_relative_gain(1.00, 0.97, False) - 0.03) < 1e-12


def test_block_bootstrap_is_paired_and_finite():
    y = np.array(["DOWN","FLAT","UP"] * 2000, dtype=object)
    rich = np.tile(np.eye(3), (2000,1))
    base = np.roll(rich, 1, axis=1)
    out = _block_bootstrap_ci(y, rich, base)
    assert out["n_blocks"] >= 5
    assert np.isfinite(out["mean_block_accuracy_delta"])
    assert out["ci95_low"] > 0.0


def test_research_only_marker_is_present():
    text = Path("src/rich_production_research.py").read_text(encoding="utf-8")
    assert 'research_only": True' in text
    assert 'production_changed": False' in text

def test_extended_microstructure_features_are_append_only():
    expected_tail = [
        "rv60","volume_intensity_5m_60m","trade_intensity_5m_60m",
        "flow_30m","flow_60m","flow_toxicity_30m","flow_toxicity_60m",
        "amihud_15m","amihud_30m","range_intensity_10m",
    ]
    assert list(RICH_FEATURES[-10:]) == expected_tail
    assert [RICH_FEATURES[i] for i in LEGACY_INDICES] == [
        "ret1","ret3","ret5","ret10","accel","rv5","rv10",
        "rangepos10","body","upper","lower","volratio","voltrend",
        "ema_gap_5m","ema_gap_10m",
    ]
