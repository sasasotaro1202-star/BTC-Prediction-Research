import numpy as np

from src.risk_gated_selective_routing_oos import (
    CLASSES,
    EXPERTS,
    _apply_gate,
    _choose_threshold,
    _norm,
    _metrics,
)


def test_norm_is_finite_and_sums_to_one():
    p = _norm(np.asarray([0.2, 0.3, 0.5]))
    assert p.shape == (1, 3)
    assert np.isfinite(p).all()
    assert np.isclose(p.sum(), 1.0)


def test_gate_preserves_low_risk_and_switches_only_high_risk():
    baseline = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]], dtype=float)
    expert = np.asarray([[0.2, 0.2, 0.6], [0.7, 0.2, 0.1]], dtype=float)
    risk = np.asarray([0.2, 0.9], dtype=float)
    out = _apply_gate(baseline, expert, risk, 0.75)
    np.testing.assert_allclose(out[0], baseline[0])
    np.testing.assert_allclose(out[1], expert[1])


def test_threshold_selection_uses_only_supplied_validation_rows():
    y = ["DOWN", "DOWN", "UP", "UP", "FLAT", "FLAT"] * 20
    baseline = np.tile(np.asarray([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]], dtype=float), (40, 1))
    expert = np.tile(np.asarray([[0.7, 0.2, 0.1], [0.2, 0.2, 0.6], [0.1, 0.7, 0.2]], dtype=float), (40, 1))
    risk = np.linspace(0.1, 0.9, len(y))
    threshold, info = _choose_threshold(y, baseline, expert, risk)
    assert threshold in (0.60, 0.65, 0.70, 0.75, 0.80)
    assert info["selected"]["threshold"] == threshold


def test_metrics_contract():
    y = ["DOWN", "FLAT", "UP"]
    p = np.eye(3, dtype=float)
    m = _metrics(y, p)
    assert m["n"] == 3
    assert m["accuracy"] == 1.0
    assert m["logloss"] >= 0.0
    assert m["brier"] >= 0.0
