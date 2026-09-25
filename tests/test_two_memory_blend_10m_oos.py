import numpy as np
from src.two_memory_blend_10m_oos import _align, _blend, _weight, _norm, _metrics


def test_weight_is_bounded_and_causal():
    assert _weight(None, None) == 0.5
    assert 0.15 <= _weight(1.0, 0.5) <= 0.85
    assert 0.15 <= _weight(0.5, 1.0) <= 0.85


def test_blend_normalizes():
    frozen = np.asarray([0.8, 0.1, 0.1])
    recent = np.asarray([0.1, 0.2, 0.7])
    out = _blend(frozen, recent, 0.6)
    assert out.shape == (1, 3)
    assert np.isclose(out.sum(), 1.0)
    assert np.isfinite(out).all()


def test_metrics_contract():
    m = _metrics(["DOWN", "FLAT", "UP"], np.eye(3))
    assert m["accuracy"] == 1.0
    assert m["logloss"] >= 0.0
    assert m["brier"] >= 0.0
    assert m["ece"] >= 0.0
