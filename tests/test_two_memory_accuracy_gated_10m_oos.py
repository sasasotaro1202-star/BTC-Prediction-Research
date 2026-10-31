import numpy as np

from src.two_memory_accuracy_gated_10m_oos import (
    GATE_THRESHOLD,
    _apply_gate,
    _gate_features,
    _fit_gate,
)


def test_gate_features_are_finite_and_case_level():
    frozen = np.asarray([[0.70, 0.20, 0.10], [0.34, 0.33, 0.33]], dtype=float)
    recent = np.asarray([[0.10, 0.20, 0.70], [0.33, 0.34, 0.33]], dtype=float)
    blend = np.asarray([[0.40, 0.20, 0.40], [0.335, 0.335, 0.33]], dtype=float)
    x = _gate_features(frozen, recent, blend, 0.6)
    assert x.shape[0] == 2
    assert x.shape[1] >= 15
    assert np.isfinite(x).all()


def test_gate_is_safe_without_prior_training():
    frozen = np.asarray([[0.70, 0.20, 0.10]], dtype=float)
    recent = np.asarray([[0.10, 0.20, 0.70]], dtype=float)
    blend = np.asarray([[0.40, 0.20, 0.40]], dtype=float)
    out, use, prob = _apply_gate(None, frozen, recent, blend, 0.6)
    assert np.allclose(out, frozen)
    assert not bool(use[0])
    assert prob[0] == 0.0
    assert GATE_THRESHOLD == 0.65


def test_gate_requires_both_classes_in_prior_labels():
    x = [np.zeros(18).tolist()] * 1000
    y = [0] * 1000
    assert _fit_gate(x, y) is None
