import numpy as np

from src.risk_adjusted_prequential_oos import (
    EXPERTS,
    _new_state,
    _route,
    _uncertainty_features,
)


def _probs():
    base = np.asarray(
        [
            [0.70, 0.20, 0.10],
            [0.10, 0.20, 0.70],
            [0.45, 0.10, 0.45],
        ],
        dtype=float,
    )
    return {
        "logreg": base.copy(),
        "extra_trees": np.roll(base, 1, axis=1),
        "hgb": base.copy(),
        "soft_equal": base.copy(),
    }


def test_uncertainty_features_are_finite_and_bounded():
    f = _uncertainty_features(_probs())
    for name in ("confidence", "margin", "entropy", "disagreement", "agreement", "uncertainty"):
        assert np.isfinite(f[name]).all()
        assert np.all((f[name] >= 0.0) & (f[name] <= 1.0))


def test_route_normalizes_and_preserves_probability_contract():
    p = _probs()
    reliability = {name: np.full(3, 0.5) for name in EXPERTS}
    routed, weights = _route(p, reliability, _new_state())
    assert routed.shape == (3, 3)
    assert weights.shape == (3, len(EXPERTS))
    assert np.isfinite(routed).all()
    assert np.isfinite(weights).all()
    assert np.allclose(routed.sum(axis=1), 1.0)
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_routing_does_not_read_labels():
    p = _probs()
    reliability = {name: np.asarray([0.1, 0.5, 0.9]) for name in EXPERTS}
    routed_a, _ = _route(p, reliability, _new_state())
    routed_b, _ = _route(p, reliability, _new_state())
    assert np.allclose(routed_a, routed_b)
