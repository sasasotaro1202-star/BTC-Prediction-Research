import numpy as np

from src.gain_aware_routing_oos import (
    ALTERNATIVES,
    _metrics,
    _norm,
    _route,
    causal_state_features,
)


def test_norm_contract():
    p = _norm(np.asarray([0.2, 0.3, 0.5], dtype=float))
    assert p.shape == (1, 3)
    assert np.isfinite(p).all()
    assert np.isclose(float(p.sum()), 1.0)


def test_route_changes_only_when_predicted_gain_clears_threshold():
    production = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]], dtype=float)
    alternatives = {
        name: production.copy()
        for name in ALTERNATIVES
    }
    alternatives["logreg"][1] = np.asarray([0.7, 0.2, 0.1])
    gains = np.zeros((2, len(ALTERNATIVES)), dtype=float)
    gains[1, 0] = 0.02
    out, choose = _route(production, alternatives, gains, 0.01, 0.5)
    assert choose.tolist() == [False, True]
    np.testing.assert_allclose(out[0], production[0])
    np.testing.assert_allclose(
        out[1],
        0.5 * production[1] + 0.5 * alternatives["logreg"][1],
    )


def test_causal_state_is_prior_only():
    y = ["DOWN", "UP", "FLAT"]
    p = {
        name: np.tile(np.asarray([[0.7, 0.2, 0.1]], dtype=float), (3, 1))
        for name in ("production",) + ALTERNATIVES
    }
    state = causal_state_features(y, p)
    assert state["loss_production"][0] == 1.05
    assert state["loss_production"][1] != state["loss_production"][0]
    assert state["gain_logreg"][0] == 0.0


def test_metrics_nonnegative():
    y = ["DOWN", "FLAT", "UP"]
    p = np.eye(3, dtype=float)
    m = _metrics(y, p)
    assert m["accuracy"] == 1.0
    assert m["logloss"] >= 0.0
    assert m["brier"] >= 0.0
