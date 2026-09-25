# Validation revision marker: execute-only change; algorithm untouched.
from src.rolling_retrain_10m_oos import _metrics, _norm, _rf
import numpy as np


def test_probabilities_and_metrics_contract():
    p = _norm(np.asarray([0.2, 0.3, 0.5]))
    assert p.shape == (1, 3)
    assert np.isclose(p.sum(), 1.0)
    m = _metrics(["DOWN", "FLAT", "UP"], np.eye(3))
    assert m["accuracy"] == 1.0
    assert m["logloss"] >= 0
    assert m["brier"] >= 0


def test_rf_configuration_is_fixed():
    model = _rf()
    assert model.n_estimators == 200
    assert model.max_depth == 10
    assert model.min_samples_leaf == 10
    assert model.random_state == 42
