import numpy as np
from src.rolling_window_grid_10m_oos import _metrics, _norm


def test_probability_contract():
    p = _norm(np.asarray([0.2, 0.3, 0.5]))
    assert p.shape == (1, 3)
    assert np.isclose(p.sum(), 1.0)
    assert np.isfinite(p).all()


def test_metrics_contract():
    m = _metrics(["DOWN", "FLAT", "UP"], np.eye(3))
    assert m["accuracy"] == 1.0
    assert m["logloss"] >= 0
    assert m["brier"] >= 0
    assert m["ece"] >= 0
