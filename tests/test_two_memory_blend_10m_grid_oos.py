from src.two_memory_blend_10m_grid_oos import _blend, _weight, _norm, _metrics
import numpy as np


def test_weight_grid_is_bounded():
    assert _weight(None, None, 3.0) == 0.5
    assert 0.15 <= _weight(1.0, 0.5, 1.0) <= 0.85
    assert 0.15 <= _weight(0.5, 1.0, 10.0) <= 0.85
    assert _weight(1.0, 0.5, 0.0) == 0.5


def test_blend_is_normalized():
    p = _blend(np.asarray([0.8,0.1,0.1]), np.asarray([0.1,0.2,0.7]), 0.5)
    assert p.shape == (1,3)
    assert np.isclose(p.sum(),1.0)
    assert np.isfinite(p).all()


def test_metrics_contract():
    m=_metrics(["DOWN","FLAT","UP"],np.eye(3))
    assert m["accuracy"]==1.0
    assert m["logloss"]>=0
    assert m["brier"]>=0
    assert m["ece"]>=0
