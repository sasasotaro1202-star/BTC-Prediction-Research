import numpy as np
from src.tail_benefit_gated_routing_oos import _candidate, _uncertainty

def test_uncertainty_is_bounded():
    p = {k: np.array([[0.6, 0.2, 0.2]]) for k in ("production", "logreg", "extra_trees", "hgb")}
    u = _uncertainty(p)
    assert np.isfinite(u).all()
    assert 0.0 <= float(u[0]) <= 1.0

def test_candidate_is_normalized():
    p = {k: np.array([[0.7, 0.2, 0.1]]) for k in ("production", "logreg", "extra_trees", "hgb")}
    c = _candidate(p)
    assert c.shape == (1, 3)
    assert np.isclose(c.sum(), 1.0)
    assert np.all(c >= 0.0)
