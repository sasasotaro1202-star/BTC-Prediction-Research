import numpy as np
from src.targeted_uncertainty_routing_oos import _uncertainty

def test_uncertainty_bounded():
    p={"logreg":np.array([[.6,.2,.2]]), "extra_trees":np.array([[.6,.2,.2]]), "hgb":np.array([[.6,.2,.2]]), "soft_equal":np.array([[.6,.2,.2]])}
    u=_uncertainty(p)
    assert np.isfinite(u).all()
    assert 0.0 <= float(u[0]) <= 1.0
