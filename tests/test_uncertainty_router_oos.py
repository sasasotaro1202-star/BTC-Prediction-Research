import numpy as np
from src.uncertainty_router_oos import (
    EXPERTS, _entropy, _router_features, _route,
)


def _rows(n=12):
    return [{"x":[0.01,0.02,0.03,0.04,0.00,0.10,0.11,0.50,0,0,0,1.2,1.0,0,0],
             "y":"UP"} for _ in range(n)]


def _probs(n=12):
    p=np.tile(np.asarray([0.1,0.2,0.7],dtype=float),(n,1))
    return {
        "logreg": p.copy(),
        "extra_trees": p.copy(),
        "hgb": p.copy(),
        "soft_equal": p.copy(),
    }


def test_entropy_is_finite():
    assert np.isfinite(_entropy(np.asarray([0.2,0.3,0.5])))


def test_router_features_finite_and_2d():
    rows=_rows()
    f=_router_features(_probs(),rows,include_context=True)
    assert f.ndim==2
    assert f.shape[0]==len(rows)
    assert np.isfinite(f).all()


def test_route_falls_back_to_soft_equal_without_router():
    rows=_rows()
    probs=_probs()
    X=_router_features(probs,rows,include_context=False)
    out,chosen=_route(X,probs,{},rows)
    assert out.shape==(len(rows),3)
    assert all(x=="soft_equal" for x in chosen.tolist())


def test_route_probabilities_normalize():
    rows=_rows()
    probs=_probs()
    X=_router_features(probs,rows,include_context=False)
    out,chosen=_route(X,probs,{},rows)
    assert np.allclose(out.sum(axis=1),1.0)
