import numpy as np
from src.risk_conditional_rescue_oos import _risk_features, _route

def _rows(n):
    return [{"x":[0.0,0.0,0.01,0.02,0.0,0.01,0.02,0.5,0.0,0.0,0.0,1.0,1.0,0.0,0.0],"y":"UP"} for _ in range(n)]

def test_risk_features_finite():
    rows=_rows(3)
    probs={k:np.tile(v,(3,1)) for k,v in {
        "logreg":[0.4,0.3,0.3],"extra_trees":[0.5,0.2,0.3],"hgb":[0.45,0.25,0.3]
    }.items()}
    x=_risk_features(probs,rows)
    assert x.shape[0]==3
    assert np.isfinite(x).all()

def test_no_router_models_is_fail_closed():
    rows=_rows(2)
    probs={k:np.tile(v,(2,1)) for k,v in {
        "logreg":[0.4,0.3,0.3],"extra_trees":[0.5,0.2,0.3],"hgb":[0.45,0.25,0.3]
    }.items()}
    out,use,chosen,high,q,scores=_route(rows,probs,{},0.60)
    ensemble=np.mean(np.stack(list(probs.values()),axis=0),axis=0)
    assert np.allclose(out,ensemble)
    assert not use.any()
    assert (chosen=="ensemble").all()


def test_risk_threshold_is_external_and_causal():
    rows=_rows(2)
    probs={k:np.tile(v,(2,1)) for k,v in {
        "logreg":[0.6,0.25,0.15],"extra_trees":[0.5,0.3,0.2],"hgb":[0.55,0.25,0.2]
    }.items()}
    out,use,chosen,high,q,scores=_route(rows,probs,{},0.60,0.0)
    assert high.all()
    assert np.isclose(q,0.0)
