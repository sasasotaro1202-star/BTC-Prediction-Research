import numpy as np
from src.conservative_soft_routing_oos import _uncertainty, _mix

def test_uncertainty_bounded():
    p={k:np.array([[.6,.2,.2]]) for k in ("production","logreg","extra_trees","hgb")}
    u=_uncertainty(p)
    assert np.isfinite(u).all()
    assert 0.0 <= float(u[0]) <= 1.0

def test_soft_route_is_capped_and_normalized():
    p={k:np.array([[.6,.2,.2]]) for k in ("production","logreg","extra_trees","hgb")}
    final,r=_mix(p,np.array([True]),{"logreg":.10,"extra_trees":.10,"hgb":.05})
    assert np.isclose(r[0].sum(),1.0)
    assert np.isclose(final[0].sum(),1.0)
    assert r[0,0] >= .75

def test_low_risk_keeps_champion_exactly():
    p={k:np.array([[.6,.2,.2]]) for k in ("production","logreg","extra_trees","hgb")}
    final,_=_mix(p,np.array([False]),{"logreg":.2,"extra_trees":.03,"hgb":.02})
    assert np.allclose(final,p["production"])
