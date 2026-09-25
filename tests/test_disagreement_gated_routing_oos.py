import numpy as np

from src.disagreement_gated_routing_oos import (
    _new_state,
    _route_one,
    _update_state,
)

def _row():
    return {"x":[0.0,0.0,0.001,0.002,0.0,0.0001,0.0002,0.5]+[0.0]*7,"y":"UP"}

def test_no_route_when_alternative_does_not_disagree():
    state=_new_state()
    state["seen"]=1000
    probs={
      "production":np.array([0.1,0.2,0.7]),
      "logreg":np.array([0.1,0.2,0.69]),
      "extra_trees":np.array([0.1,0.2,0.7]),
      "hgb":np.array([0.1,0.2,0.68]),
    }
    out,reason,trace=_route_one(state,probs,_row())
    assert trace["routed"] is False
    assert np.allclose(out,probs["production"])

def test_route_only_after_causal_history_advantage():
    state=_new_state()
    state["seen"]=1000
    state["global"]["production"]=[100,120]
    state["global"]["logreg"]=[115,120]
    probs={
      "production":np.array([0.6,0.2,0.2]),
      "logreg":np.array([0.1,0.2,0.7]),
      "extra_trees":np.array([0.6,0.2,0.2]),
      "hgb":np.array([0.6,0.2,0.2]),
    }
    out,reason,trace=_route_one(state,probs,_row())
    assert trace["routed"] is True
    assert trace["expert"]=="logreg"
    assert int(np.argmax(out))==2

def test_update_is_post_prediction():
    state=_new_state()
    probs={
      "production":np.array([0.6,0.2,0.2]),
      "logreg":np.array([0.1,0.2,0.7]),
      "extra_trees":np.array([0.6,0.2,0.2]),
      "hgb":np.array([0.6,0.2,0.2]),
    }
    before=state["seen"]
    _update_state(state,probs,_row())
    assert state["seen"]==before+1
    assert state["global"]["production"]==[0,1]
