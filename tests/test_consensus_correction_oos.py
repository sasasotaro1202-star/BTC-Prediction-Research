import numpy as np
from src.consensus_correction_oos import _apply, EXPERTS

def test_two_of_three_correction_activates_only_for_majority_against_rf():
    base=np.asarray([[0.7,0.2,0.1]],dtype=float)
    alt=np.asarray([[0.1,0.2,0.7]],dtype=float)
    same=np.asarray([[0.7,0.2,0.1]],dtype=float)
    probs={"random_forest":base,"logreg":alt,"extra_trees":alt,"hgb":same}
    candidates,counts=_apply(probs)
    out,active=candidates["consensus_2of3"]
    assert active.tolist()==[True]
    assert np.argmax(out[0])==2
    assert counts.tolist()==[2]

def test_three_of_three_requires_all_three_to_disagree():
    base=np.asarray([[0.7,0.2,0.1]],dtype=float)
    alt=np.asarray([[0.1,0.2,0.7]],dtype=float)
    probs={"random_forest":base,"logreg":alt,"extra_trees":alt,"hgb":alt}
    candidates,counts=_apply(probs)
    out,active=candidates["consensus_3of3"]
    assert active.tolist()==[True]
    assert np.argmax(out[0])==2
    assert counts.tolist()==[3]

def test_no_correction_when_no_disagreement():
    base=np.asarray([[0.7,0.2,0.1]],dtype=float)
    probs={"random_forest":base,"logreg":base,"extra_trees":base,"hgb":base}
    candidates,counts=_apply(probs)
    assert candidates["consensus_2of3"][1].tolist()==[False]
    assert np.allclose(candidates["consensus_2of3"][0],base)
