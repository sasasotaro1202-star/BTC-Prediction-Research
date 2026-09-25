import numpy as np
from src.conformal_expert_set_oos import conformal_pvalues, route_block

def test_conformal_pvalues_bounded():
    cal_rows = [{"y": "UP"} for _ in range(40)] + [{"y": "DOWN"} for _ in range(40)]
    cal_probs = np.tile(np.asarray([[0.1, 0.1, 0.8]], dtype=float), (80, 1))
    p = conformal_pvalues(cal_probs, cal_rows, np.asarray([[0.1, 0.1, 0.8]], dtype=float))
    assert p.shape == (1,)
    assert np.isfinite(p).all()
    assert 0.0 <= p[0] <= 1.0

def test_route_block_falls_back_safely():
    cal = [{"y": "UP"} for _ in range(60)]
    base = np.tile(np.asarray([[0.2, 0.2, 0.6]], dtype=float), (60, 1))
    parts = {"logreg": base, "extra_trees": base, "hgb": base, "soft_equal": base}
    routed, sizes, pvals = route_block(parts, cal, parts)
    assert routed.shape == (60, 3)
    assert np.isfinite(routed).all()
    assert np.isfinite(pvals).all()
    assert np.allclose(routed.sum(axis=1), 1.0)

def test_routing_is_probability_contract():
    cal = [{"y": "DOWN"} for _ in range(30)] + [{"y": "UP"} for _ in range(30)]
    p = np.asarray([[0.45, 0.10, 0.45]], dtype=float)
    parts = {"logreg": p, "extra_trees": p, "hgb": p, "soft_equal": p}
    cal_parts = parts
    routed, sizes, _ = route_block(cal_parts, cal, parts)
    assert routed.shape == (1, 3)
    assert np.isfinite(routed).all()
