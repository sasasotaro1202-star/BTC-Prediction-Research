import numpy as np
from src.high_disagreement_soft_oos import _ci, _soft_correction, SOFTMAX_TEMPERATURE, DISAGREE_THRESHOLD, EXPERTS


def _row():
    return {"y": "UP"}


def test_ci_finite():
    out = _ci([-0.1, 0.0, 0.1, -0.05])
    assert np.isfinite(out["mean"])
    assert out["lower"] <= out["upper"]


def test_soft_correction_only_activates_on_disagreement():
    rows = [_row()]
    base = np.asarray([[0.1, 0.1, 0.8]], dtype=float)
    probs = {name: base.copy() for name in EXPERTS}
    candidate, active, disagreement, weights = _soft_correction(probs, rows, {})
    assert active.tolist() == [False]
    assert np.allclose(candidate, base)
    assert disagreement.tolist() == [0.0]
    assert np.isclose(weights.sum(), 1.0)


def test_temperature_and_threshold_are_fixed():
    assert SOFTMAX_TEMPERATURE == 2.0
    assert DISAGREE_THRESHOLD == 0.50
