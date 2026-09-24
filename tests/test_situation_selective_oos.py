import numpy as np
from src.situation_selective_oos import _entropy, _margin, _state_score

def test_entropy_and_margin_are_bounded():
    p = np.array([0.8, 0.1, 0.1])
    assert 0.0 <= _entropy(p) <= 1.0
    assert 0.0 <= _margin(p) <= 1.0

def test_state_score_is_finite_and_causal():
    row = {
        "production": [0.8, 0.1, 0.1],
        "x": [0.0, 0.0, 0.001, 0.0, 0.0, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.001, 0.001],
        "y": "UP",
    }
    score = _state_score(row)
    assert np.isfinite(score)
