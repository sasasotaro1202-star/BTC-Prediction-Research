import numpy as np
from src.production_core_oos import _metrics, _norm, _stability


def test_probability_normalization():
    p = _norm(np.asarray([0.2, 0.3, 0.5]))
    assert p.shape == (1, 3)
    assert np.isclose(p.sum(), 1.0)
    assert np.isfinite(p).all()


def test_metrics_contract():
    y = ["DOWN", "FLAT", "UP"]
    p = np.eye(3)
    m = _metrics(y, p)
    assert m["n"] == 3
    assert m["accuracy"] == 1.0
    assert m["logloss"] >= 0.0
    assert m["brier"] >= 0.0
    assert m["ece"] >= 0.0


def test_stability_is_deterministic():
    blocks = [
        {"accuracy": 0.4, "logloss": 1.0},
        {"accuracy": 0.5, "logloss": 0.9},
    ]
    a = _stability(blocks)
    b = _stability(blocks)
    assert a == b
