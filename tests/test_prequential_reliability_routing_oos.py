import math
import numpy as np

from src.prequential_reliability_routing_oos import (
    EXPERTS,
    _feature_context,
    _fresh_state,
    _loss,
    _route_rows,
    _weights,
)


def _row(y="UP"):
    return {
        "id": "t",
        "x": [0.0, 0.0, 0.001, 0.002, 0.0, 0.0001, 0.0002, 0.5] + [0.0] * 7,
        "y": y,
        "production": np.asarray([0.15, 0.20, 0.65], dtype=float),
        "expert_probs": {
            "logreg": np.asarray([0.20, 0.20, 0.60], dtype=float),
            "extra_trees": np.asarray([0.15, 0.15, 0.70], dtype=float),
            "hgb": np.asarray([0.25, 0.20, 0.55], dtype=float),
        },
    }


def test_context_is_deterministic():
    assert _feature_context(_row()["x"]) == "UP|LOW|MID"


def test_weights_normalize():
    s = _fresh_state()
    w = _weights(s["global_ema"], s["context_ema"], s["context_counts"], "UP|LOW|MID", "online_context", 100)
    assert np.isfinite(w).all()
    assert np.isclose(float(w.sum()), 1.0)


def test_current_outcome_cannot_change_current_prediction():
    rows_up = [_row("UP") for _ in range(60)]
    rows_down = [_row("DOWN") for _ in range(60)]
    p_up, _, _, _ = _route_rows(rows_up, "online_context", _fresh_state())
    p_down, _, _, _ = _route_rows(rows_down, "online_context", _fresh_state())
    assert np.allclose(p_up[0], p_down[0])


def test_loss_is_finite():
    assert math.isfinite(_loss(np.asarray([0.2, 0.3, 0.5]), "UP"))
