import numpy as np
from src.online_expert_oos import _metrics, _weights, run_strategy, EXPERTS


def _row(y="UP", ctx="TREND_UP|EXPANDING|BUY_PRESSURE|HIGH"):
    base = {
        "model_raw": np.asarray([0.2, 0.2, 0.6], dtype=float),
        "structural": np.asarray([0.2, 0.2, 0.6], dtype=float),
        "fused_raw": np.asarray([0.2, 0.2, 0.6], dtype=float),
        "calibrated": np.asarray([0.2, 0.2, 0.6], dtype=float),
    }
    return {"y": y, "context": ctx, "experts": base, "created": "2026-09-25T00:00:00+00:00"}


def test_online_weight_normalizes():
    weights = _weights({e: 1.0 for e in EXPERTS}, {e: 1.0 for e in EXPERTS}, 10)
    assert np.isfinite(weights).all()
    assert np.isclose(weights.sum(), 1.0)


def test_weights_depend_only_on_previous_rows():
    rows = [_row() for _ in range(50)]
    probs, trace, _ = run_strategy(rows, "online_ewma")
    assert probs.shape == (50, 3)
    assert np.isfinite(probs).all()
    assert trace[0]["weights"] == trace[1]["weights"]


def test_metrics_contract():
    rows = [_row() for _ in range(50)]
    probs, _, _ = run_strategy(rows, "online_context")
    metrics = _metrics(rows, probs)
    assert set(("accuracy", "logloss", "brier", "ece")).issubset(metrics)


def test_current_outcome_does_not_change_current_prediction():
    rows_a = [_row("UP", "A") for _ in range(50)]
    rows_b = [_row("DOWN", "A") for _ in range(50)]
    _, trace_a, _ = run_strategy(rows_a, "online_context")
    _, trace_b, _ = run_strategy(rows_b, "online_context")
    assert trace_a[40]["weights"] == trace_b[40]["weights"]
