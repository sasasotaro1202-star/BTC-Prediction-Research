import numpy as np
from src.archive_online_hedge_oos import EXPERTS, WARMUP, run_hedge


def _row(y, shift=0.0):
    base = {
        "logreg": np.array([0.2, 0.3, 0.5], float),
        "extra": np.array([0.2, 0.4, 0.4], float),
        "rf": np.array([0.25, 0.35, 0.4], float),
        "hgb": np.array([0.3, 0.3, 0.4], float),
        "ensemble": np.array([0.2, 0.35, 0.45], float),
    }
    if shift:
        base["logreg"] = np.array([0.6, 0.2, 0.2], float)
    return {"timestamp": 1, "y": y, "probs": base}


def test_hedge_updates_only_after_current_prediction():
    rows_up = [_row("UP") for _ in range(WARMUP + 2)]
    rows_down = [_row("DOWN") for _ in range(WARMUP + 2)]
    _, trace_up, _ = run_hedge(rows_up)
    _, trace_down, _ = run_hedge(rows_down)
    assert trace_up[WARMUP] == trace_down[WARMUP]


def test_holdout_inherits_development_state():
    dev = [_row("UP") for _ in range(WARMUP + 10)]
    hold = [_row("DOWN") for _ in range(2)]
    _, _, state = run_hedge(dev)
    _, traced, _ = run_hedge(hold, state=state)
    assert traced[0] != {expert: 1.0 / len(EXPERTS) for expert in EXPERTS}


def test_probability_outputs_are_valid():
    rows = [_row("UP") for _ in range(WARMUP + 5)]
    probs, _, _ = run_hedge(rows)
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=1), 1.0)
