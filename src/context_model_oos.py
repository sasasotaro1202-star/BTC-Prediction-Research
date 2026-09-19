"""Research-only contextual model routing for BTC short-horizon classification.

The router tests whether different market contexts genuinely benefit from
different estimators. It is deliberately not connected to production model
promotion. All routing thresholds and specialist training data are determined
from information available before each test block.
"""
from __future__ import annotations
import math
from typing import Callable, Iterable

import numpy as np


CONTEXTS = (
    "trend_up",
    "trend_down",
    "range",
    "high_vol_trend_up",
    "high_vol_trend_down",
    "high_vol_range",
)
MIN_SPECIALIST_TRAIN = 250


def _median(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=float)
    if len(a) == 0:
        return 0.0
    return float(np.median(a))


def fit_context_thresholds(train_rows):
    """Fit thresholds only on the pre-test training window."""
    vols = [float(r["x"][6]) for r in train_rows]
    rets = [float(r["x"][3]) for r in train_rows]
    return {
        "vol_median": _median(vols),
        "trend_band": max(0.00015, 0.35 * _median(abs(v) for v in rets)),
    }


def context_of(row, thresholds):
    ret10 = float(row["x"][3])
    vol10 = float(row["x"][6])
    high = vol10 > thresholds["vol_median"]
    if ret10 > thresholds["trend_band"]:
        base = "trend_up"
    elif ret10 < -thresholds["trend_band"]:
        base = "trend_down"
    else:
        base = "range"
    return ("high_vol_" + base) if high else base


def route_predictions(train_rows, test_rows, factory: Callable[[], object]):
    """Walk one already-defined chronological split using contextual specialists.

    A specialist is trained only when its context has enough historical rows.
    Otherwise the global model is used. No test labels are used for routing.
    """
    if len(train_rows) < MIN_SPECIALIST_TRAIN:
        return None

    thresholds = fit_context_thresholds(train_rows)
    global_model = factory()
    X = np.asarray([r["x"] for r in train_rows], dtype=float)
    y = np.asarray([r["y"] for r in train_rows])
    if len(set(y.tolist())) < 3:
        return None
    global_model.fit(X, y)

    specialists = {}
    for context in CONTEXTS:
        subset = [r for r in train_rows if context_of(r, thresholds) == context]
        labels = {r["y"] for r in subset}
        if len(subset) < MIN_SPECIALIST_TRAIN or len(labels) < 3:
            continue
        model = factory()
        model.fit(np.asarray([r["x"] for r in subset], dtype=float),
                  np.asarray([r["y"] for r in subset]))
        specialists[context] = model

    def aligned(model, rows):
        probs = np.asarray(model.predict_proba(np.asarray([r["x"] for r in rows], dtype=float)), dtype=float)
        classes = list(model.classes_)
        out = np.full((len(rows), 3), 1e-6, dtype=float)
        names = ["DOWN", "FLAT", "UP"]
        for j, cls in enumerate(classes):
            if cls in names:
                out[:, names.index(cls)] = probs[:, j]
        out /= out.sum(axis=1, keepdims=True)
        return out

    global_probs = aligned(global_model, test_rows)
    routed = global_probs.copy()
    used = {}
    for context in CONTEXTS:
        idx = [i for i, r in enumerate(test_rows) if context_of(r, thresholds) == context]
        if not idx or context not in specialists:
            used[context] = {"n": len(idx), "specialist": False}
            continue
        p = aligned(specialists[context], [test_rows[i] for i in idx])
        routed[idx] = p
        used[context] = {"n": len(idx), "specialist": True}
    return {
        "global_probs": global_probs,
        "routed_probs": routed,
        "contexts": [context_of(r, thresholds) for r in test_rows],
        "specialist_usage": used,
        "thresholds": thresholds,
    }


def strict_improvement(global_metrics, routed_metrics):
    """Promotion-style research gate; never call this from production."""
    if not global_metrics or not routed_metrics:
        return False
    return (
        routed_metrics["logloss"] <= global_metrics["logloss"] - 0.005
        and routed_metrics["brier"] <= global_metrics["brier"] - 0.002
        and routed_metrics["accuracy"] >= global_metrics["accuracy"] - 0.01
    )
