"""Research-only contextual model routing for BTC short-horizon classification.

The router tests whether different market contexts genuinely benefit from
different estimators. It is deliberately not connected to production model
promotion. All routing thresholds and specialist training data are determined
from information available before each test block.
"""
from __future__ import annotations
from typing import Callable, Iterable, Mapping

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


def _select_factory(rows, factories: Mapping[str, Callable[[], object]]):
    """Select one estimator using only the tail of the context's training data."""
    if len(rows) < MIN_SPECIALIST_TRAIN:
        return None, None
    split=max(int(len(rows)*0.80), len(rows)-100)
    fit, valid=rows[:split], rows[split:]
    if len(fit)<100 or len(valid)<50:
        return None, None
    names=[r['y'] for r in fit]
    if len(set(names))<3:
        return None, None
    Xfit=np.asarray([r['x'] for r in fit],dtype=float); yfit=np.asarray(names)
    Xv=np.asarray([r['x'] for r in valid],dtype=float); yv=np.asarray([r['y'] for r in valid])
    best=None
    for name, factory in factories.items():
        try:
            model=factory(); model.fit(Xfit,yfit)
            p=model.predict_proba(Xv); classes=list(model.classes_); aligned=np.full((len(valid),3),1e-6,float)
            for j,cls in enumerate(classes):
                if cls in ('DOWN','FLAT','UP'): aligned[:,('DOWN','FLAT','UP').index(cls)]=p[:,j]
            aligned/=aligned.sum(axis=1,keepdims=True)
            yidx=np.asarray([('DOWN','FLAT','UP').index(v) for v in yv])
            ll=float(-np.mean(np.log(np.clip(aligned[np.arange(len(yidx)),yidx],1e-12,1.0))))
            brier=float(np.mean(np.sum((aligned-np.eye(3)[yidx])**2,axis=1)))
            score=(ll,brier)
            if best is None or score<best[0]: best=(score,name)
        except Exception:
            continue
    return (factories[best[1]],best[1]) if best else (None,None)


def route_predictions(train_rows, test_rows, factories: Mapping[str, Callable[[], object]]):
    """Walk one already-defined chronological split using contextual specialists.

    A specialist is trained only when its context has enough historical rows.
    Otherwise the global model is used. No test labels are used for routing.
    """
    if len(train_rows) < MIN_SPECIALIST_TRAIN:
        return None

    thresholds = fit_context_thresholds(train_rows)
    global_factory, global_name = _select_factory(train_rows, factories)
    if global_factory is None:
        return None
    global_model = global_factory()
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
        specialist_factory, specialist_name = _select_factory(subset, factories)
        if specialist_factory is None:
            continue
        model = specialist_factory()
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
        "global_model": global_name,
        "specialist_models": {k: v for k, v in ((ctx, _select_factory([r for r in train_rows if context_of(r, thresholds) == ctx], factories)[1]) for ctx in CONTEXTS) if v},
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
