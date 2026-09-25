"""Research-only targeted uncertainty routing for BTC direction.

Uses a prequential reliability router, but activates it only on observations
whose uncertainty exceeds a threshold derived from a previous block. Low-risk
observations keep the frozen soft-equal baseline unchanged. Current/future
outcomes never influence the current route.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_compare import CLASSES, EMBARGO_BARS, PURGE_BARS, load_archive_research_rows, metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/historical_research/targeted_uncertainty_routing_oos.json"

HORIZONS = ("5m", "10m")
EXPERTS = ("production", "logreg", "extra_trees", "hgb")
MIN_TRAIN = 3000
META_BLOCK = 500
TEST_BLOCK = 500
MIN_META_ROWS = 250
MAX_ROWS = 12000
FINAL_HOLDOUT_FRAC = 0.20
UNCERTAINTY_QUANTILE = 0.75
EPS = 1e-7


def _align(model: Any, rows: list[dict[str, Any]]) -> np.ndarray:
    X = np.asarray([r["x"] for r in rows], dtype=float)
    raw = np.asarray(model.predict_proba(X), dtype=float)
    out = np.full((len(rows), 3), EPS, dtype=float)
    idx = {c: i for i, c in enumerate(CLASSES)}
    for j, cls in enumerate(model.classes_):
        if str(cls) in idx:
            out[:, idx[str(cls)]] = raw[:, j]
    out = np.clip(out, EPS, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def _factories():
    return {
        "logreg": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.3, max_iter=2500)),
        ]),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=140, max_depth=10, min_samples_leaf=15,
            max_features="sqrt", random_state=42, n_jobs=-1
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=180, max_leaf_nodes=15, learning_rate=0.04,
            l2_regularization=1.5, random_state=42
        ),
    }


def _fit(train):
    if len(train) < MIN_TRAIN or len({r["y"] for r in train}) < 3:
        raise ValueError("insufficient_or_single_class_training_rows")
    X = np.asarray([r["x"] for r in train], dtype=float)
    y = np.asarray([r["y"] for r in train], dtype=str)
    models = {}
    for name, factory in _factories().items():
        model = factory()
        model.fit(X, y)
        models[name] = model
    return models


def _probs(models, rows):
    p = {name: _align(model, rows) for name, model in models.items()}
    production = np.asarray([r["production"] for r in rows], dtype=float)
    if production.ndim != 2 or production.shape[1] != 3:
        raise ValueError("production_probability_shape_invalid")
    production = np.clip(production, EPS, 1.0)
    production /= production.sum(axis=1, keepdims=True)
    p["production"] = production
    return p


def _uncertainty(p):
    soft = p["production"]
    entropy = -np.sum(soft * np.log(np.clip(soft, EPS, 1.0)), axis=1) / math.log(3.0)
    ordered = np.sort(soft, axis=1)[:, ::-1]
    margin = ordered[:, 0] - ordered[:, 1]
    stack = np.stack([p[e] for e in EXPERTS], axis=0)
    disagreement = np.mean(np.sum((stack - soft[None, :, :]) ** 2, axis=2), axis=0)
    disagreement = np.clip(disagreement / 0.10, 0.0, 1.0)
    uncertainty = np.clip(
        0.55 * entropy + 0.25 * disagreement + 0.20 * (1.0 - margin),
        0.0, 1.0
    )
    return uncertainty


def _router_features(p, rows):
    soft = p["soft_equal"]
    uncertainty = _uncertainty(p)
    x = np.asarray([r["x"] for r in rows], dtype=float)
    features = []
    for expert in EXPERTS:
        q = p[expert]
        ordered = np.sort(q, axis=1)[:, ::-1]
        own_pred = np.argmax(q, axis=1)
        peer = np.mean(
            np.stack([p[e][np.arange(len(rows)), own_pred] for e in EXPERTS if e != expert]),
            axis=0,
        )
        features.append(np.column_stack([
            q[:, 0], q[:, 1], q[:, 2],
            ordered[:, 0] - ordered[:, 1],
            -np.sum(q * np.log(np.clip(q, EPS, 1.0)), axis=1) / math.log(3.0),
            np.abs(ordered[:, 0] - peer),
            uncertainty,
        ]))
    base = np.mean(np.stack(features, axis=0), axis=0)
    context = np.column_stack([x[:, 2], x[:, 3], x[:, 5], x[:, 6], x[:, 7], x[:, 11], x[:, 12]])
    return np.column_stack([base, context])


def _risk_router(meta, meta_rows, test, test_rows):
    y_meta = np.asarray([CLASSES.index(r["y"]) for r in meta_rows], dtype=int)
    X_meta = _router_features(meta, meta_rows)
    models = {}
    for expert in EXPERTS:
        pred = np.argmax(meta[expert], axis=1)
        correct = (pred == y_meta).astype(int)
        if len(np.unique(correct)) < 2:
            continue
        model = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.2, max_iter=1800, class_weight="balanced")),
        ])
        model.fit(X_meta, correct)
        models[expert] = model

    X_test = _router_features(test, test_rows)
    scores = np.column_stack([
        models[e].predict_proba(X_test)[:, 1] if e in models else np.full(len(test_rows), 0.5)
        for e in EXPERTS
    ])
    chosen = np.asarray(EXPERTS)[np.argmax(scores, axis=1)]
    routed = np.empty((len(test_rows), 3), dtype=float)
    for i, e in enumerate(chosen):
        routed[i] = test[e][i]
    routed = np.clip(routed, EPS, 1.0)
    routed /= routed.sum(axis=1, keepdims=True)
    return routed


def _uncertainty_gate(prior_uncertainty):
    finite = np.asarray(prior_uncertainty, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 100:
        return 0.0
    return float(np.quantile(finite, UNCERTAINTY_QUANTILE))


def _eval_dev(rows, horizon):
    gap = int(PURGE_BARS[horizon] + EMBARGO_BARS[horizon])
    outputs = []
    prior_unc = None
    for test_start in range(MIN_TRAIN + META_BLOCK + gap, len(rows), TEST_BLOCK):
        test = rows[test_start:min(test_start + TEST_BLOCK, len(rows))]
        meta_end = test_start - gap
        meta_start = meta_end - META_BLOCK
        train = rows[:meta_start]
        meta = rows[meta_start:meta_end]
        if len(test) < TEST_BLOCK // 2 or len(meta) < MIN_META_ROWS or len(train) < MIN_TRAIN:
            continue

        models = _fit(train)
        meta_probs = _probs(models, meta)
        test_models = _fit(rows[: test_start - gap])
        test_probs = _probs(test_models, test)

        meta_unc = _uncertainty(meta_probs)
        test_unc = _uncertainty(test_probs)
        threshold = _uncertainty_gate(meta_unc)
        routed = _risk_router(meta_probs, meta, test_probs, test)

        gate = test_unc >= threshold
        baseline = test_probs["production"]
        final = baseline.copy()
        final[gate] = routed[gate]

        y = [r["y"] for r in test]
        def m(mask=None):
            yy = y if mask is None else list(np.asarray(y)[mask])
            pp = final if mask is None else final[mask]
            return metrics(yy, pp)
        base = metrics(y, baseline)
        cand = metrics(y, final)
        gated = metrics(list(np.asarray(y)[gate]), final[gate]) if gate.any() else None
        base_gated = metrics(list(np.asarray(y)[gate]), baseline[gate]) if gate.any() else None
        outputs.append({
            "n": len(test),
            "gate_n": int(gate.sum()),
            "gate_rate": float(gate.mean()),
            "uncertainty_threshold": threshold,
            "baseline": base,
            "candidate": cand,
            "delta": {k: float(cand[k] - base[k]) for k in ("accuracy","logloss","brier","ece")},
            "high_uncertainty": {
                "baseline": base_gated,
                "candidate": gated,
                "delta_accuracy": (gated["accuracy"] - base_gated["accuracy"]) if gated else None,
                "n": int(gate.sum()),
            },
        })
        prior_unc = meta_unc

    if not outputs:
        return {"status":"DEFERRED","reason":"no_valid_blocks"}
    total=sum(x["n"] for x in outputs)
    agg={}
    for side in ("baseline","candidate"):
        agg[side]={k:float(sum(x["n"]*x[side][k] for x in outputs)/total) for k in ("accuracy","logloss","brier","ece")}
        agg[side]["n"]=total
    delta={k:agg["candidate"][k]-agg["baseline"][k] for k in ("accuracy","logloss","brier","ece")}
    ll=np.asarray([x["delta"]["logloss"] for x in outputs])
    br=np.asarray([x["delta"]["brier"] for x in outputs])
    ac=np.asarray([x["delta"]["accuracy"] for x in outputs])
    return {
        "status":"OK","blocks":len(outputs),"samples":total,"aggregate":{**agg,"delta":delta},
        "relative_improvement":{
            "accuracy":(agg["candidate"]["accuracy"]-agg["baseline"]["accuracy"])/max(agg["baseline"]["accuracy"],EPS),
            "logloss":(agg["baseline"]["logloss"]-agg["candidate"]["logloss"])/max(agg["baseline"]["logloss"],EPS),
            "brier":(agg["baseline"]["brier"]-agg["candidate"]["brier"])/max(agg["baseline"]["brier"],EPS)
        },
        "stability":{
            "improved_logloss_ratio":float(np.mean(ll<0)),
            "improved_brier_ratio":float(np.mean(br<0)),
            "non_worse_accuracy_ratio":float(np.mean(ac>=-0.005))
        },
        "blocks_detail":outputs,
        "eligible":False,
    }


def _holdout(rows, horizon):
    split=int(len(rows)*(1-FINAL_HOLDOUT_FRAC))
    dev=rows[:split]; hold=rows[split:]
    gap=int(PURGE_BARS[horizon]+EMBARGO_BARS[horizon])
    meta_end=len(dev)-gap; meta_start=max(MIN_TRAIN,meta_end-META_BLOCK)
    train=dev[:meta_start]; meta=dev[meta_start:meta_end]
    if len(train)<MIN_TRAIN or len(meta)<MIN_META_ROWS or len(hold)<200:
        return {"status":"DEFERRED","reason":"insufficient_holdout_training"}
    meta_models=_fit(train); meta_probs=_probs(meta_models,meta)
    hold_models=_fit(dev); hold_probs=_probs(hold_models,hold)
    threshold=_uncertainty_gate(_uncertainty(meta_probs))
    routed=_risk_router(meta_probs,meta,hold_probs,hold)
    gate=_uncertainty(hold_probs)>=threshold
    final=hold_probs["production"].copy(); final[gate]=routed[gate]
    y=[r["y"] for r in hold]
    base=metrics(y,hold_probs["soft_equal"]); cand=metrics(y,final)
    return {
        "status":"OK","n":len(hold),"gate_n":int(gate.sum()),"gate_rate":float(gate.mean()),
        "threshold":threshold,"baseline":base,"candidate":cand,
        "delta":{k:float(cand[k]-base[k]) for k in ("accuracy","logloss","brier","ece")},
        "high_uncertainty":{
            "baseline":metrics(list(np.asarray(y)[gate]),hold_probs["soft_equal"][gate]) if gate.any() else None,
            "candidate":metrics(list(np.asarray(y)[gate]),final[gate]) if gate.any() else None
        }
    }


def evaluate(horizon):
    rows=load_archive_research_rows(horizon,MAX_ROWS)
    if len(rows)<MIN_TRAIN+META_BLOCK+TEST_BLOCK+200:
        return {"status":"DEFERRED","reason":"insufficient_archive_rows","n":len(rows)}
    dev=_eval_dev(rows[:int(len(rows)*(1-FINAL_HOLDOUT_FRAC))],horizon)
    hold=_holdout(rows,horizon)
    return {
        "status":"OK","schema_version":1,"research_only":True,"production_changed":False,
        "final_holdout_protected":True,"final_holdout_used_for_selection":False,
        "strict_point_in_time_archive_replay":False,"horizon":horizon,"n":len(rows),
        "development":dev,"final_holdout":hold,
        "policy":"current-production-champion baseline + alternative-expert prequential reliability router activated only on prior-block high-uncertainty cases; low-uncertainty retains champion; no production promotion"
    }


def main():
    payload={"schema_version":1,"research_only":True,"production_changed":False,"horizons":{h:evaluate(h) for h in HORIZONS}}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
