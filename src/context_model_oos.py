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


def _validation_slices(rows):
    """Return two chronological validation index ranges from the pre-test window."""
    if len(rows) < MIN_SPECIALIST_TRAIN:
        return []
    split = max(int(len(rows) * 0.70), len(rows) - 150)
    tail_len = len(rows) - split
    if tail_len < 50:
        return []
    mid = max(25, tail_len // 2)
    ranges = [(split, split + mid), (split + mid, len(rows))]
    return [r for r in ranges if (r[1] - r[0]) >= 25]


def _score_factory(factory, fit_rows, valid_rows):
    """Score one factory on a chronological validation slice."""
    if len(fit_rows)<100 or len(valid_rows)<25:
        return None
    labels=[r["y"] for r in fit_rows]
    if len(set(labels))<3:
        return None
    try:
        model=factory()
        model.fit(np.asarray([r["x"] for r in fit_rows],dtype=float),np.asarray(labels))
        p=aligned_for_router(model,valid_rows)
        yi=np.asarray([("DOWN","FLAT","UP").index(r["y"]) for r in valid_rows])
        ll=float(-np.mean(np.log(np.clip(p[np.arange(len(yi)),yi],1e-12,1.0))))
        br=float(np.mean(np.sum((p-np.eye(3)[yi])**2,axis=1)))
        return ll,br
    except Exception:
        return None


def _select_factory(rows, factories: Mapping[str, Callable[[], object]]):
    """Select a specialist using multiple chronological validation slices.

    Selection uses only historical rows before the caller's test block. A model
    must be competitive across the recent slices; otherwise the router falls
    back to the global ensemble instead of chasing one noisy validation tail.
    """
    if len(rows) < MIN_SPECIALIST_TRAIN:
        return None, None
    slices=_validation_slices(rows)
    if len(slices)<2:
        return None, None
    scores={}
    for name,factory in factories.items():
        per=[]
        for start, end in slices:
            fit = rows[:start]
            valid = rows[start:end]
            score = _score_factory(factory, fit, valid)
            if score is not None:
                per.append(score)
        if len(per)!=len(slices):
            continue
        scores[name]=per
    if not scores:
        return None,None
    # Normalize each metric by the cross-model median on each validation slice.
    # This prevents log loss scale from dominating Brier and reduces dependence
    # on any single metric's absolute magnitude.
    total={}
    for name,per in scores.items():
        vals=[]
        for j in range(len(slices)):
            ll_med=float(np.median([v[j][0] for v in scores.values()]))
            br_med=float(np.median([v[j][1] for v in scores.values()]))
            vals.append(0.70*(per[j][0]/max(ll_med,1e-12))+0.30*(per[j][1]/max(br_med,1e-12)))
        total[name]=float(np.mean(vals))
    ordered=sorted(total.items(),key=lambda kv:kv[1])
    winner,win_score=ordered[0]
    # Require the winner not to be materially worse on either recent slice.
    for valid_idx in range(len(slices)):
        winner_ll,winner_br=scores[winner][valid_idx]
        for _,per in scores.items():
            if per[valid_idx][0] < winner_ll and per[valid_idx][1] < winner_br:
                winner_score=0.70*(winner_ll/max(float(np.median([v[valid_idx][0] for v in scores.values()])),1e-12))+0.30*(winner_br/max(float(np.median([v[valid_idx][1] for v in scores.values()])),1e-12))
                break
        else:
            continue
        # A dominated winner is rejected; caller uses the global fallback.
        if winner_score>1.03:
            return None,None
    return factories[winner],winner


def route_predictions(train_rows, test_rows, factories: Mapping[str, Callable[[], object]]):
    """Walk one already-defined chronological split using contextual specialists.

    A specialist is trained only when its context has enough historical rows.
    Otherwise the global model is used. No test labels are used for routing.
    """
    if len(train_rows) < MIN_SPECIALIST_TRAIN:
        return None

    thresholds = fit_context_thresholds(train_rows)
    # Freeze routing labels for both training and test from training-only thresholds.
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
    specialist_names = {}
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
        specialist_names[context] = specialist_name

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
        "specialist_models": specialist_names,
        "thresholds": thresholds,
    }


def dynamic_ensemble_weights(train_rows, factories, min_weight=0.10, temperature=1.0):
    """Estimate stable soft model weights from multiple pre-test validation slices."""
    if len(train_rows) < MIN_SPECIALIST_TRAIN:
        return {}
    slices=_validation_slices(train_rows)
    if len(slices)<2:
        return {}
    scores={}
    for name,factory in factories.items():
        per=[]
        for start, end in slices:
            fit = train_rows[:start]
            valid = train_rows[start:end]
            score = _score_factory(factory, fit, valid)
            if score is not None:
                per.append(score[0])
        if len(per)==len(slices):
            scores[name]=float(np.mean(per))
    if not scores:
        return {}
    raw={name:float(np.exp(-(score-min(scores.values()))/max(float(temperature),1e-6))) for name,score in scores.items()}
    total=sum(raw.values())
    if total<=0:
        return {}
    # Shrink toward uniform weights so a short/noisy regime cannot monopolize the ensemble.
    uniform=1.0/len(raw)
    weights={name:(1.0-0.25)*(value/total)+0.25*uniform for name,value in raw.items()}
    # Project onto the bounded simplex so every weight keeps its floor after
    # normalization as well (simple iterative water-filling for a small model pool).
    names = list(weights)
    values = np.asarray([float(weights[name]) for name in names], dtype=float)
    floor = float(min_weight)
    values = np.maximum(values, floor)
    for _ in range(16):
        delta = 1.0 - float(values.sum())
        if abs(delta) <= 1e-12:
            break
        if delta > 0:
            room = np.maximum(1.0 - values, 0.0)
            active = room > 1e-12
            if not np.any(active):
                break
            values[active] += delta * room[active] / room[active].sum()
        else:
            room = np.maximum(values - floor, 0.0)
            active = room > 1e-12
            if not np.any(active):
                break
            values[active] += delta * room[active] / room[active].sum()
    values = np.maximum(values, floor)
    values /= values.sum()
    return {name: float(value) for name, value in zip(names, values)}


def aligned_for_router(model, rows):
    probs=np.asarray(model.predict_proba(np.asarray([r["x"] for r in rows],dtype=float)),dtype=float)
    classes=list(model.classes_)
    out=np.full((len(rows),3),1e-6,dtype=float)
    names=("DOWN","FLAT","UP")
    for j,cls in enumerate(classes):
        if cls in names:
            out[:,names.index(cls)]=probs[:,j]
    return out/out.sum(axis=1,keepdims=True)


def _context_blend_weight(sample_count, weights, max_blend=0.30):
    """Scale specialist influence by pre-test support and ensemble stability.

    Smaller contexts and highly concentrated model weights get less influence.
    The global ensemble remains the default and the context specialist can
    never exceed ``max_blend``.
    """
    n = max(int(sample_count), 0)
    if n < MIN_SPECIALIST_TRAIN or not weights:
        return 0.0
    vals = np.asarray([max(float(v), 0.0) for v in weights.values()], dtype=float)
    total = float(vals.sum())
    if total <= 0:
        return 0.0
    p = vals / total
    entropy = float(-np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
    max_entropy = float(np.log(len(p))) if len(p) > 1 else 1.0
    stability = entropy / max_entropy if max_entropy > 0 else 1.0
    support = min(1.0, np.sqrt(n / 1000.0))
    return float(max_blend * support * stability)

def dynamic_route_predictions(train_rows, test_rows, factories):
    """Research-only soft routing: blend global and contextual specialists.

    Context thresholds, specialist selection and model weights are frozen from
    train_rows. If a specialist is unavailable, the global model remains the
    fallback. No test labels participate in any routing decision.
    """
    base=route_predictions(train_rows,test_rows,factories)
    if base is None:
        return None
    thresholds=base["thresholds"]
    global_weights=dynamic_ensemble_weights(train_rows,factories)
    if not global_weights:
        return base
    # Train one model per factory once on the complete pre-test window.
    models={}
    X=np.asarray([r["x"] for r in train_rows],dtype=float)
    y=np.asarray([r["y"] for r in train_rows])
    for name,factory in factories.items():
        try:
            m=factory(); m.fit(X,y); models[name]=m
        except Exception:
            pass
    if not models:
        return base
    global_mix=np.zeros((len(test_rows),3),dtype=float)
    for name,w in global_weights.items():
        if name in models:
            global_mix += w*aligned_for_router(models[name],test_rows)
    routed=global_mix.copy()
    context_weights={}
    for context in CONTEXTS:
        idx=[i for i,r in enumerate(test_rows) if context_of(r,thresholds)==context]
        if not idx: continue
        subset=[r for r in train_rows if context_of(r,thresholds)==context]
        if len(subset)<MIN_SPECIALIST_TRAIN or len({r["y"] for r in subset})<3:
            context_weights[context]={"specialist":False,"n":len(idx)}
            continue
        weights=dynamic_ensemble_weights(subset,factories)
        if not weights:
            context_weights[context]={"specialist":False,"n":len(idx)}
            continue
        # Specialists are blended with a conservative 70/30 global/context mix.
        Xs=np.asarray([r["x"] for r in subset],dtype=float); ys=np.asarray([r["y"] for r in subset])
        mix=np.zeros((len(idx),3),dtype=float)
        for name,w in weights.items():
            try:
                m=factories[name](); m.fit(Xs,ys)
                mix += w*aligned_for_router(m,[test_rows[i] for i in idx])
            except Exception:
                pass
        if np.any(mix):
            blend = _context_blend_weight(len(subset), weights)
            routed[idx]=(1.0-blend)*global_mix[idx]+blend*mix
            context_weights[context]={"specialist":True,"n":len(idx),"weights":weights,"blend":blend}
    routed/=routed.sum(axis=1,keepdims=True)
    return {
        **base,
        "routed_probs":routed,
        "routing_mode":"soft_dynamic_ensemble",
        "global_weights":global_weights,
        "context_weights":context_weights,
        "production_changed":False,
        "research_only":True,
    }

def evaluate_routing(y_true, routed_probs, global_probs):
    """Return descriptive metrics; never used for routing selection."""
    names=("DOWN","FLAT","UP")
    y=np.asarray([names.index(v) for v in y_true],dtype=int)
    def m(p):
        p=np.asarray(p,dtype=float)
        ll=float(-np.mean(np.log(np.clip(p[np.arange(len(y)),y],1e-12,1.0))))
        brier=float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1)))
        acc=float(np.mean(np.argmax(p,axis=1)==y))
        return {"accuracy":acc,"logloss":ll,"brier":brier,"n":int(len(y))}
    return {"global":m(global_probs),"routed":m(routed_probs)}


def strict_improvement(global_metrics, routed_metrics):
    """Promotion-style research gate; never call this from production."""
    if not global_metrics or not routed_metrics:
        return False
    return (
        routed_metrics["logloss"] <= global_metrics["logloss"] - 0.005
        and routed_metrics["brier"] <= global_metrics["brier"] - 0.002
        and routed_metrics["accuracy"] >= global_metrics["accuracy"] - 0.01
    )
