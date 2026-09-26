# V13 E2E trigger: source-level runtime regression verification
"""Maximum Future-Generalization Predictive Control System (research-only).

This module expands the v2 controller into a guarded v6 experiment. It keeps
all adaptive state prequential: current-block labels are never used to create
current routing inputs. Development OOS is used to learn meta policies; a
chronologically isolated final holdout receives frozen meta/calibration state.

Implemented research components:
- model disagreement + temporal dynamics
- error correlation / error diversity
- multi-dimensional predictability
- future model failure + time-to-failure hazards
- drift / regime / regime-transition features
- feature/source reliability
- information shock and prediction momentum/flip
- historical error/prototype retrieval
- meta-labeling
- uncertainty decomposition
- diversity-aware dynamic soft routing
- chronological calibration
- selective prediction + conformal prediction-set diagnostics
- counterfactual stability / adversarial perturbation robustness
- invariant-feature discovery
- hard-negative density
- multi-horizon consistency hook
- adaptive compute policy
- block-bootstrap statistical validation
- frozen-holdout / promotion gate / fallback evidence

No production artifact is changed.
"""
from __future__ import annotations

import math
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.innovative_control_layer_oos import (
        EXPERTS,
        SEED,
        FUTURE_WINDOW,
        PAST_WINDOW,
        _expert_factories,
        _fit_panel,
        _norm,
        _entropy,
        _temperature,
        _apply_temperature,
        _route_probs,
        _bootstrap_ci,
        _block_regime,
        _champion_benchmark,
    )
    from src.model_compare import load_archive_research_rows, metrics
    from src.prediction_policy_oos import evaluate_policy_case, summarize_policy_blocks, evaluate_policy_holdout_case
except ModuleNotFoundError:
    from innovative_control_layer_oos import (
        EXPERTS,
        SEED,
        FUTURE_WINDOW,
        PAST_WINDOW,
        _expert_factories,
        _fit_panel,
        _norm,
        _entropy,
        _temperature,
        _apply_temperature,
        _route_probs,
        _bootstrap_ci,
        _block_regime,
        _champion_benchmark,
    )
    from model_compare import load_archive_research_rows, metrics
    from prediction_policy_oos import evaluate_policy_case, summarize_policy_blocks, evaluate_policy_holdout_case

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "historical_research"

CLASSES = ("DOWN", "FLAT", "UP")
HORIZONS = ("5m", "10m")
MIN_TRAIN = 1800
TEST_BLOCK = 100
MAX_BLOCKS = 24
FINAL_HOLDOUT_FRAC = 0.20
MIN_META_SAMPLES = 10
RETRIEVAL_K = 5
BOOTSTRAP_REPS = 1000
EPS = 1e-8
EMBARGO_MINUTES = 60
CONFORMAL_ALPHA = 0.10
FAILURE_PRIOR_STRENGTH = 4.0


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _softmax(values):
    x = np.asarray(values, dtype=float)
    x -= np.max(x)
    y = np.exp(x)
    return y / max(float(y.sum()), EPS)


def _numeric_state(state):
    """Build the causal meta-state vector.

    Retrieval and meta-label values can be unavailable while constructing a
    bootstrap/query state (and retrieval itself must not recursively depend on
    its own output). Use explicit neutral priors only for those missing
    components; fully materialized OOS blocks retain their measured values.
    """
    d = state["disagreement"]
    drift = state["drift"]
    pred = state["predictability"]
    un = state["uncertainty"]
    retrieval = state.get("retrieval")
    if not isinstance(retrieval, dict):
        retrieval = {}
    meta_label = state.get("meta_label")
    if not isinstance(meta_label, dict):
        meta_label = {}
    return np.asarray([
        d["std_probability"],
        d["probability_range"],
        d["js_divergence"],
        d["pairwise_class_disagreement"],
        d["flip_rate"],
        d["disagreement_velocity"],
        d["disagreement_acceleration"],
        state["error_correlation"]["mean_abs_error_correlation"],
        drift["feature_drift"],
        drift["prediction_drift"],
        drift["drift_score"],
        state["feature_reliability"]["global"],
        state["source_reliability"],
        state["information_shock"]["shock_score"],
        state["prediction_momentum"]["velocity"],
        state["prediction_momentum"]["acceleration"],
        pred["global"],
        pred["velocity"],
        pred["acceleration"],
        state["regime_transition"]["stay_probability"],
        float(retrieval.get("failure_similarity", 0.5)),
        un["total"],
        float(meta_label.get("reliability", 0.5)),
        state["hard_negative_density"],
    ], dtype=float)


def _panel_stats(panel, prior_panel=None):
    matrix = np.stack([panel[e] for e in EXPERTS], axis=0)
    mean_p = _norm(np.mean(matrix, axis=0))
    std = matrix.std(axis=0)
    top = np.argmax(matrix, axis=2)
    majority = np.argmax(mean_p, axis=1)
    agreement = float(np.mean(top == majority))
    pairwise = []
    l1 = []
    js = []
    for i in range(len(EXPERTS)):
        for j in range(i + 1, len(EXPERTS)):
            a, b = matrix[i], matrix[j]
            pairwise.append(float(np.mean(np.argmax(a, axis=1) != np.argmax(b, axis=1))))
            l1.append(float(np.mean(np.sum(np.abs(a - b), axis=1))))
            m = 0.5 * (a + b)
            js.append(float(0.5 * np.mean(np.sum(a * np.log(np.clip(a / m, EPS, 1e12)), axis=1))
                           + 0.5 * np.mean(np.sum(b * np.log(np.clip(b / m, EPS, 1e12)), axis=1))))
    current_disagreement = float(np.mean(std))
    previous = None
    if prior_panel:
        previous = _panel_stats(prior_panel)["current_disagreement"]
    velocity = 0.0 if previous is None else current_disagreement - previous
    acceleration = 0.0
    entropy = float(np.mean([_entropy(row) for row in mean_p]))
    margin = np.sort(mean_p, axis=1)[:, -1] - np.sort(mean_p, axis=1)[:, -2]
    flip_rate = 0.0
    if prior_panel:
        old_mean = _norm(np.mean(np.stack([prior_panel[e] for e in EXPERTS], axis=0), axis=0))
        n = min(len(old_mean), len(mean_p))
        flip_rate = float(np.mean(np.argmax(old_mean[-n:], axis=1) != np.argmax(mean_p[-n:], axis=1)))
    return {
        "mean_probability": mean_p.mean(axis=0).tolist(),
        "std_probability": float(np.mean(std)),
        "min_probability": float(matrix.min(axis=(0, 2)).mean()),
        "max_probability": float(matrix.max(axis=(0, 2)).mean()),
        "probability_range": float((matrix.max(axis=0) - matrix.min(axis=0)).mean()),
        "prediction_entropy": entropy,
        "top_class_agreement_rate": agreement,
        "majority_margin": float(np.mean(margin)),
        "pairwise_class_disagreement": float(np.mean(pairwise)) if pairwise else 0.0,
        "l1_distance": float(np.mean(l1)) if l1 else 0.0,
        "js_divergence": float(np.mean(js)) if js else 0.0,
        "current_disagreement": current_disagreement,
        "disagreement_velocity": velocity,
        "disagreement_acceleration": acceleration,
        "flip_rate": flip_rate,
    }


def _error_corr(blocks):
    if not blocks:
        return {"mean_abs_error_correlation": 0.0, "by_expert": {e: 0.0 for e in EXPERTS}}
    per = {e: [] for e in EXPERTS}
    for b in blocks[-6:]:
        for e in EXPERTS:
            y = np.asarray(b["errors"][e], dtype=float)
            per[e].extend(y.tolist())
    out = {}
    for e in EXPERTS:
        vals = []
        a = np.asarray(per[e], dtype=float)
        for other in EXPERTS:
            if other == e:
                continue
            b = np.asarray(per[other], dtype=float)
            if len(a) < 4 or np.std(a) <= EPS or np.std(b) <= EPS:
                vals.append(0.0)
            else:
                vals.append(abs(float(np.corrcoef(a, b)[0, 1])))
        out[e] = float(np.mean(vals)) if vals else 0.0
    return {
        "mean_abs_error_correlation": float(np.mean(list(out.values()))),
        "by_expert": out,
    }


def _feature_reliability(current_rows, prior_rows):
    if not current_rows:
        return {"global": 0.0, "per_feature": []}
    a = np.asarray([r["x"] for r in current_rows], dtype=float)
    if not prior_rows:
        vals = np.ones(a.shape[1], dtype=float)
    else:
        b = np.asarray([r["x"] for r in prior_rows], dtype=float)
        scale = np.std(b, axis=0) + 1e-8
        shift = np.abs(a.mean(axis=0) - b.mean(axis=0)) / scale
        finite = np.mean(np.isfinite(a), axis=0)
        vals = np.clip(1.0 - 0.20 * np.minimum(shift, 5.0), 0.0, 1.0) * finite
    return {"global": float(np.mean(vals)), "per_feature": vals.tolist()}


def _source_reliability(rows, prior_blocks):
    sources = [str(r.get("data_source", "unknown")) for r in rows]
    source = max(set(sources), key=sources.count) if sources else "unknown"
    historical = {}
    for b in prior_blocks:
        for src, pair in b.get("source_outcomes", {}).items():
            historical.setdefault(src, []).extend(pair)
    vals = historical.get(source, [])
    rate = (sum(vals) + 3.0) / (len(vals) + 6.0) if vals else 0.85
    return {
        "source": source,
        "reliability": float(np.clip(rate, 0.0, 1.0)),
        "historical_n": len(vals),
    }


def _information_shock(current_rows, prior_rows):
    if not current_rows or not prior_rows:
        return {
            "event_rate": 0.0,
            "update_rate": 0.0,
            "information_entropy": 0.0,
            "shock_score": 0.0,
        }
    a = np.asarray([r["x"] for r in current_rows], dtype=float)
    b = np.asarray([r["x"] for r in prior_rows], dtype=float)
    z = np.abs(a.mean(axis=0) - b.mean(axis=0)) / (np.std(b, axis=0) + 1e-8)
    shock = float(np.clip(np.mean(np.minimum(z, 6.0)) / 6.0, 0.0, 1.0))
    entropy = float(np.mean(np.abs(a).mean(axis=1) > np.quantile(np.abs(b).mean(axis=1), 0.90)))
    return {
        "event_rate": float(min(1.0, entropy)),
        "update_rate": float(min(1.0, shock * 1.2)),
        "information_entropy": float(np.mean(np.minimum(z, 6.0)) / 6.0),
        "shock_score": float(np.clip(0.45 * shock + 0.55 * entropy, 0.0, 1.0)),
    }


def _prediction_momentum(panel, prior_panel, prior2_panel):
    current = _norm(np.mean(np.stack([panel[e] for e in EXPERTS], axis=0), axis=0))
    previous = _norm(np.mean(np.stack([prior_panel[e] for e in EXPERTS], axis=0), axis=0)) if prior_panel else current
    prev2 = _norm(np.mean(np.stack([prior2_panel[e] for e in EXPERTS], axis=0), axis=0)) if prior2_panel else previous
    n = min(len(current), len(previous))
    velocity = float(np.mean(np.linalg.norm(current[-n:] - previous[-n:], axis=1)))
    acceleration = float(
        velocity - np.mean(np.linalg.norm(previous[-n:] - prev2[-n:], axis=1))
    ) if len(previous) else 0.0
    persistence = float(np.mean(np.argmax(current[-n:], axis=1) == np.argmax(previous[-n:], axis=1)))
    return {
        "velocity": velocity,
        "acceleration": acceleration,
        "persistence": persistence,
        "reversal": float(max(0.0, -acceleration)),
    }


def _regime_transition(current_regime, blocks):
    transitions = {}
    for a, b in zip(blocks[:-1], blocks[1:]):
        key = a["regime"]
        transitions.setdefault(key, {})
        transitions[key][b["regime"]] = transitions[key].get(b["regime"], 0) + 1
    row = transitions.get(current_regime, {})
    total = sum(row.values())
    if total == 0:
        probs = {r: 0.0 for r in ("TREND", "RANGE")}
        probs[current_regime] = 1.0
    else:
        probs = {k: v / total for k, v in row.items()}
        for r in ("TREND", "RANGE"):
            probs.setdefault(r, 0.0)
    next_regime = max(probs, key=probs.get)
    return {
        "current": current_regime,
        "next_probability": probs,
        "next_regime": next_regime,
        "stay_probability": float(probs.get(current_regime, 0.0)),
    }


def _safe_binary(X, y):
    if len(y) < MIN_META_SAMPLES or len(set(map(int, y))) < 2:
        return None
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.25, max_iter=2000, class_weight="balanced")),
    ])
    model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
    return model


def _smoothed_binary_rate(labels, strength=FAILURE_PRIOR_STRENGTH):
    """Prequential empirical prior with explicit shrinkage toward 0.5."""
    n = len(labels)
    if n == 0:
        return 0.5
    return float((sum(map(int, labels)) + 0.5 * strength) / (n + strength))


def _meta_training(blocks, before_index):
    usable = blocks[:before_index]
    predict_pairs = []
    failure_pairs = {e: [] for e in EXPERTS}
    meta_pairs = []
    hazard_pairs = {h: [] for h in (1, 2, 3)}
    for i, b in enumerate(usable):
        state_vec = _numeric_state(b["state"])
        if i + 1 < len(usable):
            next_block = usable[i + 1]
            next_acc = float(next_block["soft"]["accuracy"])
            predict_pairs.append((state_vec, int(next_acc >= 0.45)))
        correct = b["soft_correct"]
        meta_pairs.extend((state_vec, int(v)) for v in correct.tolist())
        for e in EXPERTS:
            if i + FUTURE_WINDOW < len(usable):
                future = usable[i + 1:min(len(usable), i + 1 + FUTURE_WINDOW)]
                past = usable[max(0, i - PAST_WINDOW + 1):i + 1]
                if past and len(future) >= FUTURE_WINDOW:
                    p_ll = float(np.mean([x["metrics"][e]["logloss"] for x in past]))
                    f_ll = float(np.mean([x["metrics"][e]["logloss"] for x in future]))
                    p_acc = float(np.mean([x["metrics"][e]["accuracy"] for x in past]))
                    f_acc = float(np.mean([x["metrics"][e]["accuracy"] for x in future]))
                    failure = int((f_ll - p_ll) >= 0.05 or (f_acc - p_acc) <= -0.05)
                    failure_pairs[e].append((
                        np.concatenate([state_vec, [float(b["metrics"][e]["logloss"])]])
                        , failure
                    ))
                    for h in (1, 2, 3):
                        ff = future[:h]
                        within = int(any(
                            (x["metrics"][e]["logloss"] - p_ll) >= 0.05
                            or (x["metrics"][e]["accuracy"] - p_acc) <= -0.05
                            for x in ff
                        ))
                        hazard_pairs[h].append((state_vec, within))
    predict_model = _safe_binary([x for x, _ in predict_pairs], [y for _, y in predict_pairs])
    meta_model = _safe_binary([x for x, _ in meta_pairs], [y for _, y in meta_pairs])
    failure_models = {
        e: _safe_binary([x for x, _ in failure_pairs[e]], [y for _, y in failure_pairs[e]])
        for e in EXPERTS
    }
    hazard_models = {
        h: _safe_binary([x for x, _ in hazard_pairs[h]], [y for _, y in hazard_pairs[h]])
        for h in (1, 2, 3)
    }
    return predict_model, meta_model, failure_models, hazard_models, {
        "predictability_samples": len(predict_pairs),
        "meta_label_samples": len(meta_pairs),
        "failure_samples": {e: len(failure_pairs[e]) for e in EXPERTS},
        "hazard_samples": {str(h): len(hazard_pairs[h]) for h in (1, 2, 3)},
        "failure_prior_by_expert": {
            e: _smoothed_binary_rate([y for _, y in failure_pairs[e]])
            for e in EXPERTS
        },
        "hazard_prior_by_horizon": {
            str(h): _smoothed_binary_rate([y for _, y in hazard_pairs[h]])
            for h in (1, 2, 3)
        },
    }


def _predictability(state, predict_model, history):
    base = {
        "data": state["feature_reliability"]["global"],
        "model": 1.0 - state["disagreement"]["std_probability"],
        "information": 1.0 - state["information_shock"]["shock_score"],
        "regime": state["regime_transition"]["stay_probability"],
        "temporal": state["prediction_momentum"]["persistence"],
    }
    if predict_model is None:
        global_score = float(np.mean(list(base.values())))
    else:
        x = _numeric_state(state).reshape(1, -1)
        global_score = float(predict_model.predict_proba(x)[0, 1])
    old = history[-1]["state"]["predictability"]["global"] if history else global_score
    old2 = history[-2]["state"]["predictability"]["global"] if len(history) >= 2 else old
    velocity = global_score - old
    acceleration = velocity - (old - old2)
    return {
        "global": float(np.clip(global_score, 0.0, 1.0)),
        "data": float(np.clip(base["data"], 0.0, 1.0)),
        "model": float(np.clip(base["model"], 0.0, 1.0)),
        "information": float(np.clip(base["information"], 0.0, 1.0)),
        "regime": float(np.clip(base["regime"], 0.0, 1.0)),
        "temporal": float(np.clip(base["temporal"], 0.0, 1.0)),
        "velocity": float(velocity),
        "acceleration": float(acceleration),
    }


def _failure_state(state, quality, failure_models, hazard_models, priors=None):
    priors = priors if isinstance(priors, dict) else {}
    expert_priors = priors.get("failure_prior_by_expert", {})
    hazard_priors = priors.get("hazard_prior_by_horizon", {})
    risks = {}
    for e in EXPERTS:
        m = failure_models.get(e)
        if m is None:
            risks[e] = float(np.clip(float(expert_priors.get(e, 0.5)), 0.0, 1.0))
        else:
            x = np.concatenate([_numeric_state(state), [quality[e]]]).reshape(1, -1)
            risks[e] = float(m.predict_proba(x)[0, 1])
    hazard = {}
    x = _numeric_state(state).reshape(1, -1)
    for h in (1, 2, 3):
        m = hazard_models.get(h)
        hazard[h] = (
            float(m.predict_proba(x)[0, 1])
            if m is not None
            else float(np.clip(float(hazard_priors.get(str(h), 0.5)), 0.0, 1.0))
        )
    cumulative = [0.0, hazard[1], max(hazard[1], hazard[2]), max(hazard[2], hazard[3])]
    p1 = np.clip(cumulative[1], 0.0, 1.0)
    p2 = np.clip(cumulative[2] - cumulative[1], 0.0, 1.0)
    p3 = np.clip(cumulative[3] - cumulative[2], 0.0, 1.0)
    remaining = max(0.0, 1.0 - p1 - p2 - p3)
    expected = 1.0 * p1 + 2.0 * p2 + 3.0 * p3 + 4.0 * remaining
    return {
        "by_expert": risks,
        "mean": float(np.mean(list(risks.values()))),
        "max": float(max(risks.values())),
        "hazard_within_1_2_3": hazard,
        "expected_time_to_failure_blocks": float(expected),
    }


def _meta_label(state, meta_model):
    if meta_model is None:
        return {"probability_correct": 0.5, "reliability": 0.5}
    x = _numeric_state(state).reshape(1, -1)
    p = float(meta_model.predict_proba(x)[0, 1])
    return {"probability_correct": p, "reliability": p}


def _retrieval(state, blocks):
    if not blocks:
        return {"failure_similarity": 0.0, "success_similarity": 0.0, "probability": [1/3, 1/3, 1/3], "neighbors": []}
    q = _numeric_state(state)
    candidates = []
    for idx, b in enumerate(blocks[:-1]):
        v = _numeric_state(b["state"])
        dist = float(np.linalg.norm(q - v))
        next_block = blocks[idx + 1]
        yp = np.zeros(3, dtype=float)
        for y in next_block["y"]:
            yp[CLASSES.index(y)] += 1.0
        yp /= max(float(yp.sum()), 1.0)
        failure = 1.0 - next_block["soft"]["accuracy"]
        similarity = 1.0 / (1.0 + dist)
        candidates.append((similarity, failure, yp, idx))
    candidates.sort(reverse=True, key=lambda z: z[0])
    chosen = candidates[:RETRIEVAL_K]
    if not chosen:
        return {"failure_similarity": 0.0, "success_similarity": 0.0, "probability": [1/3, 1/3, 1/3], "neighbors": []}
    weights = np.asarray([x[0] for x in chosen], dtype=float)
    weights /= max(float(weights.sum()), EPS)
    probs = np.sum(np.stack([x[2] for x in chosen], axis=0) * weights[:, None], axis=0)
    failure_sim = float(np.sum(weights * np.asarray([x[1] for x in chosen])))
    return {
        "failure_similarity": failure_sim,
        "success_similarity": float(1.0 - failure_sim),
        # Persist plain Python values at the JSON artifact boundary.
        "probability": _norm(probs).reshape(-1).tolist(),
        "neighbors": [{"block": int(x[3]), "similarity": float(x[0]), "failure_rate": float(x[1])} for x in chosen],
    }


def _uncertainty(state):
    data_u = 1.0 - state["feature_reliability"]["global"]
    model_u = state["disagreement"]["std_probability"]
    distribution_u = state["drift"]["drift_score"]
    information_u = state["information_shock"]["shock_score"]
    irreducible = float(np.clip(
        state["disagreement"]["prediction_entropy"] * 0.55
        + (1.0 - state["meta_label"]["reliability"]) * 0.45,
        0.0, 1.0
    ))
    total = float(np.clip(np.mean([data_u, model_u, distribution_u, information_u, irreducible]), 0.0, 1.0))
    return {
        "data": data_u,
        "model": model_u,
        "distribution": distribution_u,
        "information": information_u,
        "irreducible": irreducible,
        "total": total,
    }


def _route_weights(state, quality, failure_state, *, use_disagreement=True, use_predictability=True,
                   use_failure=True, use_drift=True, use_error_correlation=True,
                   use_retrieval=True, previous=None):
    scores = []
    mean_failure = failure_state["by_expert"]
    corr = state["error_correlation"]["by_expert"]
    for e in EXPERTS:
        s = -quality[e]
        if use_failure:
            s += -1.8 * mean_failure[e]
        if use_error_correlation:
            s += -0.9 * corr[e]
        scores.append(s)
    q = _softmax(scores)
    adaptive_strength = 1.0
    if use_predictability:
        adaptive_strength *= 0.35 + 0.65 * state["predictability"]["global"]
    if use_disagreement:
        adaptive_strength *= 0.45 + 0.55 * (1.0 - state["disagreement"]["pairwise_class_disagreement"])
    if use_drift:
        adaptive_strength *= 0.35 + 0.65 * (1.0 - state["drift"]["drift_score"])
    adaptive_strength *= 0.35 + 0.65 * (1.0 - state["uncertainty"]["total"])
    source_reliability = state["source_reliability"]
    if isinstance(source_reliability, dict):
        source_reliability = source_reliability.get("reliability", 0.0)
    adaptive_strength *= 0.35 + 0.65 * float(source_reliability)
    if use_retrieval:
        retrieval = state.get("retrieval")
        if not isinstance(retrieval, dict):
            # Explicit neutral prior for bootstrap/query states where retrieval
            # is not yet materialized. Never treats missing retrieval as evidence.
            retrieval = {"success_similarity": 0.5}
        adaptive_strength *= 0.70 + 0.30 * float(
            retrieval.get("success_similarity", 0.5)
        )
    adaptive_strength = float(np.clip(adaptive_strength, 0.15, 1.0))
    w = adaptive_strength * q + (1.0 - adaptive_strength) * np.full(len(EXPERTS), 1.0 / len(EXPERTS))
    if previous is not None:
        w = 0.75 * np.asarray(previous) + 0.25 * w
    w = np.clip(w, 0.03, 0.90)
    return w / w.sum()


def _counterfactual_stability(models, X):
    if not models or len(X) == 0:
        return {"status": "DEFERRED", "score": None}
    rng = np.random.default_rng(SEED)
    shifts = []
    for e, model in models.items():
        base = _norm(model.predict_proba(X))
        sigma = np.std(X, axis=0) + 1e-8
        pert = X + rng.normal(0.0, 0.01, size=X.shape) * sigma
        cf = _norm(model.predict_proba(pert))
        shifts.append(float(np.mean(np.sum(np.abs(base - cf), axis=1))))
    shift = float(np.mean(shifts))
    return {"status": "OK", "mean_probability_shift": shift, "stability": float(np.clip(1.0 - shift, 0.0, 1.0))}


def _invariant_features(blocks, n_features):
    if len(blocks) < 4:
        return {"status": "DEFERRED", "stable_features": []}
    scores = []
    y_map = {"DOWN": -1.0, "FLAT": 0.0, "UP": 1.0}
    for j in range(n_features):
        signs = []
        mags = []
        for b in blocks[-12:]:
            x = np.asarray(b["x"], dtype=float)[:, j]
            y = np.asarray([y_map[v] for v in b["y"]], dtype=float)
            if np.std(x) <= EPS or np.std(y) <= EPS:
                continue
            corr = float(np.corrcoef(x, y)[0, 1])
            if math.isfinite(corr):
                signs.append(np.sign(corr))
                mags.append(abs(corr))
        if not signs:
            continue
        sign_consistency = abs(float(np.mean(signs)))
        scores.append((0.65 * float(np.mean(mags)) + 0.35 * sign_consistency, j, sign_consistency))
    scores.sort(reverse=True)
    return {
        "status": "OK",
        "stable_features": [
            {"feature_index": int(j), "score": float(score), "sign_consistency": float(cons)}
            for score, j, cons in scores[:8]
        ],
    }


def _hard_negative_density(blocks):
    vals = []
    for b in blocks[-6:]:
        vals.extend([1.0 - float(x) for x in b["soft_correct"]])
    return float(np.mean(vals)) if vals else 0.5


def _conformal_from_blocks(blocks):
    nonconf = []
    for b in blocks[-8:]:
        p = np.asarray(b["calibrated"], dtype=float)
        for i, y in enumerate(b["y"]):
            nonconf.append(1.0 - float(p[i, CLASSES.index(y)]))
    if len(nonconf) < 30:
        return {"status": "DEFERRED", "q": None}
    q = float(np.quantile(np.asarray(nonconf), 1.0 - CONFORMAL_ALPHA, method="higher"))
    return {"status": "OK", "q": q, "alpha": CONFORMAL_ALPHA, "n": len(nonconf)}


def _conformal_eval(probs, ys, conf):
    if conf.get("q") is None:
        return {"coverage": None, "mean_set_size": None}
    q = float(conf["q"])
    sizes = []
    covered = []
    for p, y in zip(probs, ys):
        included = np.where(1.0 - p <= q)[0]
        sizes.append(len(included))
        covered.append(CLASSES.index(y) in included)
    return {"coverage": float(np.mean(covered)), "mean_set_size": float(np.mean(sizes)), "alpha": conf["alpha"]}


def _selective_metrics(blocks):
    scores = np.asarray([v for b in blocks for v in b["selective_score_rows"]], dtype=float)
    if len(scores) < 20:
        return {"status": "DEFERRED"}
    result = {}
    for target in (1.0, 0.95, 0.90, 0.80, 0.70):
        threshold = float(np.quantile(scores, max(0.0, 1.0 - target)))
        ps, ys = [], []
        for b in blocks:
            p = np.asarray(b["calibrated"], dtype=float)
            s = np.asarray(b["selective_score_rows"], dtype=float)
            keep = s >= threshold
            ps.extend(p[keep])
            ys.extend([y for y, k in zip(b["y"], keep) if k])
        m = metrics(ys, np.asarray(ps)) if ys else None
        result[f"{int(target*100)}%"] = {
            "target_coverage": target,
            "threshold": threshold,
            "coverage": float(len(ys) / max(1, sum(len(b["y"]) for b in blocks))),
            "n": len(ys),
            "accuracy": None if m is None else m["accuracy"],
            "logloss": None if m is None else m["logloss"],
            "brier": None if m is None else m["brier"],
            "ece": None if m is None else m["calibration_error"],
        }
    return result


def _stress(block, models):
    X = np.asarray(block["X"], dtype=float)
    y = block["y"]
    rng = np.random.default_rng(SEED + int(block["index"]))
    base_probs = block["calibrated"]
    variants = {}
    noise = X + rng.normal(0.0, 0.05, size=X.shape)
    variants["feature_noise"] = noise
    drop = X.copy()
    cols = rng.choice(X.shape[1], max(1, X.shape[1] // 10), replace=False)
    drop[:, cols] = 0.0
    variants["feature_dropout"] = drop
    shifts = {}
    for name, xt in variants.items():
        panel = {}
        for e, model in models.items():
            panel[e] = _norm(model.predict_proba(xt))
        probs = _norm(np.mean(np.stack(list(panel.values()), axis=0), axis=0))
        m = metrics(y, probs)
        shifts[name] = {
            "accuracy": m["accuracy"],
            "logloss": m["logloss"],
            "brier": m["brier"],
            "ece": m["calibration_error"],
            "probability_shift": float(np.mean(np.sum(np.abs(probs - base_probs), axis=1))),
        }
    return {"status": "OK", "scenarios": shifts}


def _binary_calibration(scores, outcomes):
    if len(scores) < 8:
        return {"status": "DEFERRED", "n": len(scores)}
    p = np.clip(np.asarray(scores, dtype=float), EPS, 1.0 - EPS)
    y = np.asarray(outcomes, dtype=float)
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    ece = 0.0
    for i in range(5):
        lo, hi = i / 5.0, (i + 1) / 5.0
        mask = (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))
        if mask.any():
            ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return {"status":"OK","n":int(len(y)),"brier":brier,"logloss":logloss,"ece":float(ece)}


def _compute_tier(score, drift, failure):
    if drift >= 0.75 or failure >= 0.75:
        return "HARD_STRESS"
    if score < 0.35:
        return "ENSEMBLE"
    if score < 0.65:
        return "ADAPTIVE_ENSEMBLE"
    return "STANDARD"


def _fit_and_predict(train_rows, test_rows):
    panel, models = _fit_panel(train_rows, test_rows)
    return panel, models


def _causal_current_snapshot(rows, panel):
    """Return only the first prediction-time snapshot from an OOS block.

    Block-level routing must not inspect later observations inside the same
    test block. Using the first row/probability vector keeps the state
    generation point-in-time safe while preserving the block OOS structure.
    """
    if not rows:
        raise ValueError("empty_current_block")
    if any(e not in panel for e in EXPERTS):
        raise ValueError("incomplete_model_panel")
    current_rows = [rows[0]]
    current_panel = {}
    for expert in EXPERTS:
        arr = np.asarray(panel[expert], dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) < 1:
            raise ValueError(f"invalid_panel_for:{expert}")
        current_panel[expert] = arr[:1].copy()
    return current_rows, current_panel


def _evaluate_variant(state, panel, quality, failure_state, previous_weights, name, retrieval_mix=0.0):
    specs = {
        "soft_ensemble": dict(use_disagreement=False, use_predictability=False, use_failure=False, use_drift=False, use_error_correlation=False, use_retrieval=False),
        "adaptive_ensemble": dict(use_disagreement=False, use_predictability=False, use_failure=False, use_drift=False, use_error_correlation=True, use_retrieval=False),
        "disagreement_model": dict(use_disagreement=True, use_predictability=False, use_failure=False, use_drift=False, use_error_correlation=False, use_retrieval=False),
        "predictability_model": dict(use_disagreement=False, use_predictability=True, use_failure=False, use_drift=False, use_error_correlation=False, use_retrieval=False),
        "future_failure_predictor": dict(use_disagreement=False, use_predictability=False, use_failure=True, use_drift=False, use_error_correlation=True, use_retrieval=False),
        "drift_aware_router": dict(use_disagreement=False, use_predictability=False, use_failure=False, use_drift=True, use_error_correlation=False, use_retrieval=False),
        "three_layers": dict(use_disagreement=True, use_predictability=True, use_failure=True, use_drift=False, use_error_correlation=True, use_retrieval=False),
        "three_layers_error_corr": dict(use_disagreement=True, use_predictability=True, use_failure=True, use_drift=False, use_error_correlation=True, use_retrieval=False),
        "three_layers_regime": dict(use_disagreement=True, use_predictability=True, use_failure=True, use_drift=True, use_error_correlation=True, use_retrieval=False),
        "three_layers_retrieval": dict(use_disagreement=True, use_predictability=True, use_failure=True, use_drift=True, use_error_correlation=True, use_retrieval=True),
        "full_architecture": dict(use_disagreement=True, use_predictability=True, use_failure=True, use_drift=True, use_error_correlation=True, use_retrieval=True),
    }
    if name == "soft_ensemble":
        w = np.full(len(EXPERTS), 1.0 / len(EXPERTS))
    else:
        w = _route_weights(state, quality, failure_state, previous=previous_weights, **specs[name])
    probs = _route_probs(panel, w)
    if retrieval_mix > 0.0:
        retrieval = state.get("retrieval")
        rp = (
            np.asarray(retrieval.get("probability"), dtype=float)
            if isinstance(retrieval, dict) and retrieval.get("probability") is not None
            else np.full(3, 1.0 / 3.0, dtype=float)
        )
        probs = _norm((1.0 - retrieval_mix) * probs + retrieval_mix * rp)
    return probs, w


def _purged_train(rows, test_start):
    start_dt = datetime.fromisoformat(str(test_start).replace("Z", "+00:00"))
    cutoff = start_dt - timedelta(minutes=EMBARGO_MINUTES)
    out = []
    for r in rows:
        try:
            c = datetime.fromisoformat(str(r["created"]).replace("Z", "+00:00"))
            t = datetime.fromisoformat(str(r["target"]).replace("Z", "+00:00"))
        except Exception:
            continue
        if c < cutoff and t < cutoff and c < t:
            out.append(r)
    return out


def _window_points(n):
    raw = list(range(MIN_TRAIN, n - TEST_BLOCK + 1, TEST_BLOCK))
    if len(raw) <= MAX_BLOCKS:
        return raw
    idx = np.linspace(0, len(raw) - 1, num=MAX_BLOCKS).astype(int)
    return [raw[i] for i in sorted(set(idx.tolist()))]


def _development(rows):
    points = _window_points(len(rows))
    holdout_row = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    dev_rows = rows[:holdout_row]
    points = [p for p in points if p + TEST_BLOCK <= len(dev_rows)]
    blocks = []
    previous_weights = None
    previous_policy_strategy = None
    previous_policy_predictability = None

    for start in points:
        test = dev_rows[start:start + TEST_BLOCK]
        if len(test) < TEST_BLOCK:
            continue
        train = _purged_train(dev_rows[:start], test[0]["created"])
        if len(train) < MIN_TRAIN:
            continue

        panel, models = _fit_and_predict(train, test)
        current_rows, current_panel = _causal_current_snapshot(test, panel)
        raw_metrics = {e: metrics([r["y"] for r in test], panel[e]) for e in EXPERTS}
        soft_probs = _norm(np.mean(np.stack(list(panel.values()), axis=0), axis=0))
        soft_metrics = metrics([r["y"] for r in test], soft_probs)
        prior_panel = blocks[-1]["panel"] if blocks else None
        prior2_panel = blocks[-2]["panel"] if len(blocks) >= 2 else None
        d = _panel_stats(current_panel, prior_panel)
        prior_rows = dev_rows[max(0, start - TEST_BLOCK):start]
        f_rel = _feature_reliability(current_rows, prior_rows)
        s_rel = _source_reliability(current_rows, blocks)
        i_shock = _information_shock(current_rows, prior_rows)
        p_momentum = _prediction_momentum(current_panel, prior_panel, prior2_panel)
        regime = _block_regime(current_rows)
        r_trans = _regime_transition(regime, blocks)
        state = {
            "disagreement": d,
            "error_correlation": _error_corr(blocks),
            "feature_reliability": f_rel,
            "source_reliability": s_rel["reliability"],
            "source_reliability_detail": s_rel,
            "information_shock": i_shock,
            "prediction_momentum": p_momentum,
            "regime_transition": r_trans,
            "hard_negative_density": _hard_negative_density(blocks),
            "predictability": {"global": 0.5, "velocity": 0.0, "acceleration": 0.0},
            "retrieval": {"failure_similarity": 0.5, "success_similarity": 0.5, "probability": [1/3,1/3,1/3], "neighbors": []},
            "meta_label": {"probability_correct": 0.5, "reliability": 0.5},
            "drift": {
                "feature_drift": float(1.0 - f_rel["global"]),
                "prediction_drift": float(i_shock["update_rate"]),
                "drift_score": float(np.clip(0.55 * (1.0 - f_rel["global"]) + 0.45 * i_shock["update_rate"], 0.0, 1.0)),
            },
            "uncertainty": {"total": 0.5},
        }
        predict_model, meta_model, failure_models, hazard_models, meta_stats = _meta_training(blocks, len(blocks))
        state["predictability"] = _predictability(state, predict_model, blocks)
        state["meta_label"] = _meta_label(state, meta_model)
        state["retrieval"] = _retrieval(state, blocks)
        state["uncertainty"] = _uncertainty(state)
        quality = {
            e: float(np.mean([b["metrics"][e]["logloss"] for b in blocks[-PAST_WINDOW:]]))
            if blocks else math.log(3.0)
            for e in EXPERTS
        }
        # Current-block labels are not used in quality/failure/meta state.
        failure_state = _failure_state(state, quality, failure_models, hazard_models, meta_stats)
        route_specs = [
            "soft_ensemble", "adaptive_ensemble", "disagreement_model",
            "predictability_model", "future_failure_predictor",
            "drift_aware_router", "three_layers", "three_layers_error_corr",
            "three_layers_regime", "three_layers_retrieval", "full_architecture",
        ]
        variants = {}
        for name in route_specs:
            mix = 0.10 if name in {"three_layers_retrieval", "full_architecture"} else 0.0
            p, w = _evaluate_variant(
                state, panel, quality, failure_state, previous_weights,
                name, retrieval_mix=mix,
            )
            variants[name] = {"probs": p, "weights": w}
        full_raw = variants["full_architecture"]["probs"]
        # Calibration uses only already-resolved OOS blocks.
        prior_cal_p = [np.asarray(b["calibrated"], dtype=float) for b in blocks[-8:]]
        prior_cal_y = [y for b in blocks[-8:] for y in b["y"]]
        if prior_cal_p and len(prior_cal_y) >= 50:
            t = _temperature(np.vstack(prior_cal_p), prior_cal_y)
            calibrated = _apply_temperature(full_raw, t)
        else:
            t = 1.0
            calibrated = full_raw

        invariant = _invariant_features(blocks, len(test[0]["x"]))
        cf = _counterfactual_stability(
            models, np.asarray([current_rows[0]["x"]], dtype=float)
        )
        causal_cf = {
            **cf,
            "instability": (
                float(np.clip(1.0 - float(cf.get("stability", 1.0)), 0.0, 1.0))
                if cf.get("stability") is not None else None
            ),
        }
        state["counterfactual_stability"] = causal_cf
        b = {
            "index": len(blocks),
            "test_start": test[0]["created"],
            "test_end": test[-1]["created"],
            "n": len(test),
            "regime": regime,
            "metrics": raw_metrics,
            "soft": soft_metrics,
            "soft_correct": (np.argmax(soft_probs, axis=1) == np.asarray([CLASSES.index(r["y"]) for r in test])).astype(int),
            "errors": {
                e: (np.argmax(panel[e], axis=1) != np.asarray([CLASSES.index(r["y"]) for r in test])).astype(int).tolist()
                for e in EXPERTS
            },
            "panel": panel,
            "y": [r["y"] for r in test],
            "X": np.asarray([r["x"] for r in test], dtype=float),
            "models": models,
            "x": [r["x"] for r in test],
            "variants": variants,
            "calibrated": calibrated,
            "calibration_temperature": t,
            "state": state,
            "failure_state": failure_state,
            "meta_counts": meta_stats,
            "invariant": invariant,
            "counterfactual": causal_cf,
            "source_outcomes": {},
            "causal_state_scope": "first_prediction_row_of_oos_block",
            "selective_score_rows": (
                0.45 * calibrated.max(axis=1)
                + 0.25 * state["predictability"]["global"]
                + 0.15 * state["meta_label"]["reliability"]
                + 0.15 * state["retrieval"]["success_similarity"]
                - 0.25 * state["uncertainty"]["total"]
            ).clip(0.0, 1.0).tolist(),
        }
        for r in test:
            src = str(r.get("data_source", "unknown"))
            b["source_outcomes"].setdefault(src, []).append(
                int(b["soft"]["accuracy"] >= 0.45)
            )
        for name, item in variants.items():
            b[name] = metrics(b["y"], item["probs"])
        b["full"] = b["full_architecture"]
        b["full_architecture_calibrated"] = metrics(b["y"], calibrated)

        # v13 policy selection is evaluated on the same causal OOS block using
        # only the block state already available before its outcome is consumed.
        policy_case = evaluate_policy_case(
            b,
            previous_strategy=previous_policy_strategy,
            previous_predictability=previous_policy_predictability,
        )
        b["prediction_policy"] = policy_case
        blocks.append(b)
        previous_policy_strategy = policy_case["strategy_selection"]["strategy"]
        previous_policy_predictability = float(policy_case["state"]["predictability"])

        previous_weights = variants["full_architecture"]["weights"]

    return blocks, dev_rows, rows[holdout_row:]


def _aggregate(blocks, key):
    n = sum(b["n"] for b in blocks)
    if not n:
        return {}
    fields = ("accuracy", "logloss", "brier", "calibration_error")
    return {
        f: float(sum(b["n"] * b[key][f] for b in blocks) / n)
        for f in fields
    } | {"n": n, "blocks": len(blocks)}


def _worst_regime(blocks, key):
    out = {}
    for regime in ("TREND", "RANGE"):
        part = [b for b in blocks if b["regime"] == regime]
        if part:
            out[regime] = _aggregate(part, key)
    return out


def _holdout_frozen(dev_blocks, dev_rows, holdout):
    if len(dev_blocks) < 12 or len(holdout) < 100:
        return {"status": "DEFERRED", "reason": "insufficient_frozen_holdout_evidence"}
    # Freeze meta/calibration state at the end of development.
    train = _purged_train(dev_rows, holdout[0]["created"])
    panel, models = _fit_and_predict(train, holdout)
    current_rows, current_panel = _causal_current_snapshot(holdout, panel)
    d = _panel_stats(current_panel, dev_blocks[-1]["panel"])
    prior_rows = dev_rows[-TEST_BLOCK:]
    f_rel = _feature_reliability(current_rows, prior_rows)
    s_rel = _source_reliability(current_rows, dev_blocks)
    i_shock = _information_shock(current_rows, prior_rows)
    p_momentum = _prediction_momentum(
        current_panel, dev_blocks[-1]["panel"], dev_blocks[-2]["panel"]
    )
    regime = _block_regime(current_rows)
    state = {
        "disagreement": d,
        "error_correlation": _error_corr(dev_blocks),
        "feature_reliability": f_rel,
        "source_reliability": s_rel["reliability"],
        "source_reliability_detail": s_rel,
        "information_shock": i_shock,
        "prediction_momentum": p_momentum,
        "regime_transition": _regime_transition(regime, dev_blocks),
        "hard_negative_density": _hard_negative_density(dev_blocks),
        "predictability": dev_blocks[-1]["state"]["predictability"],
        "meta_label": dev_blocks[-1]["state"]["meta_label"],
        "retrieval": _retrieval({
            "disagreement": d, "error_correlation": _error_corr(dev_blocks),
            "feature_reliability": f_rel, "source_reliability": s_rel["reliability"],
            "information_shock": i_shock, "prediction_momentum": p_momentum,
            "regime_transition": _regime_transition(regime, dev_blocks),
            "hard_negative_density": _hard_negative_density(dev_blocks),
            "predictability": dev_blocks[-1]["state"]["predictability"],
            "meta_label": dev_blocks[-1]["state"]["meta_label"],
            "uncertainty": {"total": 0.5},
            "drift": {"feature_drift": 1.0 - f_rel["global"], "prediction_drift": i_shock["update_rate"], "drift_score": 0.5},
        }, dev_blocks),
        "drift": {
            "feature_drift": float(1.0 - f_rel["global"]),
            "prediction_drift": float(i_shock["update_rate"]),
            "drift_score": float(np.clip(0.55 * (1.0 - f_rel["global"]) + 0.45 * i_shock["update_rate"], 0.0, 1.0)),
        },
        "uncertainty": {"total": 0.5},
    }
    holdout_cf = _counterfactual_stability(
        models, np.asarray([current_rows[0]["x"]], dtype=float)
    )
    state["counterfactual_stability"] = {
        **holdout_cf,
        "instability": (
            float(np.clip(1.0 - float(holdout_cf.get("stability", 1.0)), 0.0, 1.0))
            if holdout_cf.get("stability") is not None else None
        ),
    }
    state["uncertainty"] = _uncertainty(state)
    quality = {
        e: float(np.mean([b["metrics"][e]["logloss"] for b in dev_blocks[-PAST_WINDOW:]]))
        for e in EXPERTS
    }
    # Reconstruct frozen meta models using development blocks only.
    predict_model, meta_model, failure_models, hazard_models, meta_stats = _meta_training(dev_blocks, len(dev_blocks))
    state["predictability"] = _predictability(state, predict_model, dev_blocks)
    state["meta_label"] = _meta_label(state, meta_model)
    state["retrieval"] = _retrieval(state, dev_blocks)
    state["uncertainty"] = _uncertainty(state)
    failure_state = _failure_state(state, quality, failure_models, hazard_models, meta_stats)
    policy_variants = {}
    for name in (
        "soft_ensemble", "adaptive_ensemble", "three_layers_regime",
        "three_layers_retrieval", "full_architecture",
    ):
        mix = 0.10 if name == "three_layers_retrieval" else 0.0
        p, w = _evaluate_variant(
            state,
            panel,
            quality,
            failure_state,
            dev_blocks[-1]["variants"]["full_architecture"]["weights"],
            name,
            retrieval_mix=mix,
        )
        policy_variants[name] = {"probs": p, "weights": w}

    full_raw = policy_variants["full_architecture"]["probs"]
    weights = policy_variants["full_architecture"]["weights"]
    baseline = _norm(np.mean(np.stack(list(panel.values()), axis=0), axis=0))
    prior_cal_p = [np.asarray(b["calibrated"], dtype=float) for b in dev_blocks[-8:]]
    prior_cal_y = [y for b in dev_blocks[-8:] for y in b["y"]]
    t = _temperature(np.vstack(prior_cal_p), prior_cal_y) if len(prior_cal_y) >= 50 else 1.0
    full_cal = _apply_temperature(full_raw, t)
    y = [r["y"] for r in holdout]
    holdout_policy_block = {
        "index": -1,
        "y": y,
        "state": state,
        "failure_state": failure_state,
        "variants": policy_variants,
    }
    previous_policy = dev_blocks[-1].get("prediction_policy", {})
    previous_strategy = (
        previous_policy.get("strategy_selection", {}).get("strategy")
        if isinstance(previous_policy, dict)
        else None
    )
    previous_predictability = float(dev_blocks[-1]["state"]["predictability"]["global"])
    policy_holdout = evaluate_policy_holdout_case(
        holdout_policy_block,
        previous_strategy=previous_strategy,
        previous_predictability=previous_predictability,
    )
    policy_holdout = {
        "status": "FROZEN_HOLDOUT_EVALUATED",
        **policy_holdout,
        "selection_frozen_before_holdout": True,
    }
    return {
        "status": "OK",
        "n": len(holdout),
        "baseline": metrics(y, baseline),
        "full": metrics(y, full_cal),
        "delta": {
            "accuracy": metrics(y, full_cal)["accuracy"] - metrics(y, baseline)["accuracy"],
            "logloss": metrics(y, full_cal)["logloss"] - metrics(y, baseline)["logloss"],
            "brier": metrics(y, full_cal)["brier"] - metrics(y, baseline)["brier"],
            "ece": metrics(y, full_cal)["calibration_error"] - metrics(y, baseline)["calibration_error"],
        },
        "coverage": _conformal_eval(full_cal, y, _conformal_from_blocks(dev_blocks)),
        "weights": weights.tolist(),
        "temperature": t,
        "counterfactual": holdout_cf,
        "causal_state_scope": "first_prediction_row_of_frozen_holdout",
        "stress": None,
        "prediction_policy": policy_holdout,
    }


def evaluate(horizon, max_rows=9000):
    rows = load_archive_research_rows(horizon, max_rows)
    if len(rows) < MIN_TRAIN + TEST_BLOCK * 13:
        return {
            "status": "DEFERRED",
            "research_mode": "REDUCED",
            "reason": "insufficient_archive_rows",
            "n_rows": len(rows),
            "required": MIN_TRAIN + TEST_BLOCK * 13,
            "production_changed": False,
            "promotion": {
                "eligible": False,
                "promotion_allowed": False,
                "decision": "HOLD",
                "reason": "research evidence deferred: insufficient archive rows",
            },
        }

    blocks, dev_rows, holdout = _development(rows)
    if len(blocks) < 12:
        return {
            "status": "DEFERRED",
            "research_mode": "REDUCED",
            "reason": "insufficient_oos_blocks",
            "n_rows": len(rows),
            "blocks": len(blocks),
            "production_changed": False,
            "promotion": {
                "eligible": False,
                "promotion_allowed": False,
                "decision": "HOLD",
                "reason": "research evidence deferred: insufficient chronological OOS blocks",
            },
        }

    names = [
        "soft_ensemble", "adaptive_ensemble", "disagreement_model",
        "predictability_model", "future_failure_predictor", "drift_aware_router",
        "three_layers", "three_layers_error_corr", "three_layers_regime",
        "three_layers_retrieval", "full_architecture",
    ]
    dev_cut = max(2, int(len(blocks) * (1.0 - FINAL_HOLDOUT_FRAC)))
    dev_blocks = blocks[:dev_cut]
    descriptive_hold_blocks = blocks[dev_cut:]
    summary = {n: _aggregate(dev_blocks, n) for n in names}
    baseline = summary["soft_ensemble"]
    full = summary["full_architecture"]
    deltas = {
        "accuracy": full["accuracy"] - baseline["accuracy"],
        "logloss": full["logloss"] - baseline["logloss"],
        "brier": full["brier"] - baseline["brier"],
        "ece": full["calibration_error"] - baseline["calibration_error"],
    }

    ci_ll = _bootstrap_ci(
        [b["full_architecture"]["logloss"] - b["soft_ensemble"]["logloss"] for b in dev_blocks],
        reps=BOOTSTRAP_REPS,
    )
    worst = _worst_regime(dev_blocks, "full_architecture")
    selective = _selective_metrics(dev_blocks)
    conf = _conformal_from_blocks(dev_blocks)
    last_stress = _stress(dev_blocks[-1], dev_blocks[-1]["models"])
    invariant = dev_blocks[-1]["invariant"]
    holdout = _holdout_frozen(dev_blocks, dev_rows, holdout)
    policy_oos = summarize_policy_blocks(dev_blocks)

    # Detection lead time on development history: compare risk at T to failure
    # actually realized after T. Both are measured strictly out-of-sample.
    leads = []
    false_alarms = 0
    for i, b in enumerate(dev_blocks[:-FUTURE_WINDOW]):
        r = float(b["failure_state"]["mean"])
        future = dev_blocks[i + 1:i + 1 + FUTURE_WINDOW]
        failed = any(
            (np.mean([x["metrics"][e]["logloss"] for x in future]) -
             np.mean([x["metrics"][e]["logloss"] for x in dev_blocks[max(0, i-PAST_WINDOW+1):i+1]])) >= 0.05
            for e in EXPERTS
        )
        if r >= 0.5 and failed:
            leads.append(1)
        elif r >= 0.5 and not failed:
            false_alarms += 1

    period = {}
    n = len(dev_blocks)
    for name, lo, hi in (("early", 0, max(1, n//3)), ("middle", max(1,n//3), max(2,2*n//3)), ("recent", max(2,2*n//3), n)):
        part = dev_blocks[lo:hi]
        if part:
            period[name] = {
                "accuracy_delta": float(np.mean([b["full_architecture"]["accuracy"] - b["soft_ensemble"]["accuracy"] for b in part])),
                "logloss_delta": float(np.mean([b["full_architecture"]["logloss"] - b["soft_ensemble"]["logloss"] for b in part])),
                "brier_delta": float(np.mean([b["full_architecture"]["brier"] - b["soft_ensemble"]["brier"] for b in part])),
            }

    mc = _champion_benchmark(horizon, rows)
    promotion = {
        "eligible": False,
        "decision": "HOLD",
        "reason": "v6_research_only_requires_live_primary_shadow_and_longer_independent_evidence",
        "development_relative_logloss_gain": float(
            (baseline["logloss"] - full["logloss"]) / max(abs(baseline["logloss"]), EPS)
        ),
        "development_relative_brier_gain": float(
            (baseline["brier"] - full["brier"]) / max(abs(baseline["brier"]), EPS)
        ),
    }

    return {
        "status": "OK",
        "research_mode": "REDUCED",
        "research_only": True,
        "production_changed": False,
        "horizon": horizon,
        "n_rows": len(rows),
        "development_rows": len(dev_rows),
        "development_blocks": len(dev_blocks),
        "internal_holdout_blocks": len(descriptive_hold_blocks),
        "oos_summary": summary,
        "baseline": baseline,
        "full_architecture": full,
        "deltas_vs_soft": deltas,
        "worst_regime": worst,
        "period_breakdown": period,
        "statistical_validation": {
            "moving_block_bootstrap_logloss_delta": ci_ll,
            "block_count": len(dev_blocks),
        },
        "selective_prediction": selective,
        "conformal": conf,
        "conformal_holdout": holdout.get("coverage") if isinstance(holdout, dict) else None,
        "failure_monitoring": {
            "detection_lead_blocks": float(np.mean(leads)) if leads else None,
            "detected_failure_cases": len(leads),
            "false_alarm_count": false_alarms,
            "latest_failure_risk": dev_blocks[-1]["failure_state"],
            "mean_risk_calibration": _binary_calibration(
                [b["failure_state"]["mean"] for b in dev_blocks[:-FUTURE_WINDOW]],
                [
                    int(any(
                        (
                            np.mean([x["metrics"][e]["logloss"] for x in dev_blocks[i + 1:i + 1 + FUTURE_WINDOW]])
                            - np.mean([x["metrics"][e]["logloss"] for x in dev_blocks[max(0, i - PAST_WINDOW + 1):i + 1]])
                        ) >= 0.05
                        or (
                            np.mean([x["metrics"][e]["accuracy"] for x in dev_blocks[i + 1:i + 1 + FUTURE_WINDOW]])
                            - np.mean([x["metrics"][e]["accuracy"] for x in dev_blocks[max(0, i - PAST_WINDOW + 1):i + 1]])
                        ) <= -0.05
                        for e in EXPERTS
                    )) for i in range(len(dev_blocks) - FUTURE_WINDOW)
                ],
            ),
        },
        "predictability": dev_blocks[-1]["state"]["predictability"],
        "predictability_calibration": _binary_calibration(
            [b["state"]["predictability"]["global"] for b in dev_blocks[:-1]],
            [int(dev_blocks[i + 1]["soft"]["accuracy"] >= 0.45) for i in range(len(dev_blocks) - 1)],
        ),
        "error_correlation": dev_blocks[-1]["state"]["error_correlation"],
        "feature_reliability": dev_blocks[-1]["state"]["feature_reliability"],
        "source_reliability": dev_blocks[-1]["state"]["source_reliability_detail"],
        "information_shock": dev_blocks[-1]["state"]["information_shock"],
        "prediction_momentum": dev_blocks[-1]["state"]["prediction_momentum"],
        "regime_transition": dev_blocks[-1]["state"]["regime_transition"],
        "retrieval": dev_blocks[-1]["state"]["retrieval"],
        "uncertainty": dev_blocks[-1]["state"]["uncertainty"],
        "counterfactual_stability": dev_blocks[-1]["counterfactual"],
        "invariant_features": invariant,
        "hard_negative_density": dev_blocks[-1]["state"]["hard_negative_density"],
        "adaptive_compute": {
            "latest_tier": _compute_tier(
                dev_blocks[-1]["state"]["predictability"]["global"],
                dev_blocks[-1]["state"]["drift"]["drift_score"],
                dev_blocks[-1]["failure_state"]["mean"],
            ),
            "policy": "easy=>standard; medium=>adaptive_ensemble; hard=>full_stress",
        },
        "model_aging": {
            "research_block_retrain_age": 0,
            "note": "research experts are refit causally per OOS block; production model age tracked separately",
        },
        "uncertainty_decomposition": dev_blocks[-1]["state"]["uncertainty"],
        "hidden_state": {
            "momentum": float(np.clip(
                0.5 + dev_blocks[-1]["state"]["prediction_momentum"]["velocity"], 0.0, 1.0
            )),
            "stress": float(np.clip(
                0.5 * dev_blocks[-1]["state"]["information_shock"]["shock_score"]
                + 0.5 * dev_blocks[-1]["state"]["drift"]["drift_score"], 0.0, 1.0
            )),
            "uncertainty": float(dev_blocks[-1]["state"]["uncertainty"]["total"]),
            "method": "deterministic_observation_state_proxy; not_causal_latent_state_claim",
        },
        "multi_horizon_consistency_hook": {
            "status": "PENDING_CROSS_HORIZON_ALIGNMENT",
            "horizon": horizon,
            "note": "cross-horizon alignment requires matched 5m/10m OOS timestamps and is not used for routing in this run",
        },
        "robustness": {
            "status": "PARTIAL",
            "development_stress": last_stress,
            "latest_counterfactual": dev_blocks[-1]["counterfactual"],
        },
        "meta_leakage_policy": {
            "future_labels_not_used_for_current_state": True,
            "failure_labels_require_future_window_completion": True,
            "meta_models_fit_only_on_prior_blocks": True,
            "quality_prior_excludes_current_block": True,
            "calibration_uses_prior_oos_only": True,
            "frozen_holdout_used_for_selection": False,
        },
        "pit_policy": {
            "archive_rows_closed": True,
            "created_before_target": True,
            "training_target_before_embargo": True,
            "random_split": False,
        },
        "champion_benchmark": mc,
        "offline_holdout": holdout,
        "prediction_policy_oos": {
            "status": policy_oos.get("status", "DEFERRED"),
            "development": policy_oos,
            "frozen_holdout": holdout.get("prediction_policy", {}) if isinstance(holdout, dict) else {},
            "selection_scope": "development_OOS_only",
            "frozen_holdout_used_for_selection": False,
        },
        "shadow": {
            "status": "OFFLINE_REPLAY_ONLY",
            "production_live_shadow": False,
            "production_changed": False,
        },
        "promotion": promotion,
        "artifacts_expected": True,
    }


def write_artifacts(result):
    h = result["horizon"]
    root = OUT_DIR
    prefix = root / f"maximum_future_generalization_{h}"
    payloads = {
        "_disagreement_features.json": {
            "horizon": h, "disagreement": result.get("error_correlation", {}), "latest": result.get("predictability", {})
        },
        "_predictability_model.json": result.get("predictability", {}),
        "_future_failure_model.json": result.get("failure_monitoring", {}),
        "_drift_detector.json": {
            "information_shock": result.get("information_shock", {}),
            "regime_transition": result.get("regime_transition", {}),
        },
        "_dynamic_router.json": {
            "summary": result.get("oos_summary", {}),
            "deltas_vs_soft": result.get("deltas_vs_soft", {}),
            "promotion": result.get("promotion", {}),
        },
        "_calibration_artifact.json": {
            "method": "chronological_temperature_from_prior_blocks",
            "selective": result.get("selective_prediction", {}),
            "conformal": result.get("conformal", {}),
        },
        "_statistical_validation.json": result.get("statistical_validation", {}),
        "_robustness.json": result.get("robustness", {}),
        "_selective_policy.json": result.get("selective_prediction", {}),
        "_shadow_results.json": result.get("shadow", {}),
        "_challenger_results.json": result.get("offline_holdout", {}),
        "_promotion_gate.json": result.get("promotion", {}),
        "_prediction_policy.json": result.get("prediction_policy_oos", {}),
        "_fallback_config.json": {
            "research_only": True,
            "policy": "full=>reduced=>soft_equal=>verified_baseline",
            "unsafe_states": [
                "PIT_FAIL", "LEAKAGE_SUSPICION", "ARTIFACT_CORRUPTION",
                "FEATURE_SCHEMA_MISMATCH", "EXTREME_PROBABILITY",
                "SEVERE_DRIFT", "META_MODEL_UNAVAILABLE"
            ],
        },
    }
    for suffix, payload in payloads.items():
        (prefix.with_name(prefix.name + suffix)).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for h in HORIZONS:
        try:
            result = evaluate(h)
        except Exception as exc:
            result = {
                "status": "FAILED",
                "research_only": True,
                "production_changed": False,
                "horizon": h,
                "reason": f"{type(exc).__name__}:{exc}",
            }
        outputs[h] = result
        if result.get("status") == "OK":
            write_artifacts(result)
    aggregate = {
        "schema_version": 1,
        "experiment": "maximum_future_generalization_predictive_control_v6",
        "generated_at_utc": utc_now(),
        "research_only": True,
        "production_changed": False,
        "research_mode": "REDUCED",
        "horizons": outputs,
    }
    (OUT_DIR / "maximum_future_generalization_v6_registry.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
