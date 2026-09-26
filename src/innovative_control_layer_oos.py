"""Research-only v2 innovative BTC prediction control layer.

Implements a prequential meta-control experiment around four causal base
experts. Every meta-model sample uses state observed at time T and a label
derived only from OOS blocks strictly after T and completed before the current
decision block. No production artifact is mutated.

Outputs:
- disagreement feature diagnostics
- predictability score
- future model failure risks
- drift/regime signals
- dynamic soft routing
- chronological calibration
- selective coverage
- adversarial/stress checks
- ablation/statistical validation
- promotion/shadow/challenger status
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

from label_policy import CLASSES
from model_compare import load_archive_research_rows, load_primary_production_strict_rows, metrics
from runtime_production_model import resolve_production_model

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "historical_research"
MODEL_DIR = ROOT / "models"

HORIZONS = ("5m", "10m")
EXPERTS = ("logreg", "extra_trees", "hgb", "lightgbm")
MIN_TRAIN = 2000
TEST_BLOCK = 100
MAX_BLOCKS = 36
FUTURE_WINDOW = 3
PAST_WINDOW = 3
MIN_META_SAMPLES = 8
FINAL_HOLDOUT_FRAC = 0.20
BOOTSTRAP_REPS = 1000
BLOCK_BOOTSTRAP_LEN = 3
SEED = 42

EPS = 1e-8


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _expert_factories():
    out = {
        "logreg": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.1, max_iter=3000)),
        ]),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=260,
            max_depth=8,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=220,
            max_leaf_nodes=15,
            learning_rate=0.04,
            l2_regularization=1.0,
            random_state=SEED,
        ),
    }
    if LGBMClassifier is not None:
        out["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=220,
            num_leaves=15,
            learning_rate=0.03,
            min_child_samples=30,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
    else:
        out["lightgbm"] = out["hgb"]
    return out


def _norm(p):
    arr = np.asarray(p, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 3 or not np.isfinite(arr).all():
        raise ValueError("invalid probability matrix")
    arr = np.clip(arr, EPS, 1.0)
    return arr / arr.sum(axis=1, keepdims=True)


def _aligned(model, X):
    raw = np.asarray(model.predict_proba(X), dtype=float)
    out = np.full((len(X), 3), EPS, dtype=float)
    for j, c in enumerate(getattr(model, "classes_", [])):
        c = str(c)
        if c in CLASSES:
            out[:, CLASSES.index(c)] = raw[:, j]
    return _norm(out)


def _entropy(p):
    q = np.clip(np.asarray(p, dtype=float), EPS, 1.0)
    return float(-np.sum(q * np.log(q)) / math.log(3.0))


def _rank_disagreement(panel):
    ranks = np.argsort(-panel, axis=2)
    if len(panel) == 0:
        return 0.0
    pairwise = []
    for i in range(len(EXPERTS)):
        for j in range(i + 1, len(EXPERTS)):
            pairwise.append(float(np.mean(ranks[i] != ranks[j])))
    return float(np.mean(pairwise)) if pairwise else 0.0


def disagreement_features(panel_probs: dict[str, np.ndarray]) -> dict[str, float]:
    matrix = np.stack([panel_probs[e] for e in EXPERTS], axis=0)
    mean_p = matrix.mean(axis=0)
    std_p = matrix.std(axis=0)
    top = np.argmax(matrix, axis=2)
    agreement = float(np.mean(top == np.argmax(mean_p, axis=1)))
    margin = np.sort(mean_p, axis=1)[:, -1] - np.sort(mean_p, axis=1)[:, -2]
    pairwise = []
    for i in range(len(EXPERTS)):
        for j in range(i + 1, len(EXPERTS)):
            pairwise.append(float(np.mean(top[i] != top[j])))
    return {
        "mean_probability_down": float(mean_p[:, 0].mean()),
        "mean_probability_flat": float(mean_p[:, 1].mean()),
        "mean_probability_up": float(mean_p[:, 2].mean()),
        "std_probability": float(std_p.mean()),
        "min_probability": float(matrix.min(axis=(0, 2)).mean()),
        "max_probability": float(matrix.max(axis=(0, 2)).mean()),
        "probability_range": float((matrix.max(axis=0) - matrix.min(axis=0)).mean()),
        "prediction_entropy": float(np.mean([_entropy(x) for x in mean_p])),
        "top_class_agreement_rate": agreement,
        "majority_margin": float(margin.mean()),
        "rank_disagreement": _rank_disagreement(matrix),
        "pairwise_disagreement": float(np.mean(pairwise)) if pairwise else 0.0,
    }


def _aggregate_panel(panel_probs):
    return _norm(np.mean(np.stack(list(panel_probs.values()), axis=0), axis=0))


def _block_regime(rows):
    x = np.asarray([r["x"] for r in rows], dtype=float)
    # Production feature order: ret_5m index 2, ret_15m/30m are absent, so
    # use a conservative short-horizon trend proxy and volatility.
    trend = float(np.mean(x[:, 2])) if len(x) else 0.0
    vol = float(np.mean(x[:, 6])) if len(x) else 0.0
    threshold = max(0.0007, 1.5 * max(vol, 1e-8))
    return "TREND" if abs(trend) > threshold else "RANGE"


def _feature_shift(current_rows, prior_rows):
    if not current_rows or not prior_rows:
        return 0.0
    a = np.asarray([r["x"] for r in current_rows], dtype=float)
    b = np.asarray([r["x"] for r in prior_rows], dtype=float)
    means_a, means_b = a.mean(axis=0), b.mean(axis=0)
    scale = np.std(b, axis=0) + 1e-8
    z = np.abs(means_a - means_b) / scale
    return float(np.mean(np.clip(z, 0.0, 5.0)) / 5.0)


def _prediction_shift(current_panel, prior_panel):
    if not current_panel or not prior_panel:
        return 0.0
    cur = _aggregate_panel(current_panel).mean(axis=0)
    prev = _aggregate_panel(prior_panel).mean(axis=0)
    return float(np.mean(np.abs(cur - prev)) / 2.0)


def drift_features(current_rows, prior_rows, current_panel, prior_panel, *,
                   current_disagreement=0.0, prior_disagreement=0.0,
                   recent_calibration=0.0, older_calibration=0.0) -> dict[str, float]:
    feature_drift = _feature_shift(current_rows, prior_rows)
    prediction_drift = _prediction_shift(current_panel, prior_panel)
    completeness = float(np.mean([
        float(np.isfinite(r["x"]).all()) for r in current_rows
    ])) if current_rows else 0.0
    disagreement_drift = float(abs(current_disagreement - prior_disagreement))
    calibration_drift = float(abs(recent_calibration - older_calibration))
    return {
        "feature_drift": feature_drift,
        "prediction_drift": prediction_drift,
        "label_rate_proxy": 0.0,
        "model_disagreement_drift": disagreement_drift,
        "calibration_drift": calibration_drift,
        "data_quality_drift": float(1.0 - completeness),
        "drift_score": float(np.clip(
            0.45 * feature_drift
            + 0.25 * prediction_drift
            + 0.15 * disagreement_drift
            + 0.15 * calibration_drift
            + 0.10 * (1.0 - completeness),
            0.0, 1.0
        )),
    }


def _state_vector(disagreement, drift, predictability_hint=1.0, failure_hint=0.5):
    return np.array([
        disagreement["std_probability"],
        disagreement["probability_range"],
        disagreement["prediction_entropy"],
        disagreement["top_class_agreement_rate"],
        disagreement["majority_margin"],
        disagreement["rank_disagreement"],
        disagreement["pairwise_disagreement"],
        disagreement["recent_disagreement"],
        disagreement["disagreement_change_rate"],
        disagreement["rolling_disagreement"],
        disagreement["regime_conditioned_disagreement"],
        drift["feature_drift"],
        drift["prediction_drift"],
        drift["model_disagreement_drift"],
        drift["calibration_drift"],
        drift["drift_score"],
        predictability_hint,
        failure_hint,
    ], dtype=float)


def _state_vector_expert(state, quality_ll):
    return np.concatenate([state, [quality_ll]], dtype=float)


def _binary_model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.25, max_iter=1800, class_weight="balanced")),
    ])


def _safe_fit_binary(X, y):
    if len(y) < MIN_META_SAMPLES or len(set(map(int, y))) < 2:
        return None
    m = _binary_model()
    m.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
    return m


def _temperature(probs, ys):
    if len(ys) < 50 or len(set(ys)) < 3:
        return 1.0
    p = _norm(probs)
    y = np.asarray([CLASSES.index(v) for v in ys], dtype=int)
    best_t, best_ll = 1.0, float("inf")
    for t in np.linspace(0.7, 2.5, 37):
        z = np.log(np.clip(p, EPS, 1.0)) / float(t)
        z -= z.max(axis=1, keepdims=True)
        q = np.exp(z)
        q /= q.sum(axis=1, keepdims=True)
        score = metrics(ys, q)["logloss"]
        if score < best_ll:
            best_ll, best_t = score, float(t)
    return best_t


def _apply_temperature(p, t):
    p = _norm(p)
    if abs(float(t) - 1.0) < 1e-12:
        return p
    z = np.log(p) / float(t)
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z)
    q /= q.sum(axis=1, keepdims=True)
    return q


def _fit_panel(train_rows, test_rows):
    factories = _expert_factories()
    X = np.asarray([r["x"] for r in train_rows], dtype=float)
    y = np.asarray([r["y"] for r in train_rows])
    Xt = np.asarray([r["x"] for r in test_rows], dtype=float)
    panel = {}
    models = {}
    for expert in EXPERTS:
        model = factories[expert]()
        model.fit(X, y)
        panel[expert] = _aligned(model, Xt)
        models[expert] = model
    return panel, models


def _window_points(n):
    starts = list(range(MIN_TRAIN, max(MIN_TRAIN + 1, n - TEST_BLOCK + 1), TEST_BLOCK))
    if len(starts) <= MAX_BLOCKS:
        return starts
    idx = np.linspace(0, len(starts) - 1, num=MAX_BLOCKS).astype(int)
    return [starts[i] for i in sorted(set(idx.tolist()))]


def _block_metrics(panel, rows):
    y = [r["y"] for r in rows]
    out = {}
    for expert in EXPERTS:
        out[expert] = metrics(y, panel[expert])
    soft = _aggregate_panel(panel)
    out["soft_ensemble"] = metrics(y, soft)
    return out


def _build_failure_labels(blocks, current_index):
    labels = {e: None for e in EXPERTS}
    if current_index < PAST_WINDOW or current_index + FUTURE_WINDOW >= len(blocks):
        return labels
    for expert in EXPERTS:
        past = [blocks[k]["metrics"][expert] for k in range(current_index - PAST_WINDOW, current_index)]
        future = [blocks[k]["metrics"][expert] for k in range(current_index + 1, current_index + 1 + FUTURE_WINDOW)]
        past_ll = float(np.mean([m["logloss"] for m in past]))
        future_ll = float(np.mean([m["logloss"] for m in future]))
        past_acc = float(np.mean([m["accuracy"] for m in past]))
        future_acc = float(np.mean([m["accuracy"] for m in future]))
        ll_delta = future_ll - past_ll
        acc_delta = future_acc - past_acc
        labels[expert] = {
            "future_logloss_delta": ll_delta,
            "future_accuracy_delta": acc_delta,
            "future_failure": int(ll_delta >= 0.05 or acc_delta <= -0.05),
        }
    return labels


def _build_predictability_label(blocks, current_index):
    if current_index < PAST_WINDOW or current_index + FUTURE_WINDOW >= len(blocks):
        return None
    past = [blocks[k]["metrics"]["soft_ensemble"] for k in range(current_index - PAST_WINDOW, current_index)]
    future = [blocks[k]["metrics"]["soft_ensemble"] for k in range(current_index + 1, current_index + 1 + FUTURE_WINDOW)]
    return {
        "future_accuracy": float(np.mean([m["accuracy"] for m in future])),
        "future_logloss": float(np.mean([m["logloss"] for m in future])),
        "easy": int(np.mean([m["accuracy"] for m in future]) >= 0.45),
    }


def _past_completed_meta(blocks, before_index):
    failure_rows = {e: [] for e in EXPERTS}
    predict_rows = []
    for i in range(before_index):
        # A meta label is usable only after its entire future evaluation window
        # has completed. This is the core protection against meta-leakage.
        if i + FUTURE_WINDOW >= before_index:
            continue
        fl = _build_failure_labels(blocks, i)
        if all(fl[e] is not None for e in EXPERTS):
            for e in EXPERTS:
                d = blocks[i]["state"]
                risk_hint = float(fl[e]["future_failure"])
                failure_rows[e].append((
                    _state_vector_expert(
                        _state_vector(d["disagreement"], d["drift"]),
                        d["quality_logloss"][e],
                    ),
                    risk_hint,
                ))
        pl = _build_predictability_label(blocks, i)
        if pl is not None:
            d = blocks[i]["state"]
            predict_rows.append((
                _state_vector(d["disagreement"], d["drift"]),
                pl["easy"],
            ))
    return failure_rows, predict_rows


def _fit_meta_models(blocks, before_index):
    failure_rows, predict_rows = _past_completed_meta(blocks, before_index)
    failure_models = {}
    for expert in EXPERTS:
        pairs = failure_rows[expert]
        if pairs:
            failure_models[expert] = _safe_fit_binary(
                [x for x, _ in pairs],
                [y for _, y in pairs],
            )
    predictability_model = _safe_fit_binary(
        [x for x, _ in predict_rows],
        [y for _, y in predict_rows],
    ) if predict_rows else None
    return failure_models, predictability_model, {
        "failure_samples": {e: len(failure_rows[e]) for e in EXPERTS},
        "predictability_samples": len(predict_rows),
    }


def _quality_weights(quality_logloss):
    vals = np.asarray([-quality_logloss[e] for e in EXPERTS], dtype=float)
    vals -= vals.max()
    w = np.exp(vals)
    return w / w.sum()


def _failure_risks(state, quality_logloss, models):
    out = {}
    for expert in EXPERTS:
        m = models.get(expert)
        if m is None:
            out[expert] = 0.5
            continue
        x = _state_vector_expert(
            _state_vector(state["disagreement"], state["drift"]),
            quality_logloss[expert],
        ).reshape(1, -1)
        out[expert] = float(m.predict_proba(x)[0, 1])
    return out


def _predictability(state, model):
    if model is None:
        return 0.5
    x = _state_vector(state["disagreement"], state["drift"]).reshape(1, -1)
    return float(model.predict_proba(x)[0, 1])


def _route_weights(
    quality_logloss,
    disagreement,
    drift,
    predictability,
    failure_risks,
    *,
    use_disagreement=True,
    use_predictability=True,
    use_failure=True,
    use_drift=True,
    previous=None,
):
    q = _quality_weights(quality_logloss)
    equal = np.full(len(EXPERTS), 1.0 / len(EXPERTS))
    if use_failure:
        q = q * np.asarray([math.exp(-1.75 * failure_risks[e]) for e in EXPERTS])
        q /= max(float(q.sum()), EPS)
    flatten = 1.0
    if use_predictability:
        flatten *= float(np.clip(predictability, 0.0, 1.0))
    if use_disagreement:
        flatten *= float(np.clip(1.0 - disagreement["pairwise_disagreement"], 0.0, 1.0))
    if use_drift:
        flatten *= float(np.clip(1.0 - drift["drift_score"], 0.0, 1.0))
    flatten = 0.20 + 0.80 * flatten
    w = flatten * q + (1.0 - flatten) * equal
    if previous is not None:
        w = 0.75 * previous + 0.25 * w
    w = np.clip(w, 1e-4, 1.0)
    return w / w.sum()


def _route_probs(panel, weights):
    matrix = np.stack([panel[e] for e in EXPERTS], axis=0)
    return _norm(np.tensordot(weights, matrix, axes=(0, 0)))


def _selective_score(probs, predictability, failure_risks, drift):
    conf = np.max(probs, axis=1)
    risk = float(np.mean(list(failure_risks.values()))) if failure_risks else 0.5
    scalar = 0.55 * float(predictability) + 0.45 * float(conf.mean()) - 0.20 * float(drift["drift_score"]) - 0.25 * risk
    return float(np.clip(scalar, 0.0, 1.0))


def _coverage_metrics(probs, ys, threshold):
    conf = np.max(probs, axis=1)
    keep = conf >= threshold
    coverage = float(keep.mean()) if len(keep) else 0.0
    if not keep.any():
        return {"coverage": 0.0, "accuracy": None, "logloss": None, "brier": None, "ece": None, "n": 0}
    sub_y = [y for y, k in zip(ys, keep) if k]
    sub_p = probs[keep]
    m = metrics(sub_y, sub_p)
    return {
        "coverage": coverage,
        "accuracy": m["accuracy"],
        "logloss": m["logloss"],
        "brier": m["brier"],
        "ece": m["calibration_error"],
        "n": int(keep.sum()),
    }


def _target_coverage_thresholds(previous_scores):
    if not previous_scores:
        return {c: 0.0 for c in (1.0, 0.95, 0.90, 0.80, 0.70)}
    vals = np.asarray(previous_scores, dtype=float)
    return {
        c: float(np.quantile(vals, max(0.0, 1.0 - c)))
        for c in (1.0, 0.95, 0.90, 0.80, 0.70)
    }


def _bootstrap_ci(values, seed=SEED, block_len=BLOCK_BOOTSTRAP_LEN, reps=BOOTSTRAP_REPS):
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 6:
        return {"low": None, "high": None, "n": n, "method": "insufficient_blocks"}
    rng = np.random.default_rng(seed)
    estimates = []
    block_len = max(1, min(int(block_len), n))
    starts = np.arange(0, n - block_len + 1)
    for _ in range(reps):
        sample = []
        while len(sample) < n:
            s = int(rng.choice(starts))
            sample.extend(x[s:s + block_len].tolist())
        estimates.append(float(np.mean(sample[:n])))
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return {"low": float(lo), "high": float(hi), "n": n, "method": "moving_block_bootstrap", "block_len": block_len, "reps": reps}


def _period_breakdown(block_results):
    if not block_results:
        return {}
    out = {}
    k = len(block_results)
    for name, lo, hi in (
        ("early", 0, max(1, k // 3)),
        ("middle", max(1, k // 3), max(2, 2 * k // 3)),
        ("recent", max(2, 2 * k // 3), k),
    ):
        part = block_results[lo:hi]
        out[name] = {
            "blocks": len(part),
            "full_vs_soft_accuracy_delta": float(np.mean([x["full"]["accuracy"] - x["soft_ensemble"]["accuracy"] for x in part])),
            "full_vs_soft_logloss_delta": float(np.mean([x["full"]["logloss"] - x["soft_ensemble"]["logloss"] for x in part])),
            "full_vs_soft_brier_delta": float(np.mean([x["full"]["brier"] - x["soft_ensemble"]["brier"] for x in part])),
        }
    return out


def _champion_benchmark(horizon, rows):
    try:
        bundle = resolve_production_model(horizon)
        trained = bundle.metadata.get("trained_at_utc")
        if not trained:
            return {"status": "DEFERRED", "reason": "production_training_timestamp_missing"}
        cutoff = datetime.fromisoformat(str(trained).replace("Z", "+00:00"))
        usable = [
            r for r in rows
            if datetime.fromisoformat(r["created"].replace("Z", "+00:00")) > cutoff
        ]
        if len(usable) < 300:
            return {"status": "DEFERRED", "reason": "insufficient_post_training_oos_rows", "n": len(usable)}
        sources = {str(r.get("data_source", "")) for r in usable}
        if sources != {"binance_vision"}:
            return {
                "status": "DEFERRED",
                "reason": "champion_benchmark_requires_binance_vision_same_product",
                "sources": sorted(sources),
                "n": len(usable),
            }
        X = np.asarray([r["x"] for r in usable], dtype=float)
        y = [r["y"] for r in usable]
        p = _aligned(bundle.load(), X)
        return {
            "status": "OK",
            "train_cutoff": trained,
            "n": len(usable),
            "metrics": metrics(y, p),
            "pit_policy": "fixed production artifact evaluated only after trained_at_utc",
        }
    except Exception as exc:
        return {"status": "DEFERRED", "reason": f"champion_error:{type(exc).__name__}:{exc}"}


def _stress_test(reference):
    base = np.asarray(reference["weights"], dtype=float)
    scenarios = {}
    for name, changes in {
        "disagreement_spike": {"disagreement_pairwise": min(1.0, reference["disagreement"] + 0.35)},
        "severe_drift": {"drift": 1.0},
        "low_predictability": {"predictability": 0.10},
        "high_failure_risk": {"failure": 0.90},
    }.items():
        disagreement = reference["disagreement"]
        drift = reference["drift"]
        pred = reference["predictability"]
        risks = reference["failure_risks"].copy()
        if "disagreement_pairwise" in changes:
            disagreement = dict(disagreement)
            disagreement["pairwise_disagreement"] = changes["disagreement_pairwise"]
        if "drift" in changes:
            drift = dict(drift, drift_score=changes["drift"])
        if "predictability" in changes:
            pred = changes["predictability"]
        if "failure" in changes:
            risks = {e: changes["failure"] for e in EXPERTS}
        w = _route_weights(
            reference["quality_logloss"],
            disagreement,
            drift,
            pred,
            risks,
            previous=base,
        )
        scenarios[name] = {
            "weight_l1_shift": float(np.abs(w - base).sum()),
            "weight_max_shift": float(np.max(np.abs(w - base))),
            "weights": w.tolist(),
        }
    return {"status": "OK", "reference": {"weights": base.tolist()}, "scenarios": scenarios}


def evaluate(horizon):
    rows = load_archive_research_rows(horizon, 12000)
    if len(rows) < MIN_TRAIN + TEST_BLOCK * 12:
        return {"status": "DEFERRED", "reason": "insufficient_archive_rows", "n": len(rows)}
    points = _window_points(len(rows))
    blocks = []
    prior_weights = None
    holdout_cut = max(12, int(len(points) * (1.0 - FINAL_HOLDOUT_FRAC)))
    frozen_failure_models = None
    frozen_predict_model = None
    frozen_meta_counts = None
    frozen_temperature = 1.0
    frozen_reference_scores = None

    for bi, test_start in enumerate(points):
        test = rows[test_start:min(test_start + TEST_BLOCK, len(rows))]
        train_end = max(MIN_TRAIN, test_start)
        test_start_dt = datetime.fromisoformat(
            str(test[0]["created"]).replace("Z", "+00:00")
        )
        embargo_cutoff = test_start_dt - timedelta(minutes=60)
        train = [
            r for r in rows[:train_end]
            if (
                datetime.fromisoformat(str(r["created"]).replace("Z", "+00:00")) < embargo_cutoff
                and datetime.fromisoformat(str(r["target"]).replace("Z", "+00:00")) < embargo_cutoff
                and datetime.fromisoformat(str(r["created"]).replace("Z", "+00:00"))
                < datetime.fromisoformat(str(r["target"]).replace("Z", "+00:00"))
            )
        ]
        if len(test) < TEST_BLOCK // 2 or len(train) < MIN_TRAIN:
            continue
        panel, _ = _fit_panel(train, test)
        metrics_now = _block_metrics(panel, test)
        prior_panel = blocks[-1]["panel"] if blocks else {}
        prior_rows = rows[max(0, test_start - TEST_BLOCK):test_start]
        disagreement = disagreement_features(panel)
        current_pairwise = disagreement["pairwise_disagreement"]
        previous_pairs = [
            float(b["state"]["disagreement"]["pairwise_disagreement"])
            for b in blocks[-3:]
        ]
        disagreement["recent_disagreement"] = previous_pairs[-1] if previous_pairs else current_pairwise
        baseline_pair = previous_pairs[-1] if previous_pairs else current_pairwise
        disagreement["disagreement_change_rate"] = float(
            (current_pairwise - baseline_pair) / max(abs(baseline_pair), 0.05)
        )
        disagreement["rolling_disagreement"] = float(
            np.mean(previous_pairs + [current_pairwise])
        )
        current_regime = _block_regime(test)
        same_regime_pairs = [
            float(b["state"]["disagreement"]["pairwise_disagreement"])
            for b in blocks
            if b["regime"] == current_regime
        ]
        disagreement["regime_conditioned_disagreement"] = float(
            np.mean(same_regime_pairs[-6:] + [current_pairwise])
        ) if same_regime_pairs else current_pairwise
        recent_cal = float(np.mean([
            b["full_architecture"]["calibration_error"] for b in blocks[-3:]
        ])) if blocks else 0.0
        older_cal = float(np.mean([
            b["full_architecture"]["calibration_error"] for b in blocks[-6:-3]
        ])) if len(blocks) >= 6 else recent_cal
        prior_pair = previous_pairs[-1] if previous_pairs else current_pairwise
        drift = drift_features(
            test, prior_rows, panel, prior_panel,
            current_disagreement=current_pairwise,
            prior_disagreement=prior_pair,
            recent_calibration=recent_cal,
            older_calibration=older_cal,
        )
        quality_logloss = {
            e: float(np.mean([
                blocks[j]["metrics"][e]["logloss"]
                for j in range(max(0, len(blocks) - PAST_WINDOW), len(blocks))
            ])) if blocks else float(metrics_now[e]["logloss"])
            for e in EXPERTS
        }
        provisional_state = {
            "disagreement": disagreement,
            "drift": drift,
            "quality_logloss": quality_logloss,
        }
        if len(blocks) >= holdout_cut:
            if len(blocks) == holdout_cut:
                frozen_failure_models, frozen_predict_model, frozen_meta_counts = _fit_meta_models(
                    blocks, holdout_cut
                )
                prior_p = [
                    np.asarray(b["routes"]["full_architecture"], dtype=float)
                    for b in blocks
                ]
                prior_y = [y for b in blocks for y in b["y"]]
                frozen_temperature = (
                    _temperature(np.vstack(prior_p), prior_y)
                    if len(prior_y) >= 50 else 1.0
                )
                frozen_reference_scores = [
                    score for b in blocks for score in b.get("selective_scores", [])
                ]
            failure_models = frozen_failure_models or {}
            predict_model = frozen_predict_model
            meta_counts = frozen_meta_counts or {
                "failure_samples": {e: 0 for e in EXPERTS},
                "predictability_samples": 0,
            }
        else:
            failure_models, predict_model, meta_counts = _fit_meta_models(blocks, len(blocks))
        failure_risks = _failure_risks(provisional_state, quality_logloss, failure_models)
        predictability = _predictability(provisional_state, predict_model)

        routes = {}
        route_specs = {
            "soft_ensemble": dict(use_disagreement=False, use_predictability=False, use_failure=False, use_drift=False),
            "adaptive_ensemble": dict(use_disagreement=False, use_predictability=False, use_failure=False, use_drift=False),
            "disagreement_model": dict(use_disagreement=True, use_predictability=False, use_failure=False, use_drift=False),
            "predictability_model": dict(use_disagreement=False, use_predictability=True, use_failure=False, use_drift=False),
            "future_failure_predictor": dict(use_disagreement=False, use_predictability=False, use_failure=True, use_drift=False),
            "drift_aware_router": dict(use_disagreement=False, use_predictability=False, use_failure=False, use_drift=True),
            "full_architecture": dict(use_disagreement=True, use_predictability=True, use_failure=True, use_drift=True),
        }
        for name, spec in route_specs.items():
            if name == "soft_ensemble":
                w = np.full(len(EXPERTS), 1.0 / len(EXPERTS))
            else:
                w = _route_weights(
                    quality_logloss,
                    disagreement,
                    drift,
                    predictability,
                    failure_risks,
                    previous=prior_weights if name == "full_architecture" else None,
                    **spec,
                )
            routes[name] = _route_probs(panel, w)

        # The fixed production artifact is benchmarked separately on a PIT-safe
        # post-training period; it is not used to train or tune this router.
        champion = _champion_benchmark(horizon, rows) if bi == 0 else None
        block = {
            "index": len(blocks),
            "test_start": test[0]["created"],
            "test_end": test[-1]["created"],
            "n": len(test),
            "regime": _block_regime(test),
            "panel": panel,
            "metrics": metrics_now,
            "state": {
                "disagreement": disagreement,
                "drift": drift,
                "quality_logloss": quality_logloss,
                "predictability": predictability,
                "failure_risks": failure_risks,
                "meta_counts": meta_counts,
            },
            "routes": {k: v.tolist() for k, v in routes.items()},
        }
        for name, probs in routes.items():
            m = metrics([r["y"] for r in test], probs)
            block[name] = m
        block["calibrated_full_temperature"] = 1.0
        calibrated_full = routes["full_architecture"]
        if routes["full_architecture"].size:
            if len(blocks) >= holdout_cut and frozen_reference_scores is not None:
                t = frozen_temperature
                calibrated_full = _apply_temperature(routes["full_architecture"], t)
                block["calibrated_full_temperature"] = t
                block["full_architecture_calibrated"] = metrics(
                    [r["y"] for r in test], calibrated_full
                )
            else:
                prior_p = []
                prior_y = []
                for old in blocks[-min(8, len(blocks)):]:
                    prior_p.append(np.asarray(old["routes"]["full_architecture"], dtype=float))
                    prior_y.extend(old["y"])
                if prior_p and len(prior_y) >= 50:
                    flat_p = np.vstack(prior_p)
                    t = _temperature(flat_p, prior_y)
                    calibrated_full = _apply_temperature(routes["full_architecture"], t)
                    block["calibrated_full_temperature"] = t
                    block["full_architecture_calibrated"] = metrics(
                        [r["y"] for r in test], calibrated_full
                    )
        block["routes"]["full_architecture_calibrated"] = calibrated_full.tolist()
        block["y"] = [r["y"] for r in test]
        block["selective_score"] = _selective_score(
            calibrated_full, predictability, failure_risks, drift
        )
        conf = calibrated_full.max(axis=1)
        block["selective_scores"] = (
            0.70 * conf
            + 0.30 * float(predictability)
            - 0.20 * float(drift["drift_score"])
            - 0.20 * float(np.mean(list(failure_risks.values())))
        ).clip(0.0, 1.0).tolist()
        prior_selective_scores.append(block["selective_score"])
        blocks.append(block)
        prior_weights = np.asarray(
            _route_weights(
                quality_logloss,
                disagreement,
                drift,
                predictability,
                failure_risks,
                previous=prior_weights,
            ),
            dtype=float,
        )

    if len(blocks) < 12:
        return {"status": "DEFERRED", "reason": "insufficient_valid_oos_blocks", "n": len(rows), "blocks": len(blocks)}

    # Recompute final holdout boundary descriptively only. It is never used for
    # any meta fit, route tuning, or calibration selection in this pass.
    holdout_n = max(1, int(len(blocks) * FINAL_HOLDOUT_FRAC))
    dev_blocks = blocks[:-holdout_n]
    hold_blocks = blocks[-holdout_n:]

    names = ("soft_ensemble", "adaptive_ensemble", "disagreement_model", "predictability_model", "future_failure_predictor", "drift_aware_router", "full_architecture")
    summary = {}
    for name in names:
        m = [b[name] for b in dev_blocks]
        summary[name] = {
            "n": int(sum(x["n"] for x in m)),
            "accuracy": float(sum(x["n"] * x["accuracy"] for x in m) / sum(x["n"] for x in m)),
            "logloss": float(sum(x["n"] * x["logloss"] for x in m) / sum(x["n"] for x in m)),
            "brier": float(sum(x["n"] * x["brier"] for x in m) / sum(x["n"] for x in m)),
            "ece": float(sum(x["n"] * x["calibration_error"] for x in m) / sum(x["n"] for x in m)),
            "blocks": len(m),
        }
    full = summary["full_architecture"]
    soft = summary["soft_ensemble"]
    deltas = {
        "accuracy": full["accuracy"] - soft["accuracy"],
        "logloss": full["logloss"] - soft["logloss"],
        "brier": full["brier"] - soft["brier"],
        "ece": full["ece"] - soft["ece"],
    }
    coverage = {}
    all_scores = np.asarray([
        score for b in dev_blocks for score in b.get("selective_scores", [])
    ], dtype=float)
    for target in (1.0, 0.95, 0.90, 0.80, 0.70):
        threshold = float(np.quantile(all_scores, max(0.0, 1.0 - target))) if len(all_scores) else 0.0
        ps, ys = [], []
        for b in dev_blocks:
            p = np.asarray(b["routes"]["full_architecture_calibrated"], dtype=float)
            row_scores = np.asarray(b.get("selective_scores", []), dtype=float)
            keep = row_scores >= threshold
            ps.extend(p[keep])
            ys.extend([y for y, k in zip(b["y"], keep) if k])
        coverage[f"{int(target*100)}%"] = _coverage_metrics(
            np.asarray(ps), ys, 0.0
        ) if ys else {"coverage": 0.0, "n": 0}
        coverage[f"{int(target*100)}%"]["target_coverage"] = target
        coverage[f"{int(target*100)}%"]["threshold"] = threshold

    block_deltas = [b["full_architecture"]["logloss"] - b["soft_ensemble"]["logloss"] for b in dev_blocks]
    stat = {
        "full_vs_soft_logloss_delta": float(np.mean(block_deltas)),
        "full_vs_soft_logloss_ci95": _bootstrap_ci(block_deltas),
        "full_vs_soft_accuracy_delta": float(np.mean([b["full_architecture"]["accuracy"] - b["soft_ensemble"]["accuracy"] for b in dev_blocks])),
        "full_vs_soft_brier_delta": float(np.mean([b["full_architecture"]["brier"] - b["soft_ensemble"]["brier"] for b in dev_blocks])),
        "method": "moving_block_bootstrap",
    }

    # Offline shadow/challenger replay: same frozen block decisions, no live mutation.
    holdout_result = {}
    for name in ("soft_ensemble", "full_architecture"):
        hs = [b[name] for b in hold_blocks]
        holdout_result[name] = {
            "n": int(sum(x["n"] for x in hs)),
            "accuracy": float(sum(x["n"] * x["accuracy"] for x in hs) / sum(x["n"] for x in hs)),
            "logloss": float(sum(x["n"] * x["logloss"] for x in hs) / sum(x["n"] for x in hs)),
            "brier": float(sum(x["n"] * x["brier"] for x in hs) / sum(x["n"] for x in hs)),
            "ece": float(sum(x["n"] * x["calibration_error"] for x in hs) / sum(x["n"] for x in hs)),
        }
    shadow = {
        "status": "OFFLINE_REPLAY_ONLY",
        "production_unchanged": True,
        "holdout_descriptive_only": True,
        "production_live_shadow_run": False,
        "challenger_live_run": False,
    }

    last = blocks[-1]
    stress = _stress_test({
        "weights": _route_weights(
            last["state"]["quality_logloss"],
            last["state"]["disagreement"],
            last["state"]["drift"],
            last["state"]["predictability"],
            last["state"]["failure_risks"],
        ).tolist(),
        "quality_logloss": last["state"]["quality_logloss"],
        "disagreement": last["state"]["disagreement"],
        "drift": last["state"]["drift"],
        "predictability": last["state"]["predictability"],
        "failure_risks": last["state"]["failure_risks"],
    })

    production_eligible = False
    promotion_reason = "research_archive_is_not_live_binance_primary_and_v2_requires_independent_longer_evidence"
    results = {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "horizon": horizon,
        "n_rows": len(rows),
        "blocks": len(blocks),
        "development_blocks": len(dev_blocks),
        "final_holdout_blocks": len(hold_blocks),
        "final_holdout_protected": True,
        "oos_summary": summary,
        "deltas_vs_soft": deltas,
        "period_breakdown": _period_breakdown(dev_blocks),
        "selective_prediction": coverage,
        "statistical_validation": stat,
        "offline_holdout": holdout_result,
        "shadow": shadow,
        "stress_test": stress,
        "champion_benchmark": _champion_benchmark(horizon, rows),
        "meta_leakage_policy": {
            "failure_labels_require_future_window_completion_before_meta_fit": True,
            "predictability_labels_require_future_window_completion_before_meta_fit": True,
            "current_block_labels_not_used_for_routing": True,
            "frozen_holdout_used_for_selection": False,
        },
        "promotion": {
            "eligible": production_eligible,
            "decision": "HOLD",
            "reason": promotion_reason,
        },
        "blocks_detail": [
            {
                k: v for k, v in b.items()
                if k not in {"panel", "y", "routes"}
            }
            for b in blocks
        ],
    }
    return results


def write_outputs(result):
    horizon = result["horizon"]
    prefix = OUT_DIR / f"innovative_control_{horizon}"
    (prefix.with_name(prefix.name + "_oos.json")).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (prefix.with_name(prefix.name + "_ablation.json")).write_text(
        json.dumps({
            "schema_version": 1,
            "research_only": True,
            "horizon": horizon,
            "models": result.get("oos_summary", {}),
            "deltas_vs_soft": result.get("deltas_vs_soft", {}),
            "period_breakdown": result.get("period_breakdown", {}),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (prefix.with_name(prefix.name + "_statistical_validation.json")).write_text(
        json.dumps(result.get("statistical_validation", {}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (prefix.with_name(prefix.name + "_stress_test.json")).write_text(
        json.dumps(result.get("stress_test", {}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (prefix.with_name(prefix.name + "_selective_policy.json")).write_text(
        json.dumps(result.get("selective_prediction", {}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (prefix.with_name(prefix.name + "_shadow_results.json")).write_text(
        json.dumps(result.get("shadow", {}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    state = result.get("blocks_detail", [])
    last_state = state[-1] if state else {}
    (prefix.with_name(prefix.name + "_disagreement_features.json")).write_text(
        json.dumps({
            "schema_version": 1,
            "research_only": True,
            "horizon": horizon,
            "feature_family": "model_disagreement",
            "latest_block": last_state.get("index"),
            "latest_disagreement": last_state.get("state", {}).get("disagreement", {}),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (prefix.with_name(prefix.name + "_predictability_model.json")).write_text(
        json.dumps({
            "schema_version": 1,
            "research_only": True,
            "horizon": horizon,
            "policy": "prequential_logistic_easy_block_probability",
            "latest_score": last_state.get("state", {}).get("predictability"),
            "meta_samples": last_state.get("state", {}).get("meta_counts", {}).get("predictability_samples", 0),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (prefix.with_name(prefix.name + "_future_failure_model.json")).write_text(
        json.dumps({
            "schema_version": 1,
            "research_only": True,
            "horizon": horizon,
            "policy": "prequential_per_expert_future_failure_probability",
            "latest_risk": last_state.get("state", {}).get("failure_risks", {}),
            "meta_samples": last_state.get("state", {}).get("meta_counts", {}).get("failure_samples", {}),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (prefix.with_name(prefix.name + "_drift_detector.json")).write_text(
        json.dumps({
            "schema_version": 1,
            "research_only": True,
            "horizon": horizon,
            "latest_drift": last_state.get("state", {}).get("drift", {}),
            "regime": last_state.get("regime"),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (prefix.with_name(prefix.name + "_dynamic_router.json")).write_text(
        json.dumps({
            "schema_version": 1,
            "research_only": True,
            "horizon": horizon,
            "policy": "quality_x_predictability_x_disagreement_x_drift_x_failure_soft_routing",
            "models": list(result.get("oos_summary", {}).keys()),
            "promotion": result.get("promotion", {}),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (prefix.with_name(prefix.name + "_calibration_artifact.json")).write_text(
        json.dumps({
            "schema_version": 1,
            "research_only": True,
            "horizon": horizon,
            "method": "chronological_temperature_from_prior_routed_blocks",
            "latest_temperature": last_state.get("calibrated_full_temperature"),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "innovative_experiment_manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "experiment_family": "btc_innovative_prediction_control_v2",
            "generated_at_utc": utc_now(),
            "commit": "runtime-populated-by-workflow",
            "dataset": "free_binance_vision_or_verified_archive_rows",
            "feature_version": "production_15_feature_schema",
            "horizons": list(HORIZONS),
            "seed": SEED,
            "frozen_holdout_frac": FINAL_HOLDOUT_FRAC,
            "production_changed": False,
            "artifacts": [
                f"innovative_control_{h}_oos.json",
                f"innovative_control_{h}_ablation.json",
                f"innovative_control_{h}_statistical_validation.json",
                f"innovative_control_{h}_stress_test.json",
                f"innovative_control_{h}_selective_policy.json",
                f"innovative_control_{h}_shadow_results.json",
                f"innovative_control_{h}_disagreement_features.json",
                f"innovative_control_{h}_predictability_model.json",
                f"innovative_control_{h}_future_failure_model.json",
                f"innovative_control_{h}_drift_detector.json",
                f"innovative_control_{h}_dynamic_router.json",
                f"innovative_control_{h}_calibration_artifact.json",
            ],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for h in HORIZONS:
        try:
            outputs[h] = evaluate(h)
        except Exception as exc:
            outputs[h] = {
                "status": "DEFERRED",
                "horizon": h,
                "reason": f"evaluation_error:{type(exc).__name__}:{exc}",
                "research_only": True,
                "production_changed": False,
            }
    aggregate = {
        "schema_version": 1,
        "experiment": "btc_innovative_prediction_control_v2",
        "generated_at_utc": utc_now(),
        "research_only": True,
        "production_changed": False,
        "horizons": outputs,
    }
    (OUT_DIR / "innovative_experiment_registry.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for result in outputs.values():
        if result.get("status") == "OK":
            write_outputs(result)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
