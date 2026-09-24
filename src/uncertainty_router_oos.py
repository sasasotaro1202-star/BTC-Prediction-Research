"""Research-only uncertainty/disagreement-aware expert routing for BTC.

The router is strictly prequential: base experts are trained before a gap,
a prior meta block produces correctness labels, and the router is then frozen
for the next test block. Current/future labels never affect current routing.
No production artifact is modified or promoted.
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

from model_compare import load_archive_research_rows, metrics, CLASSES, PURGE_BARS, EMBARGO_BARS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "uncertainty_router_oos.json"

HORIZONS = ("5m", "10m")
EXPERTS = ("logreg", "extra_trees", "hgb", "soft_equal")
MIN_TRAIN = 3000
META_BLOCK = 600
TEST_BLOCK = 600
FINAL_HOLDOUT_FRAC = 0.20
MIN_META_ROWS = 300
MAX_ROWS = 12000
EPS = 1e-7


def _align(model, rows):
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
            n_estimators=140,
            max_depth=10,
            min_samples_leaf=15,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=180,
            max_leaf_nodes=15,
            learning_rate=0.04,
            l2_regularization=1.5,
            random_state=42,
        ),
    }


def _fit_experts(train):
    X = np.asarray([r["x"] for r in train], dtype=float)
    y = np.asarray([r["y"] for r in train], dtype=str)
    if len(train) < MIN_TRAIN or len(set(y.tolist())) < 3:
        raise ValueError("insufficient_or_single_class_training_rows")
    models = {}
    for name, factory in _factories().items():
        m = factory()
        m.fit(X, y)
        models[name] = m
    return models


def _expert_probs(models, rows):
    parts = {name: _align(model, rows) for name, model in models.items()}
    parts["soft_equal"] = np.mean(
        np.stack([parts["logreg"], parts["extra_trees"], parts["hgb"]], axis=0),
        axis=0,
    )
    parts["soft_equal"] = np.clip(parts["soft_equal"], EPS, 1.0)
    return parts


def _entropy(p):
    return float(-np.sum(np.clip(p, EPS, 1.0) * np.log(np.clip(p, EPS, 1.0))))


def _router_features(probs, rows, include_context):
    names = ("logreg", "extra_trees", "hgb")
    stacked = np.stack([probs[n] for n in names], axis=0)
    argmaxes = stacked.argmax(axis=2)
    features = []
    for i in range(len(rows)):
        row = []
        for expert in EXPERTS:
            p = probs[expert][i]
            ordered = np.sort(p)[::-1]
            predicted = int(np.argmax(p))
            others = [n for n in names if n != ("soft_equal" if expert == "soft_equal" else expert)]
            disagree = sum(int(np.argmax(probs[n][i]) != predicted) for n in others)
            same_class_probs = [float(probs[n][i, predicted]) for n in names]
            row.extend([
                float(p[0]), float(p[1]), float(p[2]),
                float(np.max(p)),
                float(ordered[0] - ordered[1]),
                _entropy(p),
                float(disagree),
                float(np.std(same_class_probs)),
                float(np.mean(same_class_probs)),
            ])
        if include_context:
            x = rows[i]["x"]
            # Current 15-feature schema: selected regime/context proxies.
            row.extend([
                float(x[2]),  # ret_5m
                float(x[3]),  # ret_10m
                float(x[5]),  # volatility_5m
                float(x[6]),  # volatility_10m
                float(x[7]),  # range_position_10m
                float(x[11]), # volume_ratio
                float(x[12]), # volume_trend
            ])
        features.append(row)
    out = np.asarray(features, dtype=float)
    if out.ndim != 2 or not np.isfinite(out).all():
        raise ValueError("router_feature_matrix_invalid")
    return out


def _correctness_labels(probs, rows, expert):
    pred = np.argmax(probs[expert], axis=1)
    y = np.asarray([CLASSES.index(r["y"]) for r in rows], dtype=int)
    return (pred == y).astype(int)


def _fit_router(meta_features, meta_probs, meta_rows, mode):
    X = _router_features(meta_probs, meta_rows, include_context=(mode == "context"))
    models = {}
    for j, expert in enumerate(EXPERTS):
        y = _correctness_labels(meta_probs, meta_rows, expert)
        if len(set(y.tolist())) < 2:
            continue
        m = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.2, max_iter=1600, class_weight="balanced")),
        ])
        m.fit(X, y)
        models[expert] = m
    return models


def _route(test_features, test_probs, router_models, rows):
    if not router_models:
        chosen = np.full(len(rows), "soft_equal", dtype=object)
        return test_probs["soft_equal"].copy(), chosen

    scores = np.full((len(rows), len(EXPERTS)), -np.inf, dtype=float)
    for j, expert in enumerate(EXPERTS):
        model = router_models.get(expert)
        if model is None:
            continue
        scores[:, j] = model.predict_proba(test_features)[:, 1]
    choices = np.argmax(scores, axis=1)
    chosen = np.asarray(EXPERTS, dtype=object)[choices]
    out = np.empty((len(rows), 3), dtype=float)
    for i, expert in enumerate(chosen):
        out[i] = test_probs[str(expert)][i]
    out = np.clip(out, EPS, 1.0)
    out /= out.sum(axis=1, keepdims=True)
    return out, chosen


def _metrics(y, probs):
    result = metrics(y, probs)
    result["accuracy"] = float(result["accuracy"])
    return {
        "n": int(len(y)),
        "accuracy": float(result["accuracy"]),
        "logloss": float(result["logloss"]),
        "brier": float(result["brier"]),
        "ece": float(result.get("ece", result.get("calibration_error", math.nan))),
    }


def _block_features(rows, probs):
    f = _router_features(probs, rows, include_context=True)
    uncertainty = np.stack(
        [[
            _entropy(probs["soft_equal"][i]),
            float(np.sort(probs["soft_equal"][i])[-1] - np.sort(probs["soft_equal"][i])[-2]),
            float(np.mean([
                np.argmax(probs[n][i]) != np.argmax(probs["soft_equal"][i])
                for n in ("logreg", "extra_trees", "hgb")
            ])),
        ] for i in range(len(rows))]
    )
    return f, uncertainty


def _evaluate_mode(development, mode, horizon):
    gap = int(PURGE_BARS[horizon] + EMBARGO_BARS[horizon])
    outputs = []
    trace = []
    start = MIN_TRAIN + META_BLOCK + gap
    for test_start in range(start, len(development), TEST_BLOCK):
        test_end = min(test_start + TEST_BLOCK, len(development))
        test = development[test_start:test_end]
        meta_end = test_start - gap
        meta_start = meta_end - META_BLOCK
        base_train = development[:meta_start]
        meta = development[meta_start:meta_end]
        if len(test) < TEST_BLOCK // 2 or len(meta) < MIN_META_ROWS or len(base_train) < MIN_TRAIN:
            continue

        pre_models = _fit_experts(base_train)
        meta_probs = _expert_probs(pre_models, meta)
        meta_X = _router_features(meta_probs, meta, include_context=(mode == "context"))
        router_models = _fit_router(meta_X, meta_probs, meta, mode)

        test_train = development[:test_start - gap]
        test_models = _fit_experts(test_train)
        test_probs = _expert_probs(test_models, test)
        test_X = _router_features(test_probs, test, include_context=(mode == "context"))
        routed, chosen = _route(test_X, test_probs, router_models, test)

        y = [r["y"] for r in test]
        baseline = test_probs["soft_equal"]
        m_base = _metrics(y, baseline)
        m_router = _metrics(y, routed)
        entropy, disagree = _block_features(test, test_probs)[1][:, 0], _block_features(test, test_probs)[1][:, 2]
        high_uncert = entropy >= float(np.quantile(entropy, 0.75))
        high_disagree = disagree >= 0.66
        def sub(mask):
            return {
                "n": int(mask.sum()),
                "baseline_accuracy": float(np.mean(np.argmax(baseline[mask], axis=1) == np.asarray([CLASSES.index(v) for v in np.asarray(y)[mask]]))) if mask.any() else None,
                "router_accuracy": float(np.mean(np.argmax(routed[mask], axis=1) == np.asarray([CLASSES.index(v) for v in np.asarray(y)[mask]]))) if mask.any() else None,
            }
        outputs.append({
            "n": len(test),
            "baseline": m_base,
            "router": m_router,
            "delta": {
                "accuracy": m_router["accuracy"] - m_base["accuracy"],
                "logloss": m_router["logloss"] - m_base["logloss"],
                "brier": m_router["brier"] - m_base["brier"],
                "ece": m_router["ece"] - m_base["ece"],
            },
            "chosen": {expert: float(np.mean(chosen == expert)) for expert in EXPERTS},
            "high_uncertainty": sub(high_uncert),
            "high_disagreement": sub(high_disagree),
        })
        trace.append({"test_start": test_start, "chosen": {e: float(np.mean(chosen == e)) for e in EXPERTS}})
    if not outputs:
        return None

    total_n = sum(x["n"] for x in outputs)
    # Aggregate using block metrics weighted by sample count, while retaining
    # per-block stability for an adoption gate.
    def weighted(key):
        return float(sum(x["n"] * x[key][field] for x in outputs for field in []))
    # Reconstruct aggregate losses from block metrics.
    agg = {}
    for side in ("baseline", "router"):
        agg[side] = {}
        for field in ("accuracy", "logloss", "brier", "ece"):
            agg[side][field] = float(sum(x["n"] * x[side][field] for x in outputs) / total_n)
        agg[side]["n"] = total_n
    deltas = {
        field: agg["router"][field] - agg["baseline"][field]
        for field in ("accuracy", "logloss", "brier", "ece")
    }
    ll_improved = float(np.mean([x["delta"]["logloss"] < 0 for x in outputs]))
    br_improved = float(np.mean([x["delta"]["brier"] < 0 for x in outputs]))
    non_worse_acc = float(np.mean([x["delta"]["accuracy"] >= -0.005 for x in outputs]))
    ll_rel = (agg["baseline"]["logloss"] - agg["router"]["logloss"]) / max(abs(agg["baseline"]["logloss"]), EPS)
    br_rel = (agg["baseline"]["brier"] - agg["router"]["brier"]) / max(abs(agg["baseline"]["brier"]), EPS)
    acc_rel = (agg["router"]["accuracy"] - agg["baseline"]["accuracy"]) / max(abs(agg["baseline"]["accuracy"]), EPS)
    return {
        "mode": mode,
        "blocks": len(outputs),
        "samples": total_n,
        "aggregate": {"baseline": agg["baseline"], "router": agg["router"], "delta": deltas},
        "relative_improvement": {
            "accuracy": acc_rel,
            "logloss": ll_rel,
            "brier": br_rel,
        },
        "stability": {
            "improved_logloss_ratio": ll_improved,
            "improved_brier_ratio": br_improved,
            "non_worse_accuracy_ratio": non_worse_acc,
        },
        "eligibility": bool(
            len(outputs) >= 8
            and (ll_rel >= 0.03 or acc_rel >= 0.03)
            and br_rel >= 0.01
            and non_worse_acc >= 0.70
        ),
        "blocks_detail": outputs,
        "trace": trace,
    }


def _final_holdout(development, holdout, mode, horizon):
    gap = int(PURGE_BARS[horizon] + EMBARGO_BARS[horizon])
    meta_end = len(development) - gap
    meta_start = max(MIN_TRAIN, meta_end - META_BLOCK)
    base_train = development[:meta_start]
    meta = development[meta_start:meta_end]
    if len(base_train) < MIN_TRAIN or len(meta) < MIN_META_ROWS or len(holdout) < 100:
        return {"status": "DEFERRED", "reason": "insufficient_holdout_router_training"}
    pre_models = _fit_experts(base_train)
    meta_probs = _expert_probs(pre_models, meta)
    router_models = _fit_router(
        _router_features(meta_probs, meta, include_context=(mode == "context")),
        meta_probs,
        meta,
        mode,
    )
    test_train = development
    test_models = _fit_experts(test_train)
    hold_probs = _expert_probs(test_models, holdout)
    hold_X = _router_features(hold_probs, holdout, include_context=(mode == "context"))
    routed, chosen = _route(hold_X, hold_probs, router_models, holdout)
    y = [r["y"] for r in holdout]
    baseline = _metrics(y, hold_probs["soft_equal"])
    router = _metrics(y, routed)
    return {
        "status": "OK",
        "n": len(holdout),
        "baseline": baseline,
        "router": router,
        "delta": {
            "accuracy": router["accuracy"] - baseline["accuracy"],
            "logloss": router["logloss"] - baseline["logloss"],
            "brier": router["brier"] - baseline["brier"],
            "ece": router["ece"] - baseline["ece"],
        },
        "chosen": {expert: float(np.mean(chosen == expert)) for expert in EXPERTS},
    }


def evaluate(horizon):
    rows = load_archive_research_rows(horizon, MAX_ROWS)
    if len(rows) < MIN_TRAIN + META_BLOCK + TEST_BLOCK + 100:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_archive_rows",
            "n": len(rows),
        }
    split = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    development = rows[:split]
    holdout = rows[split:]
    results = {}
    for mode in ("uncertainty", "context"):
        try:
            result = _evaluate_mode(development, mode, horizon)
        except Exception as exc:
            result = {"status": "DEFERRED", "reason": f"evaluation_error:{type(exc).__name__}:{exc}"}
        results[mode] = result
    for mode in ("uncertainty", "context"):
        if results[mode] is not None:
            results[f"{mode}_holdout"] = _final_holdout(development, holdout, mode, horizon)
    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "final_holdout_protected": True,
        "final_holdout_used_for_selection": False,
        "strict_point_in_time_archive_replay": False,
        "horizon": horizon,
        "n": len(rows),
        "development_n": len(development),
        "final_holdout_n": len(holdout),
        "base_experts": list(EXPERTS),
        "gap_bars": int(PURGE_BARS[horizon] + EMBARGO_BARS[horizon]),
        "modes": results,
        "policy": (
            "prequential_router_from_prior_meta_block_only; "
            "uncertainty_ablation_vs_context; holdout_descriptive_only"
        ),
    }


def main():
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "horizons": {h: evaluate(h) for h in HORIZONS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
