"""Research-only conservative correction for high expert disagreement.

Unlike hard model switching, the candidate:
1. uses a prior meta block to estimate each expert's correctness;
2. forms a soft probability mixture from those estimates;
3. applies the mixture only when at least two of four experts disagree with
   the RF champion direction; otherwise it keeps the RF baseline unchanged.

Everything is chronological and research-only. Final holdout is descriptive.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_compare import (
    CLASSES,
    EMBARGO_BARS,
    PURGE_BARS,
    load_archive_research_rows,
    metrics,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "high_disagreement_soft_oos.json"

HORIZONS = ("5m", "10m")
EXPERTS = ("random_forest", "logreg", "extra_trees", "hgb")
MIN_TRAIN = 3000
META_BLOCK = 600
TEST_BLOCK = 600
FINAL_HOLDOUT_FRAC = 0.20
MIN_META_ROWS = 300
MAX_ROWS = 12000
DISAGREE_THRESHOLD = 0.50   # >=2/4 experts disagree with RF
SOFTMAX_TEMPERATURE = 2.0   # fixed; no tuning on test/holdout
EPS = 1e-7


def _factory(name: str):
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=180,
            max_depth=10,
            min_samples_leaf=15,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
    if name == "logreg":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.3, max_iter=2500)),
        ])
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=140,
            max_depth=10,
            min_samples_leaf=15,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=180,
            max_leaf_nodes=15,
            learning_rate=0.04,
            l2_regularization=1.5,
            random_state=42,
        )
    raise ValueError(f"unknown_expert:{name}")


def _align(model, rows):
    X = np.asarray([r["x"] for r in rows], dtype=float)
    raw = np.asarray(model.predict_proba(X), dtype=float)
    out = np.full((len(rows), 3), EPS, dtype=float)
    lookup = {c: i for i, c in enumerate(CLASSES)}
    for j, cls in enumerate(model.classes_):
        if str(cls) in lookup:
            out[:, lookup[str(cls)]] = raw[:, j]
    out = np.clip(out, EPS, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def _fit_experts(rows):
    y = np.asarray([r["y"] for r in rows], dtype=str)
    if len(rows) < MIN_TRAIN or len(set(y.tolist())) < 3:
        raise ValueError("insufficient_expert_training_rows")
    models = {}
    for name in EXPERTS:
        model = _factory(name)
        model.fit(np.asarray([r["x"] for r in rows], dtype=float), y)
        models[name] = model
    return models


def _expert_probs(models, rows):
    return {name: _align(models[name], rows) for name in EXPERTS}


def _router_features(probs):
    rf = probs["random_forest"]
    feat = []
    for i in range(len(rf)):
        row = []
        rf_pred = int(np.argmax(rf[i]))
        for name in EXPERTS:
            p = probs[name][i]
            ordered = np.sort(p)[::-1]
            row.extend([
                float(p[0]),
                float(p[1]),
                float(p[2]),
                float(ordered[0] - ordered[1]),
                float(-np.sum(np.clip(p, EPS, 1.0) * np.log(np.clip(p, EPS, 1.0)))),
                float(np.argmax(p) != rf_pred),
            ])
        feat.append(row)
    out = np.asarray(feat, dtype=float)
    if out.ndim != 2 or not np.isfinite(out).all():
        raise ValueError("router_features_invalid")
    return out


def _fit_correctness_models(meta_probs, meta_rows):
    X = _router_features(meta_probs)
    y_true = np.asarray([CLASSES.index(r["y"]) for r in meta_rows], dtype=int)
    models = {}
    for name in EXPERTS:
        y = (np.argmax(meta_probs[name], axis=1) == y_true).astype(int)
        if len(np.unique(y)) < 2:
            continue
        model = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=0.2,
                max_iter=1600,
                class_weight="balanced",
            )),
        ])
        model.fit(X, y)
        models[name] = model
    return models


def _soft_correction(test_probs, test_rows, router_models):
    X = _router_features(test_probs)
    n = len(test_rows)
    scores = np.zeros((n, len(EXPERTS)), dtype=float)
    for j, name in enumerate(EXPERTS):
        model = router_models.get(name)
        if model is not None:
            scores[:, j] = model.predict_proba(X)[:, 1]
    scaled = scores / SOFTMAX_TEMPERATURE
    scaled -= np.max(scaled, axis=1, keepdims=True)
    weights = np.exp(scaled)
    weights /= weights.sum(axis=1, keepdims=True)

    expert_stack = np.stack([test_probs[name] for name in EXPERTS], axis=1)
    mixed = np.sum(weights[:, :, None] * expert_stack, axis=1)

    rf_pred = np.argmax(test_probs["random_forest"], axis=1)
    disagreements = np.mean(
        np.stack([np.argmax(test_probs[name], axis=1) for name in EXPERTS], axis=1)
        != rf_pred[:, None],
        axis=1,
    )
    active = disagreements >= DISAGREE_THRESHOLD

    out = test_probs["random_forest"].copy()
    out[active] = mixed[active]
    out = np.clip(out, EPS, 1.0)
    out /= out.sum(axis=1, keepdims=True)
    return out, active, disagreements, weights


def _metrics(y, probs):
    m = metrics(y, probs)
    return {
        "n": int(len(y)),
        "accuracy": float(m["accuracy"]),
        "logloss": float(m["logloss"]),
        "brier": float(m["brier"]),
        "ece": float(m.get("ece", m.get("calibration_error", math.nan))),
    }


def _causal_train(rows, test_start, horizon):
    cutoff = datetime.fromisoformat(str(test_start).replace("Z", "+00:00"))
    cutoff = cutoff.astimezone(timezone.utc)
    gap = cutoff - timedelta(minutes=int(PURGE_BARS[horizon] + EMBARGO_BARS[horizon]))
    out = []
    for row in rows:
        created = datetime.fromisoformat(str(row["created"]).replace("Z", "+00:00"))
        target = datetime.fromisoformat(str(row["target"]).replace("Z", "+00:00"))
        created = created.astimezone(timezone.utc)
        target = target.astimezone(timezone.utc)
        if created < target and target < gap:
            out.append(row)
    return out


def _ci(values, seed=42, draws=2000):
    arr = np.asarray(values, dtype=float)
    if len(arr) < 4:
        return {"lower": None, "upper": None, "n": int(len(arr))}
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(draws, len(arr)), replace=True).mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
        "n": int(len(arr)),
    }


def evaluate(horizon: str):
    rows = load_archive_research_rows(horizon, MAX_ROWS)
    if len(rows) < MIN_TRAIN + META_BLOCK + TEST_BLOCK + 100:
        return {
            "status": "DEFERRED",
            "research_only": True,
            "production_changed": False,
            "promotion_allowed": False,
            "n": len(rows),
            "reason": "insufficient_archive_rows",
        }

    split = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    development = rows[:split]
    holdout = rows[split:]
    blocks = []

    for test_start in range(MIN_TRAIN + META_BLOCK, len(development), TEST_BLOCK):
        test = development[test_start:min(test_start + TEST_BLOCK, len(development))]
        if len(test) < TEST_BLOCK // 2:
            continue

        gap = PURGE_BARS[horizon] + EMBARGO_BARS[horizon]
        meta_end = test_start - gap
        meta_start = meta_end - META_BLOCK
        if meta_start < MIN_TRAIN:
            continue

        base_train = _causal_train(development[:meta_start], development[meta_start]["created"], horizon)
        meta = development[meta_start:meta_end]
        if len(base_train) < MIN_TRAIN or len(meta) < MIN_META_ROWS:
            continue

        meta_models = _fit_experts(base_train)
        meta_probs = _expert_probs(meta_models, meta)
        router_models = _fit_correctness_models(meta_probs, meta)

        test_train = _causal_train(development[:test_start], test[0]["created"], horizon)
        if len(test_train) < MIN_TRAIN:
            continue
        test_models = _fit_experts(test_train)
        test_probs = _expert_probs(test_models, test)
        candidate, active, disagreements, _ = _soft_correction(test_probs, test, router_models)
        y = [r["y"] for r in test]
        baseline = test_probs["random_forest"]

        mb = _metrics(y, baseline)
        mc = _metrics(y, candidate)
        active_mask = active
        high = {
            "n": int(active_mask.sum()),
            "candidate_accuracy": None,
            "baseline_accuracy": None,
            "candidate_logloss": None,
            "baseline_logloss": None,
        }
        if active_mask.any():
            high["candidate_accuracy"] = float(
                np.mean(np.argmax(candidate[active_mask], axis=1)
                        == np.asarray([CLASSES.index(v) for v in np.asarray(y)[active_mask]]))
            )
            high["baseline_accuracy"] = float(
                np.mean(np.argmax(baseline[active_mask], axis=1)
                        == np.asarray([CLASSES.index(v) for v in np.asarray(y)[active_mask]]))
            )
            yi = np.asarray([CLASSES.index(v) for v in np.asarray(y)[active_mask]], dtype=int)
            high["candidate_logloss"] = float(
                -np.mean(np.log(np.clip(candidate[active_mask, yi], EPS, 1.0)))
            )
            high["baseline_logloss"] = float(
                -np.mean(np.log(np.clip(baseline[active_mask, yi], EPS, 1.0)))
            )

        blocks.append({
            "n": len(test),
            "baseline": mb,
            "candidate": mc,
            "delta": {
                "accuracy": mc["accuracy"] - mb["accuracy"],
                "logloss": mc["logloss"] - mb["logloss"],
                "brier": mc["brier"] - mb["brier"],
                "ece": mc["ece"] - mb["ece"],
            },
            "high_disagreement": high,
            "coverage": float(active.mean()),
            "mean_disagreement": float(disagreements.mean()),
        })

    if len(blocks) < 8:
        return {
            "status": "DEFERRED",
            "research_only": True,
            "production_changed": False,
            "promotion_allowed": False,
            "n": len(rows),
            "blocks": len(blocks),
            "reason": "insufficient_oos_blocks",
        }

    d_ll = np.asarray([b["delta"]["logloss"] for b in blocks], dtype=float)
    d_br = np.asarray([b["delta"]["brier"] for b in blocks], dtype=float)
    d_ac = np.asarray([b["delta"]["accuracy"] for b in blocks], dtype=float)
    d_ece = np.asarray([b["delta"]["ece"] for b in blocks], dtype=float)

    dev = {
        "mean_delta": {
            "accuracy": float(d_ac.mean()),
            "logloss": float(d_ll.mean()),
            "brier": float(d_br.mean()),
            "ece": float(d_ece.mean()),
        },
        "mean_delta_ci95": {
            "accuracy": _ci(d_ac.tolist()),
            "logloss": _ci(d_ll.tolist()),
            "brier": _ci(d_br.tolist()),
            "ece": _ci(d_ece.tolist()),
        },
        "non_worse_accuracy_ratio": float(np.mean(d_ac >= -0.005)),
        "logloss_improved_ratio": float(np.mean(d_ll < 0)),
        "brier_improved_ratio": float(np.mean(d_br < 0)),
        "blocks": len(blocks),
        "samples": int(sum(b["n"] for b in blocks)),
    }

    holdout_train = _causal_train(development, holdout[0]["created"], horizon)
    if len(holdout_train) < MIN_TRAIN:
        return {
            "status": "DEFERRED",
            "research_only": True,
            "production_changed": False,
            "promotion_allowed": False,
            "n": len(rows),
            "blocks": len(blocks),
            "reason": "insufficient_purged_holdout_training_rows",
        }

    meta_end = max(MIN_TRAIN, len(holdout_train) - META_BLOCK)
    meta_start = max(MIN_TRAIN, meta_end - META_BLOCK)
    meta_base = _causal_train(holdout_train[:meta_start], holdout_train[meta_start]["created"], horizon)
    meta = holdout_train[meta_start:meta_end]
    meta_models = _fit_experts(meta_base)
    meta_probs = _expert_probs(meta_models, meta)
    router_models = _fit_correctness_models(meta_probs, meta)

    hold_models = _fit_experts(holdout_train)
    hold_probs = _expert_probs(hold_models, holdout)
    hold_candidate, hold_active, _, _ = _soft_correction(hold_probs, holdout, router_models)
    y_hold = [r["y"] for r in holdout]
    hold_base = _metrics(y_hold, hold_probs["random_forest"])
    hold_cand = _metrics(y_hold, hold_candidate)

    eligibility = bool(
        (
            dev["mean_delta"]["accuracy"] >= 0.03
            or dev["mean_delta"]["logloss"] <= -0.03
        )
        and dev["mean_delta"]["brier"] <= -0.006
        and dev["non_worse_accuracy_ratio"] >= 0.70
        and hold_cand["logloss"] <= hold_base["logloss"]
        and hold_cand["brier"] <= hold_base["brier"]
        and hold_cand["accuracy"] >= hold_base["accuracy"] - 0.005
    )

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "final_holdout_protected": True,
        "final_holdout_used_for_selection": False,
        "strict_point_in_time_archive_replay": False,
        "horizon": horizon,
        "n": len(rows),
        "development": dev,
        "final_holdout": {
            "n": len(holdout),
            "active_coverage": float(hold_active.mean()),
            "baseline": hold_base,
            "candidate": hold_cand,
            "delta": {
                "accuracy": hold_cand["accuracy"] - hold_base["accuracy"],
                "logloss": hold_cand["logloss"] - hold_base["logloss"],
                "brier": hold_cand["brier"] - hold_base["brier"],
                "ece": hold_cand["ece"] - hold_base["ece"],
            },
        },
        "config": {
            "experts": list(EXPERTS),
            "disagreement_threshold": DISAGREE_THRESHOLD,
            "softmax_temperature": SOFTMAX_TEMPERATURE,
            "purge_bars": int(PURGE_BARS[horizon]),
            "embargo_bars": int(EMBARGO_BARS[horizon]),
        },
        "eligibility_candidate": eligibility,
        "blocks": blocks,
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
