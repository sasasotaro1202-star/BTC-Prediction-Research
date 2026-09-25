"""Research-only causal benefit gate for BTC routing.

Production Champion remains default. The candidate is a stable equal-weight blend
of production plus research experts, but it may replace the Champion only when
prior-label-trained meta models predict both case-level accuracy benefit and
log-loss benefit. The current/future block is never used to fit its gate.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_compare import (
    CLASSES,
    EMBARGO_BARS,
    PURGE_BARS,
    load_archive_research_rows,
    metrics as core_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "tail_benefit_gated_routing_oos.json"
HORIZONS = ("5m", "10m")
EXPERTS = ("production", "logreg", "extra_trees", "hgb")
RESEARCH_EXPERTS = ("logreg", "extra_trees", "hgb")
MIN_TRAIN = 3000
META_BLOCK = 500
TEST_BLOCK = 500
MAX_ROWS = 12000
FINAL_HOLDOUT_FRAC = 0.20
UNCERTAINTY_QUANTILE = 0.75
BENEFIT_ACCURACY_THRESHOLD = 0.65
BENEFIT_LOGLOSS_THRESHOLD = 0.55
EPS = 1e-7


def metrics(y: list[str], probs: np.ndarray) -> dict[str, float | int]:
    raw = core_metrics(y, probs)
    return {
        "n": int(len(y)),
        "accuracy": float(raw["accuracy"]),
        "logloss": float(raw["logloss"]),
        "brier": float(raw["brier"]),
        "ece": float(raw["calibration_error"]),
    }


def _align(model: Any, rows: list[dict[str, Any]]) -> np.ndarray:
    X = np.asarray([r["x"] for r in rows], dtype=float)
    raw = np.asarray(model.predict_proba(X), dtype=float)
    out = np.full((len(rows), 3), EPS, dtype=float)
    idx = {c: i for i, c in enumerate(CLASSES)}
    for j, cls in enumerate(getattr(model, "classes_", [])):
        if str(cls) in idx:
            out[:, idx[str(cls)]] = raw[:, j]
    if out.shape != (len(rows), 3) or not np.isfinite(out).all():
        raise ValueError("expert_probability_invalid")
    out = np.clip(out, EPS, 1.0)
    out /= out.sum(axis=1, keepdims=True)
    return out


def _fit_experts(train: list[dict[str, Any]]) -> dict[str, Any]:
    if len(train) < MIN_TRAIN or len({r["y"] for r in train}) < 3:
        raise ValueError("insufficient_or_single_class_training_rows")
    X = np.asarray([r["x"] for r in train], dtype=float)
    y = np.asarray([r["y"] for r in train], dtype=str)
    factories = {
        "logreg": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.3, max_iter=2500, class_weight="balanced")),
        ]),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=140, max_depth=10, min_samples_leaf=15,
            max_features="sqrt", random_state=42, n_jobs=-1,
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=180, max_leaf_nodes=15, learning_rate=0.04,
            l2_regularization=1.5, random_state=42,
        ),
    }
    out = {}
    for name, factory in factories.items():
        m = factory()
        m.fit(X, y)
        out[name] = m
    return out


def _champion_probs(rows: list[dict[str, Any]], horizon: str) -> np.ndarray:
    mp = ROOT / "models" / f"{horizon}.joblib"
    meta_path = ROOT / "models" / f"{horizon}.json"
    if not mp.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"production_artifact_missing:{horizon}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("candidate") is True:
        raise ValueError(f"production_candidate_rejected:{horizon}")
    features = meta.get("features")
    if not isinstance(features, list) or len(features) != len(rows[0]["x"]):
        raise ValueError(f"production_feature_schema_invalid:{horizon}")
    model = joblib.load(mp)
    return _align(model, rows)


def _probs(models: dict[str, Any], rows: list[dict[str, Any]], horizon: str) -> dict[str, np.ndarray]:
    out = {k: _align(v, rows) for k, v in models.items()}
    out["production"] = _champion_probs(rows, horizon)
    return out


def _uncertainty(probs: dict[str, np.ndarray]) -> np.ndarray:
    stack = np.stack([probs[e] for e in EXPERTS], axis=0)
    champ = probs["production"]
    entropy = -np.sum(champ * np.log(np.clip(champ, EPS, 1.0)), axis=1) / math.log(3.0)
    ordered = np.sort(champ, axis=1)[:, ::-1]
    margin = ordered[:, 0] - ordered[:, 1]
    disagreement = np.mean(np.sum((stack - champ[None, :, :]) ** 2, axis=2), axis=0)
    disagreement = np.clip(disagreement / 0.10, 0.0, 1.0)
    votes = np.argmax(stack, axis=2)
    champ_vote = np.argmax(champ, axis=1)
    vote_disagreement = 1.0 - np.mean(votes == champ_vote[None, :], axis=0)
    return np.clip(
        0.45 * entropy + 0.25 * disagreement
        + 0.20 * (1.0 - margin) + 0.10 * vote_disagreement,
        0.0, 1.0,
    )


def _candidate(probs: dict[str, np.ndarray]) -> np.ndarray:
    parts = [probs["production"]] + [probs[e] for e in RESEARCH_EXPERTS]
    out = np.mean(np.stack(parts, axis=0), axis=0)
    out = np.clip(out, EPS, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def _gate_features(probs: dict[str, np.ndarray], rows: list[dict[str, Any]], candidate: np.ndarray) -> np.ndarray:
    p = probs["production"]
    ordered_p = np.sort(p, axis=1)[:, ::-1]
    ordered_c = np.sort(candidate, axis=1)[:, ::-1]
    stack = np.stack([probs[e] for e in EXPERTS], axis=0)
    disagreement = np.mean(np.sum((stack - p[None, :, :]) ** 2, axis=2), axis=0)
    agreement = np.mean(np.argmax(stack, axis=2) == np.argmax(p, axis=1)[None, :], axis=0)
    u = _uncertainty(probs)
    x = np.asarray([r["x"] for r in rows], dtype=float)
    return np.column_stack([
        p[:, 0], p[:, 1], p[:, 2],
        candidate[:, 0], candidate[:, 1], candidate[:, 2],
        ordered_p[:, 0] - ordered_p[:, 1],
        ordered_c[:, 0] - ordered_c[:, 1],
        u, disagreement, agreement,
        np.abs(candidate - p).max(axis=1),
        x[:, 2], x[:, 3], x[:, 5], x[:, 6], x[:, 7], x[:, 11], x[:, 12],
    ]).astype(float)


def _fit_benefit_models(X: np.ndarray, probs: dict[str, np.ndarray], rows: list[dict[str, Any]], candidate: np.ndarray):
    y_idx = np.asarray([CLASSES.index(r["y"]) for r in rows], dtype=int)
    pp = probs["production"]
    cp = candidate
    prod_correct = (np.argmax(pp, axis=1) == y_idx).astype(int)
    cand_correct = (np.argmax(cp, axis=1) == y_idx).astype(int)
    acc_gain = (cand_correct > prod_correct).astype(int)
    prod_loss = -np.log(np.clip(pp[np.arange(len(rows)), y_idx], EPS, 1.0))
    cand_loss = -np.log(np.clip(cp[np.arange(len(rows)), y_idx], EPS, 1.0))
    ll_gain = (cand_loss < prod_loss).astype(int)

    result = []
    for target in (acc_gain, ll_gain):
        if len(np.unique(target)) < 2:
            result.append(None)
            continue
        model = Pipeline([
            ("scale", StandardScaler()),
            ("gate", LogisticRegression(C=0.15, max_iter=2000, class_weight="balanced")),
        ])
        model.fit(X, target)
        result.append(model)
    return tuple(result)


def _p(model: Any, X: np.ndarray) -> np.ndarray:
    if model is None:
        return np.zeros(len(X), dtype=float)
    out = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if not np.isfinite(out).all():
        raise ValueError("benefit_probability_nonfinite")
    return np.clip(out, 0.0, 1.0)


def _evaluate_segment(rows, horizon: str, training_end: int, test_rows: list[dict[str, Any]], meta_rows: list[dict[str, Any]]):
    gap = int(PURGE_BARS[horizon] + EMBARGO_BARS[horizon])
    base_train = rows[: max(0, training_end - META_BLOCK - gap)]
    if len(base_train) < MIN_TRAIN or len(meta_rows) < META_BLOCK // 2 or len(test_rows) < TEST_BLOCK // 2:
        return None
    meta_models = _fit_experts(base_train)
    meta_probs = _probs(meta_models, meta_rows, horizon)
    meta_candidate = _candidate(meta_probs)
    acc_model, ll_model = _fit_benefit_models(
        _gate_features(meta_probs, meta_rows, meta_candidate),
        meta_probs, meta_rows, meta_candidate
    )
    if acc_model is None and ll_model is None:
        return None

    test_train = rows[: max(0, training_end - gap)]
    test_models = _fit_experts(test_train)
    test_probs = _probs(test_models, test_rows, horizon)
    test_candidate = _candidate(test_probs)
    X = _gate_features(test_probs, test_rows, test_candidate)
    u = _uncertainty(test_probs)
    threshold = float(np.quantile(_uncertainty(meta_probs), UNCERTAINTY_QUANTILE))
    acc_b = _p(acc_model, X)
    ll_b = _p(ll_model, X)
    gate = (u >= threshold) & (acc_b >= BENEFIT_ACCURACY_THRESHOLD) & (ll_b >= BENEFIT_LOGLOSS_THRESHOLD)

    baseline = test_probs["production"]
    final = baseline.copy()
    final[gate] = test_candidate[gate]
    y = [r["y"] for r in test_rows]
    return {
        "n": len(test_rows),
        "gate_n": int(gate.sum()),
        "gate_rate": float(gate.mean()),
        "threshold": threshold,
        "benefit_accuracy_mean": float(acc_b.mean()),
        "benefit_logloss_mean": float(ll_b.mean()),
        "baseline": metrics(y, baseline),
        "candidate": metrics(y, final),
        "gated_baseline": metrics([y[i] for i in np.flatnonzero(gate)], baseline[gate]) if gate.any() else None,
        "gated_candidate": metrics([y[i] for i in np.flatnonzero(gate)], final[gate]) if gate.any() else None,
    }


def _aggregate(blocks):
    total = sum(b["n"] for b in blocks)
    def agg(name, key):
        return float(sum(b[name][key] * b["n"] for b in blocks) / total)
    base = {k: agg("baseline", k) for k in ("accuracy", "logloss", "brier", "ece")}
    cand = {k: agg("candidate", k) for k in ("accuracy", "logloss", "brier", "ece")}
    deltas = {k: cand[k] - base[k] for k in base}
    acc_d = np.asarray([b["candidate"]["accuracy"] - b["baseline"]["accuracy"] for b in blocks])
    ll_d = np.asarray([b["candidate"]["logloss"] - b["baseline"]["logloss"] for b in blocks])
    br_d = np.asarray([b["candidate"]["brier"] - b["baseline"]["brier"] for b in blocks])
    return {
        "baseline": base, "candidate": cand, "delta": deltas,
        "relative_improvement": {
            "accuracy": deltas["accuracy"] / max(base["accuracy"], EPS),
            "logloss": -deltas["logloss"] / max(base["logloss"], EPS),
            "brier": -deltas["brier"] / max(base["brier"], EPS),
        },
        "stability": {
            "improved_accuracy_ratio": float(np.mean(acc_d > 0)),
            "non_worse_accuracy_ratio": float(np.mean(acc_d >= -0.005)),
            "improved_logloss_ratio": float(np.mean(ll_d < 0)),
            "improved_brier_ratio": float(np.mean(br_d < 0)),
        },
    }


def _final_holdout(rows, horizon, dev):
    split = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    development = rows[:split]
    holdout = rows[split:]
    gap = int(PURGE_BARS[horizon] + EMBARGO_BARS[horizon])
    meta_end = len(development) - gap
    meta_start = max(MIN_TRAIN, meta_end - META_BLOCK)
    base_train = development[:meta_start]
    meta = development[meta_start:meta_end]
    if len(base_train) < MIN_TRAIN or len(meta) < META_BLOCK // 2 or len(holdout) < 200:
        return {"status": "DEFERRED", "reason": "insufficient_holdout_training", "n": len(holdout)}

    meta_models = _fit_experts(base_train)
    meta_probs = _probs(meta_models, meta, horizon)
    meta_candidate = _candidate(meta_probs)
    acc_model, ll_model = _fit_benefit_models(
        _gate_features(meta_probs, meta, meta_candidate), meta_probs, meta, meta_candidate
    )
    hold_models = _fit_experts(development)
    hold_probs = _probs(hold_models, holdout, horizon)
    hold_candidate = _candidate(hold_probs)
    X = _gate_features(hold_probs, holdout, hold_candidate)
    threshold = float(np.quantile(_uncertainty(meta_probs), UNCERTAINTY_QUANTILE))
    acc_b = _p(acc_model, X)
    ll_b = _p(ll_model, X)
    gate = ( _uncertainty(hold_probs) >= threshold
             ) & (acc_b >= BENEFIT_ACCURACY_THRESHOLD) & (ll_b >= BENEFIT_LOGLOSS_THRESHOLD)
    baseline = hold_probs["production"]
    final = baseline.copy()
    final[gate] = hold_candidate[gate]
    y = [r["y"] for r in holdout]
    return {
        "status": "OK",
        "n": len(holdout), "gate_n": int(gate.sum()), "gate_rate": float(gate.mean()),
        "uncertainty_threshold": threshold,
        "baseline": metrics(y, baseline),
        "candidate": metrics(y, final),
        "delta": {k: float(metrics(y, final)[k] - metrics(y, baseline)[k]) for k in ("accuracy","logloss","brier","ece")},
        "gated_baseline": metrics([y[i] for i in np.flatnonzero(gate)], baseline[gate]) if gate.any() else None,
        "gated_candidate": metrics([y[i] for i in np.flatnonzero(gate)], final[gate]) if gate.any() else None,
        "used_development_labels_only": True,
    }


def evaluate(horizon: str) -> dict[str, Any]:
    rows = load_archive_research_rows(horizon, MAX_ROWS)
    if len(rows) < MIN_TRAIN + META_BLOCK + TEST_BLOCK + 200:
        return {"status": "DEFERRED", "reason": "insufficient_archive_rows", "n": len(rows)}
    split = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    development = rows[:split]
    blocks = []
    gap = int(PURGE_BARS[horizon] + EMBARGO_BARS[horizon])
    start = MIN_TRAIN + META_BLOCK + gap
    for test_start in range(start, len(development), TEST_BLOCK):
        test_end = min(test_start + TEST_BLOCK, len(development))
        test_rows = development[test_start:test_end]
        meta_end = test_start - gap
        meta_start = meta_end - META_BLOCK
        meta_rows = development[meta_start:meta_end]
        result = _evaluate_segment(development, horizon, test_start, test_rows, meta_rows)
        if result is not None:
            blocks.append(result)
    if len(blocks) < 8:
        return {"status": "DEFERRED", "reason": "insufficient_valid_oos_blocks", "blocks": len(blocks), "n": len(rows)}
    agg = _aggregate(blocks)
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
        "final_holdout_n": len(rows) - split,
        "candidate_policy": "Champion default; switch only on prior-meta high uncertainty + jointly predicted accuracy/logloss benefit",
        "thresholds": {
            "uncertainty_quantile": UNCERTAINTY_QUANTILE,
            "benefit_accuracy_probability": BENEFIT_ACCURACY_THRESHOLD,
            "benefit_logloss_probability": BENEFIT_LOGLOSS_THRESHOLD,
        },
        "development": agg | {"blocks_detail": blocks},
        "final_holdout": _final_holdout(rows, horizon, agg),
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
