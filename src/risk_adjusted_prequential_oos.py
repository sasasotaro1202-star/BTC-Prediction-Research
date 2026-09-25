"""Research-only risk-adjusted prequential expert routing for BTC.

The current observation is routed using only prediction-time probabilities,
uncertainty/disagreement features, and reliability state learned from prior
settled observations. No production artifact is modified or promoted.
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

from model_compare import (
    CLASSES,
    EMBARGO_BARS,
    PURGE_BARS,
    load_archive_research_rows,
    metrics,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "risk_adjusted_prequential_oos.json"

HORIZONS = ("5m", "10m")
EXPERTS = ("logreg", "extra_trees", "hgb", "soft_equal")
MODEL_EXPERTS = ("logreg", "extra_trees", "hgb")
MIN_TRAIN = 3000
META_BLOCK = 500
TEST_BLOCK = 500
FINAL_HOLDOUT_FRAC = 0.20
MIN_META_ROWS = 250
MAX_ROWS = 12000
EWMA_ALPHA = 0.15
ETA = 3.0
RISK_FLOOR = 0.25
RISK_CAP = 1.00
EPS = 1e-7


def _align(model: Any, rows: list[dict[str, Any]]) -> np.ndarray:
    X = np.asarray([r["x"] for r in rows], dtype=float)
    raw = np.asarray(model.predict_proba(X), dtype=float)
    out = np.full((len(rows), 3), EPS, dtype=float)
    index = {name: i for i, name in enumerate(CLASSES)}
    for j, cls in enumerate(model.classes_):
        if str(cls) in index:
            out[:, index[str(cls)]] = raw[:, j]
    out = np.clip(out, EPS, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def _factories() -> dict[str, Any]:
    return {
        "logreg": lambda: Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.3, max_iter=2500)),
            ]
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=160,
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


def _fit_experts(train: list[dict[str, Any]]) -> dict[str, Any]:
    if len(train) < MIN_TRAIN or len({r["y"] for r in train}) < 3:
        raise ValueError("insufficient_or_single_class_training_rows")
    X = np.asarray([r["x"] for r in train], dtype=float)
    y = np.asarray([r["y"] for r in train], dtype=str)
    out = {}
    for name, factory in _factories().items():
        model = factory()
        model.fit(X, y)
        out[name] = model
    return out


def _expert_probs(models: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    parts = {name: _align(model, rows) for name, model in models.items()}
    parts["soft_equal"] = np.mean(
        np.stack([parts[name] for name in MODEL_EXPERTS], axis=0),
        axis=0,
    )
    parts["soft_equal"] = np.clip(parts["soft_equal"], EPS, 1.0)
    parts["soft_equal"] /= parts["soft_equal"].sum(axis=1, keepdims=True)
    return parts


def _uncertainty_features(probs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    stack = np.stack([probs[name] for name in MODEL_EXPERTS], axis=0)
    soft = probs["soft_equal"]
    confidence = soft.max(axis=1)
    ordered = np.sort(soft, axis=1)[:, ::-1]
    margin = ordered[:, 0] - ordered[:, 1]
    entropy = -np.sum(soft * np.log(np.clip(soft, EPS, 1.0)), axis=1) / math.log(3.0)
    disagreement = np.mean(
        np.sum((stack - soft[None, :, :]) ** 2, axis=2), axis=0
    )
    disagreement = np.clip(disagreement / 0.10, 0.0, 1.0)
    model_argmax = np.argmax(stack, axis=2)
    soft_argmax = np.argmax(soft, axis=1)
    agreement = np.mean(model_argmax == soft_argmax[None, :], axis=0)
    global_uncertainty = np.clip(
        0.50 * entropy + 0.30 * disagreement + 0.20 * (1.0 - margin),
        0.0,
        1.0,
    )
    return {
        "confidence": confidence,
        "margin": margin,
        "entropy": entropy,
        "disagreement": disagreement,
        "agreement": agreement,
        "uncertainty": global_uncertainty,
    }


def _expert_feature_matrix(probs: dict[str, np.ndarray], rows: list[dict[str, Any]], expert: str) -> np.ndarray:
    feat = _uncertainty_features(probs)
    p = probs[expert]
    ordered = np.sort(p, axis=1)[:, ::-1]
    own_entropy = -np.sum(p * np.log(np.clip(p, EPS, 1.0)), axis=1) / math.log(3.0)
    own_margin = ordered[:, 0] - ordered[:, 1]
    own_confidence = ordered[:, 0]
    pred = np.argmax(p, axis=1)
    peers = [n for n in MODEL_EXPERTS if n != expert]
    peer_same = np.mean([probs[n][np.arange(len(rows)), pred] for n in peers], axis=0)
    x = np.asarray([r["x"] for r in rows], dtype=float)
    return np.column_stack(
        [
            own_confidence,
            own_margin,
            own_entropy,
            feat["confidence"],
            feat["margin"],
            feat["entropy"],
            feat["disagreement"],
            feat["agreement"],
            feat["uncertainty"],
            np.abs(own_confidence - peer_same),
            x[:, 2],  # ret_5m
            x[:, 3],  # ret_10m
            x[:, 5],  # volatility_5m
            x[:, 6],  # volatility_10m
            x[:, 7],  # range_position_10m
            x[:, 11], # volume_ratio
            x[:, 12], # volume_trend
        ]
    ).astype(float)


def _fit_reliability_models(
    probs: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    labels = np.asarray([CLASSES.index(r["y"]) for r in rows], dtype=int)
    models: dict[str, Any] = {}
    for expert in MODEL_EXPERTS:
        pred = np.argmax(probs[expert], axis=1)
        correct = (pred == labels).astype(int)
        if len(np.unique(correct)) < 2:
            continue
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("risk", LogisticRegression(C=0.2, max_iter=1800, class_weight="balanced")),
            ]
        )
        model.fit(_expert_feature_matrix(probs, rows, expert), correct)
        models[expert] = model
    return models


def _reliability_probs(
    probs: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    models: dict[str, Any],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    fallback = np.clip(1.0 - _uncertainty_features(probs)["uncertainty"], 0.0, 1.0)
    for expert in MODEL_EXPERTS:
        model = models.get(expert)
        if model is None:
            out[expert] = fallback.copy()
        else:
            pred = np.asarray(
                model.predict_proba(_expert_feature_matrix(probs, rows, expert))[:, 1],
                dtype=float,
            )
            if not np.isfinite(pred).all():
                raise ValueError(f"reliability_probability_nonfinite:{expert}")
            out[expert] = np.clip(pred, 0.0, 1.0)
    out["soft_equal"] = np.mean(
        np.stack([out[name] for name in MODEL_EXPERTS], axis=0), axis=0
    )
    return out


def _new_state() -> dict[str, Any]:
    return {
        "ema": {expert: 0.0 for expert in EXPERTS},
        "seen": 0,
    }


def _route(
    probs: dict[str, np.ndarray],
    reliability: dict[str, np.ndarray],
    state: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    uncertainty = _uncertainty_features(probs)["uncertainty"]
    weights = np.zeros((len(next(iter(probs.values()))), len(EXPERTS)), dtype=float)
    for j, expert in enumerate(EXPERTS):
        prior_loss = float(state["ema"][expert])
        rel = np.asarray(reliability[expert], dtype=float)
        # Reliability score is multiplicative rather than a direct probability
        # replacement, limiting damage when the risk model is misspecified.
        quality = RISK_FLOOR + (RISK_CAP - RISK_FLOOR) * rel
        # In high uncertainty, place less weight on experts with poor learned
        # reliability; do not automatically flatten the final prediction.
        quality *= 1.0 - 0.25 * uncertainty
        weights[:, j] = math.exp(-ETA * prior_loss) * quality

    weights_sum = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, np.maximum(weights_sum, EPS))
    matrix = np.stack([probs[e] for e in EXPERTS], axis=1)
    routed = np.sum(weights[:, :, None] * matrix, axis=1)
    routed = np.clip(routed, EPS, 1.0)
    routed /= routed.sum(axis=1, keepdims=True)
    return routed, weights


def _update_state(
    probs: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    state: dict[str, Any],
) -> None:
    for i, row in enumerate(rows):
        for expert in EXPERTS:
            p = probs[expert][i]
            idx = CLASSES.index(row["y"])
            loss = -math.log(max(float(p[idx]), EPS))
            state["ema"][expert] = (
                (1.0 - EWMA_ALPHA) * state["ema"][expert] + EWMA_ALPHA * loss
            )
        state["seen"] += 1


def _metrics(y: list[str], p: np.ndarray) -> dict[str, float | int]:
    raw = metrics(y, p)
    return {
        "n": int(len(y)),
        "accuracy": float(raw["accuracy"]),
        "logloss": float(raw["logloss"]),
        "brier": float(raw["brier"]),
        "ece": float(raw.get("ece", raw.get("calibration_error", math.nan))),
    }


def _evaluate_development(rows: list[dict[str, Any]], horizon: str) -> dict[str, Any]:
    gap = int(PURGE_BARS[horizon] + EMBARGO_BARS[horizon])
    outputs: list[dict[str, Any]] = []
    state = _new_state()

    start = MIN_TRAIN + META_BLOCK + gap
    for test_start in range(start, len(rows), TEST_BLOCK):
        test_end = min(test_start + TEST_BLOCK, len(rows))
        test = rows[test_start:test_end]
        meta_end = test_start - gap
        meta_start = meta_end - META_BLOCK
        base_train = rows[:meta_start]
        meta = rows[meta_start:meta_end]
        if len(test) < TEST_BLOCK // 2 or len(meta) < MIN_META_ROWS or len(base_train) < MIN_TRAIN:
            continue

        # Reliability models are fitted only on the earlier meta block.
        meta_models = _fit_experts(base_train)
        meta_probs = _expert_probs(meta_models, meta)
        reliability_models = _fit_reliability_models(meta_probs, meta)

        train_rows = rows[: test_start - gap]
        test_models = _fit_experts(train_rows)
        test_probs = _expert_probs(test_models, test)
        test_reliability = _reliability_probs(test_probs, test, reliability_models)

        baseline = test_probs["soft_equal"]
        routed, weights = _route(test_probs, test_reliability, state)
        y = [r["y"] for r in test]

        outputs.append(
            {
                "n": len(test),
                "baseline": _metrics(y, baseline),
                "candidate": _metrics(y, routed),
                "delta": {
                    "accuracy": _metrics(y, routed)["accuracy"] - _metrics(y, baseline)["accuracy"],
                    "logloss": _metrics(y, routed)["logloss"] - _metrics(y, baseline)["logloss"],
                    "brier": _metrics(y, routed)["brier"] - _metrics(y, baseline)["brier"],
                    "ece": _metrics(y, routed)["ece"] - _metrics(y, baseline)["ece"],
                },
                "mean_weights": {
                    expert: float(weights[:, j].mean()) for j, expert in enumerate(EXPERTS)
                },
            }
        )
        # IMPORTANT: update state only after the current block prediction is fixed.
        _update_state(test_probs, test, state)

    if not outputs:
        return {
            "status": "DEFERRED",
            "reason": "no_valid_blocks",
            "blocks": 0,
        }

    total = sum(item["n"] for item in outputs)
    def agg(side: str, field: str) -> float:
        return float(sum(item["n"] * item[side][field] for item in outputs) / total)

    aggregate = {
        "baseline": {field: agg("baseline", field) for field in ("accuracy", "logloss", "brier", "ece")},
        "candidate": {field: agg("candidate", field) for field in ("accuracy", "logloss", "brier", "ece")},
    }
    delta = {
        field: aggregate["candidate"][field] - aggregate["baseline"][field]
        for field in ("accuracy", "logloss", "brier", "ece")
    }
    ll_improved = float(np.mean([item["delta"]["logloss"] < 0.0 for item in outputs]))
    br_improved = float(np.mean([item["delta"]["brier"] < 0.0 for item in outputs]))
    non_worse_acc = float(np.mean([item["delta"]["accuracy"] >= -0.005 for item in outputs]))
    ll_rel = (aggregate["baseline"]["logloss"] - aggregate["candidate"]["logloss"]) / max(aggregate["baseline"]["logloss"], EPS)
    br_rel = (aggregate["baseline"]["brier"] - aggregate["candidate"]["brier"]) / max(aggregate["baseline"]["brier"], EPS)

    return {
        "status": "OK",
        "blocks": len(outputs),
        "samples": total,
        "aggregate": {**aggregate, "delta": delta},
        "relative_improvement": {
            "logloss": ll_rel,
            "brier": br_rel,
            "accuracy": (aggregate["candidate"]["accuracy"] - aggregate["baseline"]["accuracy"]) / max(aggregate["baseline"]["accuracy"], EPS),
        },
        "stability": {
            "improved_logloss_ratio": ll_improved,
            "improved_brier_ratio": br_improved,
            "non_worse_accuracy_ratio": non_worse_acc,
        },
        "eligible": bool(
            len(outputs) >= 8
            and (ll_rel >= 0.03 or (aggregate["candidate"]["accuracy"] - aggregate["baseline"]["accuracy"]) >= 0.01)
            and br_rel >= 0.01
            and non_worse_acc >= 0.70
        ),
        "blocks_detail": outputs,
    }


def _evaluate_holdout(rows: list[dict[str, Any]], horizon: str) -> dict[str, Any]:
    split = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    development = rows[:split]
    holdout = rows[split:]
    gap = int(PURGE_BARS[horizon] + EMBARGO_BARS[horizon])
    meta_end = len(development) - gap
    meta_start = max(MIN_TRAIN, meta_end - META_BLOCK)
    base_train = development[:meta_start]
    meta = development[meta_start:meta_end]
    if len(base_train) < MIN_TRAIN or len(meta) < MIN_META_ROWS or len(holdout) < 200:
        return {"status": "DEFERRED", "reason": "insufficient_holdout_training"}

    meta_models = _fit_experts(base_train)
    meta_probs = _expert_probs(meta_models, meta)
    reliability_models = _fit_reliability_models(meta_probs, meta)

    # The complete development period is available before the holdout starts.
    # Carry its settled-loss state forward, while fitting reliability models only
    # on the earlier meta block. No holdout outcome is used before routing.
    hold_models = _fit_experts(development)
    hold_probs = _expert_probs(hold_models, holdout)
    hold_reliability = _reliability_probs(hold_probs, holdout, reliability_models)

    development_probs = _expert_probs(hold_models, development)
    state = _new_state()
    _update_state(development_probs, development, state)
    baseline = hold_probs["soft_equal"]
    candidate, weights = _route(hold_probs, hold_reliability, state)
    y = [r["y"] for r in holdout]
    base = _metrics(y, baseline)
    cand = _metrics(y, candidate)
    return {
        "status": "OK",
        "n": len(holdout),
        "baseline": base,
        "candidate": cand,
        "delta": {
            "accuracy": cand["accuracy"] - base["accuracy"],
            "logloss": cand["logloss"] - base["logloss"],
            "brier": cand["brier"] - base["brier"],
            "ece": cand["ece"] - base["ece"],
        },
        "mean_weights": {
            expert: float(weights[:, j].mean()) for j, expert in enumerate(EXPERTS)
        },
    }


def evaluate(horizon: str) -> dict[str, Any]:
    rows = load_archive_research_rows(horizon, MAX_ROWS)
    if len(rows) < MIN_TRAIN + META_BLOCK + TEST_BLOCK + 200:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_archive_rows",
            "n": len(rows),
        }
    development_end = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    development = rows[:development_end]
    dev = _evaluate_development(development, horizon)
    hold = _evaluate_holdout(rows, horizon)
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
        "final_holdout_n": len(rows) - development_end,
        "config": {
            "ewma_alpha": EWMA_ALPHA,
            "eta": ETA,
            "risk_floor": RISK_FLOOR,
            "meta_block": META_BLOCK,
            "test_block": TEST_BLOCK,
        },
        "development": dev,
        "final_holdout": hold,
        "policy": (
            "prior_meta_reliability_models_plus_prequential_loss_state; "
            "current_outcome_applied_only_after_current_block_prediction; "
            "no_production_promotion"
        ),
    }


def main() -> None:
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "horizons": {horizon: evaluate(horizon) for horizon in HORIZONS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
