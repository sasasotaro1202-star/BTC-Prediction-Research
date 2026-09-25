"""Research-only prequential expert routing for BTC short-horizon direction.

The router predicts first, then incorporates the realized outcome into its
state. Expert weights therefore use only information available before the
current event. The production Champion remains unchanged.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
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
    MIN_OOS,
    MIN_TRAIN,
    PURGE_BARS,
    _temperature,
    aligned,
    apply_temperature,
    load_archive_research_rows,
    metrics,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "prequential_reliability_routing_oos.json"

HORIZONS = ("5m", "10m")
MAX_ROWS = 12000
FINAL_HOLDOUT_FRAC = 0.20
TEST_BLOCK = 500
CAL_BLOCK = 500
GAP_EXTRA = 0

EMA_ALPHA = 0.05
ETA = 2.0
CONTEXT_SHRINK_K = 30.0
WARMUP_EVENTS = 50
EPS = 1e-7

EXPERTS = ("production", "logreg", "extra_trees", "hgb")


def _prob_vec(value: Any) -> np.ndarray:
    p = np.asarray(value, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("invalid_probability_vector")
    total = float(p.sum())
    if total <= 0:
        raise ValueError("probability_vector_nonpositive")
    p = np.clip(p, EPS, 1.0)
    return p / p.sum()


def _feature_context(x: list[float]) -> str:
    # Deterministic context only; no threshold is learned from test outcomes.
    ret5 = float(x[2])
    ret10 = float(x[3])
    vol5 = float(x[5])
    vol10 = float(x[6])
    range10 = float(x[7])
    if ret5 > 0 and ret10 > 0:
        trend = "UP"
    elif ret5 < 0 and ret10 < 0:
        trend = "DOWN"
    else:
        trend = "MIXED"
    volatility = "HIGH" if vol10 >= max(vol5, EPS) else "LOW"
    if range10 < 1 / 3:
        location = "LOW"
    elif range10 > 2 / 3:
        location = "HIGH"
    else:
        location = "MID"
    return f"{trend}|{volatility}|{location}"


def _loss(p: np.ndarray, y: str) -> float:
    i = CLASSES.index(y)
    return float(-math.log(max(float(p[i]), EPS)))


def _update(
    global_ema: dict[str, float],
    context_ema: dict[str, dict[str, float]],
    context_counts: dict[str, int],
    context: str,
    probs: dict[str, np.ndarray],
    y: str,
) -> None:
    for expert in EXPERTS:
        value = _loss(probs[expert], y)
        global_ema[expert] = (1 - EMA_ALPHA) * global_ema[expert] + EMA_ALPHA * value
        context_ema[context][expert] = (
            (1 - EMA_ALPHA) * context_ema[context][expert] + EMA_ALPHA * value
        )
    context_counts[context] += 1


def _weights(
    global_ema: dict[str, float],
    context_ema: dict[str, dict[str, float]],
    context_counts: dict[str, int],
    context: str,
    strategy: str,
    seen: int,
) -> np.ndarray:
    if seen < WARMUP_EVENTS:
        return np.full(len(EXPERTS), 1.0 / len(EXPERTS))
    if strategy == "static_production":
        out = np.zeros(len(EXPERTS), dtype=float)
        out[0] = 1.0
        return out
    if strategy == "static_equal":
        return np.full(len(EXPERTS), 1.0 / len(EXPERTS))
    if strategy not in {"online_global", "online_context"}:
        raise ValueError(f"unknown_strategy:{strategy}")

    estimates: list[float] = []
    for expert in EXPERTS:
        if strategy == "online_global":
            estimate = global_ema[expert]
        else:
            count = context_counts[context]
            local = context_ema[context][expert]
            estimate = (
                (count / (count + CONTEXT_SHRINK_K)) * local
                + (CONTEXT_SHRINK_K / (count + CONTEXT_SHRINK_K)) * global_ema[expert]
            )
        estimates.append(estimate)

    scores = -ETA * np.asarray(estimates, dtype=float)
    scores -= np.max(scores)
    weights = np.exp(scores)
    weights /= weights.sum()
    return weights


def _route_rows(
    rows: list[dict[str, Any]],
    strategy: str,
    initial_state: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    state = initial_state or {
        "global_ema": {expert: math.log(3.0) for expert in EXPERTS},
        "context_ema": defaultdict(lambda: {expert: math.log(3.0) for expert in EXPERTS}),
        "context_counts": defaultdict(int),
        "seen": 0,
    }
    preds: list[np.ndarray] = []
    weights_trace: list[np.ndarray] = []
    labels: list[int] = []

    for row in rows:
        context = _feature_context(row["x"])
        probs = {expert: _prob_vec(row["production"]) if expert == "production" else row["expert_probs"][expert] for expert in EXPERTS}
        weights = _weights(
            state["global_ema"],
            state["context_ema"],
            state["context_counts"],
            context,
            strategy,
            int(state["seen"]),
        )
        matrix = np.stack([probs[expert] for expert in EXPERTS], axis=0)
        prediction = np.sum(weights[:, None] * matrix, axis=0)
        prediction = np.clip(prediction, EPS, 1.0)
        prediction /= prediction.sum()

        preds.append(prediction)
        weights_trace.append(weights)
        labels.append(CLASSES.index(row["y"]))

        # Causal order: update only after current prediction has been fixed.
        _update(
            state["global_ema"],
            state["context_ema"],
            state["context_counts"],
            context,
            probs,
            row["y"],
        )
        state["seen"] = int(state["seen"]) + 1

    if not preds:
        raise ValueError("empty_route_rows")
    return np.stack(preds), np.asarray(labels, dtype=int), state, [
        {"weights": w.tolist()} for w in weights_trace
    ]


def _factories() -> dict[str, Any]:
    return {
        "logreg": lambda: Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.3, max_iter=2500)),
            ]
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=240,
            max_depth=9,
            min_samples_leaf=12,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=200,
            max_leaf_nodes=15,
            learning_rate=0.04,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


def _fit_alternatives(train: list[dict[str, Any]], cal: list[dict[str, Any]]) -> dict[str, Any]:
    if len(train) < MIN_TRAIN or len(cal) < 100:
        raise ValueError("insufficient_training_or_calibration_rows")
    X_train = np.asarray([r["x"] for r in train], dtype=float)
    y_train = np.asarray([r["y"] for r in train], dtype=str)
    X_cal = np.asarray([r["x"] for r in cal], dtype=float)
    y_cal = np.asarray([r["y"] for r in cal], dtype=str)
    if len(set(y_train)) < 3 or len(set(y_cal)) < 3:
        raise ValueError("single_class_train_or_calibration")

    fitted: dict[str, Any] = {}
    for name, factory in _factories().items():
        calibration_model = factory()
        calibration_model.fit(X_train, y_train)
        cal_probs = aligned(calibration_model, X_cal)
        temperature = _temperature(cal_probs, y_cal)

        model = factory()
        model.fit(X_train, y_train)
        fitted[name] = (model, float(temperature))
    return fitted


def _predict_alternatives(models: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    X = np.asarray([r["x"] for r in rows], dtype=float)
    out: dict[str, np.ndarray] = {}
    for name, (model, temperature) in models.items():
        p = aligned(model, X)
        p = apply_temperature(p, temperature)
        out[name] = p
    return out


def _prepare_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in raw_rows:
        out.append(
            {
                "id": row["id"],
                "x": row["x"],
                "y": row["y"],
                "production": _prob_vec(row["production"]),
            }
        )
    return out


def _augment_with_models(rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], models: dict[str, Any]) -> None:
    probs = _predict_alternatives(models, test_rows)
    for name in ("logreg", "extra_trees", "hgb"):
        for row, p in zip(test_rows, probs[name]):
            if "expert_probs" not in row:
                row["expert_probs"] = {}
            row["expert_probs"][name] = p


def _metrics_from_labels(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    y = [CLASSES[int(i)] for i in labels.tolist()]
    m = metrics(y, probs)
    return {
        "n": int(len(y)),
        "accuracy": float(m["accuracy"]),
        "logloss": float(m["logloss"]),
        "brier": float(m["brier"]),
        "ece": float(m.get("calibration_error", np.nan)),
    }


def _bootstrap_ci(values: np.ndarray, seed: int = 42, n_boot: int = 1000) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return {"lower": float("nan"), "mean": float(np.mean(values)), "upper": float("nan")}
    rng = np.random.default_rng(seed)
    sample_index = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[sample_index].mean(axis=1)
    q = np.quantile(means, [0.025, 0.975])
    return {"lower": float(q[0]), "mean": float(values.mean()), "upper": float(q[1])}


def _run_frozen_holdout(
    development: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    horizon: str,
    strategy: str,
    initial_state: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(development) < MIN_TRAIN + CAL_BLOCK or not holdout:
        return {"status": "DEFERRED"}, []

    gap = PURGE_BARS[horizon] + EMBARGO_BARS[horizon]
    cal_end = len(development) - gap
    cal_start = cal_end - CAL_BLOCK
    train = development[:cal_start]
    cal = development[cal_start:cal_end]
    if len(train) < MIN_TRAIN or len(cal) < 100:
        return {"status": "DEFERRED"}, []

    models = _fit_alternatives(train, cal)
    hold_copy = [dict(r) for r in holdout]
    _augment_with_models(holdout[:], hold_copy, models)

    probs, labels, _, trace = _route_rows(
        hold_copy,
        strategy,
        initial_state=initial_state,
    )
    baseline = np.stack([r["production"] for r in hold_copy], axis=0)
    baseline_metrics = _metrics_from_labels(labels, baseline)
    candidate_metrics = _metrics_from_labels(labels, probs)
    return (
        {
            "status": "OK",
            "candidate": candidate_metrics,
            "baseline": baseline_metrics,
            "delta": {
                metric: float(candidate_metrics[metric] - baseline_metrics[metric])
                for metric in ("accuracy", "logloss", "brier", "ece")
            },
            "mean_production_weight": float(np.mean([t["weights"][0] for t in trace])),
        },
        [
            {
                "n": int(len(hold_copy)),
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "delta": {
                    metric: float(candidate_metrics[metric] - baseline_metrics[metric])
                    for metric in ("accuracy", "logloss", "brier", "ece")
                },
            }
        ],
    )


def _run_prequential(
    development: list[dict[str, Any]],
    horizon: str,
    strategy: str,
    initial_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    state = initial_state
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []
    block_rows: list[dict[str, Any]] = []

    for test_start in range(MIN_TRAIN + CAL_BLOCK, len(development), TEST_BLOCK):
        test_end = min(test_start + TEST_BLOCK, len(development))
        test = development[test_start:test_end]
        gap = PURGE_BARS[horizon] + EMBARGO_BARS[horizon] + GAP_EXTRA
        fit_end = test_start - gap
        cal_end = fit_end
        cal_start = max(0, cal_end - CAL_BLOCK)
        train = development[:cal_start]
        cal = development[cal_start:cal_end]
        if len(train) < MIN_TRAIN or len(cal) < 100 or len(test) < max(50, TEST_BLOCK // 2):
            continue

        models = _fit_alternatives(train, cal)
        test_copy = [dict(r) for r in test]
        _augment_with_models(development[:], test_copy, models)

        probs, labels, state, trace = _route_rows(
            test_copy,
            strategy,
            initial_state=state,
        )
        baseline = np.stack([r["production"] for r in test_copy], axis=0)
        baseline_metrics = _metrics_from_labels(labels, baseline)
        candidate_metrics = _metrics_from_labels(labels, probs)
        block_rows.append(
            {
                "n": int(len(test)),
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "delta": {
                    k: float(candidate_metrics[k] - baseline_metrics[k])
                    for k in ("accuracy", "logloss", "brier", "ece")
                },
                "mean_production_weight": float(np.mean([t["weights"][0] for t in trace])),
            }
        )
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.tolist())

    if len(all_labels) < MIN_OOS:
        raise ValueError("insufficient_oos_samples")

    overall = _metrics_from_labels(np.asarray(all_labels), np.asarray(all_probs))
    total_baseline = np.stack([
        row["baseline"] for row in []
    ]) if False else None
    # Aggregate baseline metrics from block predictions are recomputed from block
    # deltas against the same row counts to avoid any averaging ambiguity.
    baseline_accuracy = sum(b["n"] * b["baseline"]["accuracy"] for b in block_rows) / sum(b["n"] for b in block_rows)
    baseline_logloss = sum(b["n"] * b["baseline"]["logloss"] for b in block_rows) / sum(b["n"] for b in block_rows)
    baseline_brier = sum(b["n"] * b["baseline"]["brier"] for b in block_rows) / sum(b["n"] for b in block_rows)
    baseline_ece = sum(b["n"] * b["baseline"]["ece"] for b in block_rows) / sum(b["n"] for b in block_rows)
    baseline_metrics = {
        "n": int(sum(b["n"] for b in block_rows)),
        "accuracy": baseline_accuracy,
        "logloss": baseline_logloss,
        "brier": baseline_brier,
        "ece": baseline_ece,
    }
    delta_arrays = {
        metric: np.asarray([b["delta"][metric] for b in block_rows], dtype=float)
        for metric in ("accuracy", "logloss", "brier", "ece")
    }
    return (
        {
            "candidate": overall,
            "baseline": baseline_metrics,
            "delta": {
                metric: float(overall[metric] - baseline_metrics[metric])
                for metric in ("accuracy", "logloss", "brier", "ece")
            },
            "block_stability": {
                "blocks": len(block_rows),
                "improved_logloss_ratio": float(np.mean(delta_arrays["logloss"] < 0)),
                "improved_brier_ratio": float(np.mean(delta_arrays["brier"] < 0)),
                "non_worse_accuracy_ratio": float(np.mean(delta_arrays["accuracy"] >= -0.005)),
                "ci95_block_logloss_delta": _bootstrap_ci(delta_arrays["logloss"]),
                "ci95_block_brier_delta": _bootstrap_ci(delta_arrays["brier"]),
                "ci95_block_accuracy_delta": _bootstrap_ci(delta_arrays["accuracy"]),
            },
        },
        block_rows,
        state,
    )


def _fresh_state() -> dict[str, Any]:
    return {
        "global_ema": {expert: math.log(3.0) for expert in EXPERTS},
        "context_ema": defaultdict(lambda: {expert: math.log(3.0) for expert in EXPERTS}),
        "context_counts": defaultdict(int),
        "seen": 0,
    }


def evaluate(horizon: str) -> dict[str, Any]:
    raw = load_archive_research_rows(horizon, MAX_ROWS)
    if len(raw) < MIN_TRAIN + MIN_OOS + CAL_BLOCK:
        return {
            "status": "DEFERRED",
            "research_only": True,
            "production_changed": False,
            "n": len(raw),
            "reason": "insufficient_archive_rows",
        }

    rows = _prepare_rows(raw)
    split = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    development, holdout = rows[:split], rows[split:]
    if len(development) < MIN_TRAIN + MIN_OOS:
        return {
            "status": "DEFERRED",
            "research_only": True,
            "production_changed": False,
            "n": len(rows),
            "reason": "insufficient_development_rows",
        }

    dev_result, dev_blocks, dev_state = _run_prequential(
        development,
        horizon,
        "online_context",
        initial_state=_fresh_state(),
    )

    # Frozen holdout uses one final expert fit frozen on development data.
    # Holdout outcomes affect routing only after each current prediction is fixed.
    hold_result, hold_blocks = _run_frozen_holdout(
        development,
        holdout,
        horizon,
        "online_context",
        initial_state=dev_state,
    )

    def eligible(result: dict[str, Any]) -> bool:
        if result.get("status") == "DEFERRED":
            return False
        d = result["delta"]
        s = result["block_stability"]
        return bool(
            d["logloss"] <= -0.003
            and d["brier"] <= -0.0015
            and d["accuracy"] >= -0.005
            and s["improved_logloss_ratio"] >= 0.70
            and s["improved_brier_ratio"] >= 0.70
            and s["non_worse_accuracy_ratio"] >= 0.80
            and s["ci95_block_logloss_delta"]["upper"] < 0.0
            and s["ci95_block_brier_delta"]["upper"] < 0.0
        )

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "final_holdout_protected": True,
        "final_holdout_used_for_selection": False,
        "data_source": "free_closed_binance_vision_research_rows",
        "horizon": horizon,
        "n": len(rows),
        "development_n": len(development),
        "final_holdout_n": len(holdout),
        "config": {
            "test_block": TEST_BLOCK,
            "calibration_block": CAL_BLOCK,
            "ema_alpha": EMA_ALPHA,
            "eta": ETA,
            "context_shrink_k": CONTEXT_SHRINK_K,
            "warmup_events": WARMUP_EVENTS,
            "experts": list(EXPERTS),
            "gap_bars": int(PURGE_BARS[horizon] + EMBARGO_BARS[horizon] + GAP_EXTRA),
        },
        "development": dev_result,
        "development_blocks": dev_blocks,
        "final_holdout": hold_result,
        "final_holdout_blocks": hold_blocks,
        "eligibility": eligible(dev_result) and (
            hold_result.get("status") == "OK"
            and hold_result["delta"]["logloss"] <= 0.0
            and hold_result["delta"]["brier"] <= 0.0
            and hold_result["delta"]["accuracy"] >= -0.005
        ),
        "policy": (
            "prequential causal weight updates; current outcome applied only after "
            "current prediction; hierarchical context shrink; frozen holdout protected"
        ),
    }


def main() -> None:
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
