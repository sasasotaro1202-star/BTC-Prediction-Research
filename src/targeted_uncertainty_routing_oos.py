"""Research-only targeted uncertainty routing for BTC direction.

The current production Champion is preserved as the default baseline.
An alternative expert router is activated only when prediction-time
uncertainty is high relative to a strictly prior calibration block.
All learned reliability and loss state is causal; frozen holdout is descriptive.
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
OUT = ROOT / "data/historical_research/targeted_uncertainty_routing_oos.json"

HORIZONS = ("5m", "10m")
EXPERTS = ("production", "logreg", "extra_trees", "hgb")
ALTERNATIVE_EXPERTS = ("logreg", "extra_trees", "hgb")

MIN_TRAIN = 3000
META_BLOCK = 500
TEST_BLOCK = 500
MIN_META_ROWS = 250
MAX_ROWS = 12000
FINAL_HOLDOUT_FRAC = 0.20

# Pre-registered before OOS evaluation: activate only on the highest
# quartile of uncertainty observed in the immediately preceding meta block.
UNCERTAINTY_QUANTILE = 0.75

# Prequential reliability state. Outcome is applied only after a block is
# predicted, never before the current prediction.
EWMA_ALPHA = 0.15
ETA = 2.5
EPS = 1e-7


def metrics(y: list[str], probs: np.ndarray | list[list[float]]) -> dict[str, float | int]:
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
    index = {c: i for i, c in enumerate(CLASSES)}
    for j, cls in enumerate(getattr(model, "classes_", [])):
        if str(cls) in index:
            out[:, index[str(cls)]] = raw[:, j]
    if out.shape != (len(rows), 3) or not np.isfinite(out).all():
        raise ValueError("expert_probability_invalid")
    out = np.clip(out, EPS, 1.0)
    out /= out.sum(axis=1, keepdims=True)
    return out


def _factories() -> dict[str, Any]:
    return {
        "logreg": lambda: Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.3, max_iter=2500)),
            ]
        ),
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


def _fit(train: list[dict[str, Any]]) -> dict[str, Any]:
    if len(train) < MIN_TRAIN or len({r["y"] for r in train}) < 3:
        raise ValueError("insufficient_or_single_class_training_rows")
    X = np.asarray([r["x"] for r in train], dtype=float)
    y = np.asarray([r["y"] for r in train], dtype=str)
    models: dict[str, Any] = {}
    for name, factory in _factories().items():
        model = factory()
        model.fit(X, y)
        models[name] = model
    return models


def _champion_probs(rows: list[dict[str, Any]], horizon: str) -> np.ndarray:
    model_path = ROOT / "models" / f"{horizon}.joblib"
    meta_path = ROOT / "models" / f"{horizon}.json"
    if not model_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"production_champion_artifact_missing:{horizon}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("candidate") is True:
        raise ValueError(f"production_candidate_artifact_rejected:{horizon}")
    expected_features = meta.get("features")
    if not isinstance(expected_features, list) or len(expected_features) != len(rows[0]["x"]):
        raise ValueError(f"production_feature_schema_invalid:{horizon}")

    model = joblib.load(model_path)
    X = np.asarray([r["x"] for r in rows], dtype=float)
    raw = np.asarray(model.predict_proba(X), dtype=float)
    out = np.full((len(rows), 3), EPS, dtype=float)
    index = {c: i for i, c in enumerate(CLASSES)}
    for j, cls in enumerate(getattr(model, "classes_", [])):
        if str(cls) in index:
            out[:, index[str(cls)]] = raw[:, j]
    if out.shape != (len(rows), 3) or not np.isfinite(out).all():
        raise ValueError(f"production_champion_probability_invalid:{horizon}")
    out = np.clip(out, EPS, 1.0)
    out /= out.sum(axis=1, keepdims=True)
    return out


def _probs(models: dict[str, Any], rows: list[dict[str, Any]], horizon: str) -> dict[str, np.ndarray]:
    out = {name: _align(model, rows) for name, model in models.items()}
    out["production"] = _champion_probs(rows, horizon)
    return out


def _uncertainty(probs: dict[str, np.ndarray]) -> np.ndarray:
    stack = np.stack([probs[name] for name in EXPERTS], axis=0)
    champion = probs["production"]

    entropy = -np.sum(
        champion * np.log(np.clip(champion, EPS, 1.0)), axis=1
    ) / math.log(3.0)
    ordered = np.sort(champion, axis=1)[:, ::-1]
    margin = ordered[:, 0] - ordered[:, 1]

    disagreement = np.mean(
        np.sum((stack - champion[None, :, :]) ** 2, axis=2), axis=0
    )
    disagreement = np.clip(disagreement / 0.10, 0.0, 1.0)

    expert_votes = np.argmax(stack, axis=2)
    champion_vote = np.argmax(champion, axis=1)
    vote_disagreement = 1.0 - np.mean(expert_votes == champion_vote[None, :], axis=0)

    return np.clip(
        0.45 * entropy
        + 0.25 * disagreement
        + 0.20 * (1.0 - margin)
        + 0.10 * vote_disagreement,
        0.0,
        1.0,
    )


def _router_features(
    probs: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    expert: str,
) -> np.ndarray:
    p = probs[expert]
    ordered = np.sort(p, axis=1)[:, ::-1]
    pred = np.argmax(p, axis=1)
    peer_probs = np.stack(
        [
            probs[e][np.arange(len(rows)), pred]
            for e in EXPERTS
            if e != expert
        ],
        axis=0,
    )
    x = np.asarray([r["x"] for r in rows], dtype=float)
    uncertainty = _uncertainty(probs)

    return np.column_stack(
        [
            p[:, 0],
            p[:, 1],
            p[:, 2],
            ordered[:, 0] - ordered[:, 1],
            -np.sum(p * np.log(np.clip(p, EPS, 1.0)), axis=1) / math.log(3.0),
            np.mean(peer_probs, axis=0),
            np.abs(ordered[:, 0] - np.mean(peer_probs, axis=0)),
            uncertainty,
            x[:, 2],   # ret_5m
            x[:, 3],   # ret_10m
            x[:, 5],   # volatility_5m
            x[:, 6],   # volatility_10m
            x[:, 7],   # range_position_10m
            x[:, 11],  # volume_ratio
            x[:, 12],  # volume_trend
        ]
    ).astype(float)


def _fit_reliability_models(
    probs: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    labels = np.asarray([CLASSES.index(r["y"]) for r in rows], dtype=int)
    models: dict[str, Any] = {}
    X_cache = {
        expert: _router_features(probs, rows, expert)
        for expert in EXPERTS
    }
    for expert in EXPERTS:
        predicted = np.argmax(probs[expert], axis=1)
        correct = (predicted == labels).astype(int)
        if len(np.unique(correct)) < 2:
            continue
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.2,
                        max_iter=1800,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        model.fit(X_cache[expert], correct)
        models[expert] = model
    return models


def _reliability_prob(
    probs: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    models: dict[str, Any],
) -> dict[str, np.ndarray]:
    fallback = np.clip(1.0 - _uncertainty(probs), 0.0, 1.0)
    out: dict[str, np.ndarray] = {}
    for expert in EXPERTS:
        model = models.get(expert)
        if model is None:
            out[expert] = fallback.copy()
            continue
        pred = np.asarray(
            model.predict_proba(_router_features(probs, rows, expert))[:, 1],
            dtype=float,
        )
        if not np.isfinite(pred).all():
            raise ValueError(f"reliability_probability_nonfinite:{expert}")
        out[expert] = np.clip(pred, 0.0, 1.0)
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
    n = len(next(iter(probs.values())))
    weights = np.zeros((n, len(EXPERTS)), dtype=float)

    for j, expert in enumerate(EXPERTS):
        prior_loss = float(state["ema"][expert])
        prior_quality = math.exp(-ETA * prior_loss)
        reliability_quality = 0.50 + 0.50 * reliability[expert]
        weights[:, j] = prior_quality * reliability_quality

    weights /= np.maximum(weights.sum(axis=1, keepdims=True), EPS)

    matrix = np.stack([probs[expert] for expert in EXPERTS], axis=1)
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
        yi = CLASSES.index(row["y"])
        for expert in EXPERTS:
            loss = -math.log(max(float(probs[expert][i, yi]), EPS))
            state["ema"][expert] = (
                (1.0 - EWMA_ALPHA) * state["ema"][expert]
                + EWMA_ALPHA * loss
            )
        state["seen"] += 1


def _uncertainty_gate(meta_uncertainty: np.ndarray) -> float:
    values = np.asarray(meta_uncertainty, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 100:
        raise ValueError("insufficient_uncertainty_threshold_rows")
    threshold = float(np.quantile(values, UNCERTAINTY_QUANTILE))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("uncertainty_threshold_out_of_bounds")
    return threshold


def _aggregate(
    blocks: list[dict[str, Any]],
    side: str,
) -> dict[str, float | int]:
    total = sum(int(block["n"]) for block in blocks)
    return {
        key: float(
            sum(int(block["n"]) * float(block[side][key]) for block in blocks)
            / total
        )
        for key in ("accuracy", "logloss", "brier", "ece")
    } | {"n": total}


def _evaluate_development(
    rows: list[dict[str, Any]],
    horizon: str,
) -> dict[str, Any]:
    gap = int(PURGE_BARS[horizon] + EMBARGO_BARS[horizon])
    blocks: list[dict[str, Any]] = []
    state = _new_state()

    start = MIN_TRAIN + META_BLOCK + gap
    for test_start in range(start, len(rows), TEST_BLOCK):
        test_end = min(test_start + TEST_BLOCK, len(rows))
        test = rows[test_start:test_end]

        meta_end = test_start - gap
        meta_start = meta_end - META_BLOCK
        base_train = rows[:meta_start]
        meta = rows[meta_start:meta_end]

        if (
            len(test) < TEST_BLOCK // 2
            or len(meta) < MIN_META_ROWS
            or len(base_train) < MIN_TRAIN
        ):
            continue

        # Meta models use only data strictly before the meta block.
        meta_models = _fit(base_train)
        meta_probs = _probs(meta_models, meta, horizon)
        threshold = _uncertainty_gate(_uncertainty(meta_probs))
        reliability_models = _fit_reliability_models(meta_probs, meta)

        # Test experts train only before the purged test start.
        test_models = _fit(rows[: test_start - gap])
        test_probs = _probs(test_models, test, horizon)
        test_reliability = _reliability_prob(
            test_probs, test, reliability_models
        )

        routed, weights = _route(test_probs, test_reliability, state)
        uncertainty = _uncertainty(test_probs)
        gate = uncertainty >= threshold

        baseline = test_probs["production"]
        final = baseline.copy()
        final[gate] = routed[gate]

        y = [r["y"] for r in test]
        base_m = metrics(y, baseline)
        cand_m = metrics(y, final)

        if gate.any():
            high_base = metrics(y=[y[i] for i in np.flatnonzero(gate)], probs=baseline[gate])
            high_cand = metrics(y=[y[i] for i in np.flatnonzero(gate)], probs=final[gate])
        else:
            high_base = None
            high_cand = None

        blocks.append(
            {
                "n": len(test),
                "gate_n": int(gate.sum()),
                "gate_rate": float(gate.mean()),
                "uncertainty_threshold": threshold,
                "baseline": base_m,
                "candidate": cand_m,
                "delta": {
                    key: float(cand_m[key] - base_m[key])
                    for key in ("accuracy", "logloss", "brier", "ece")
                },
                "high_uncertainty": {
                    "baseline": high_base,
                    "candidate": high_cand,
                },
                "mean_weights": {
                    expert: float(weights[:, j].mean())
                    for j, expert in enumerate(EXPERTS)
                },
            }
        )

        # IMPORTANT: current outcomes update state only after prediction is fixed.
        _update_state(test_probs, test, state)

    if not blocks:
        return {"status": "DEFERRED", "reason": "no_valid_oos_blocks", "blocks": 0}

    baseline = _aggregate(blocks, "baseline")
    candidate = _aggregate(blocks, "candidate")
    delta = {
        key: float(candidate[key] - baseline[key])
        for key in ("accuracy", "logloss", "brier", "ece")
    }

    ll_delta = np.asarray([b["delta"]["logloss"] for b in blocks], dtype=float)
    br_delta = np.asarray([b["delta"]["brier"] for b in blocks], dtype=float)
    acc_delta = np.asarray([b["delta"]["accuracy"] for b in blocks], dtype=float)

    ll_rel = (baseline["logloss"] - candidate["logloss"]) / max(baseline["logloss"], EPS)
    br_rel = (baseline["brier"] - candidate["brier"]) / max(baseline["brier"], EPS)

    return {
        "status": "OK",
        "blocks": len(blocks),
        "samples": int(baseline["n"]),
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "relative_improvement": {
            "accuracy": delta["accuracy"] / max(baseline["accuracy"], EPS),
            "logloss": ll_rel,
            "brier": br_rel,
        },
        "stability": {
            "improved_logloss_ratio": float(np.mean(ll_delta < 0.0)),
            "improved_brier_ratio": float(np.mean(br_delta < 0.0)),
            "non_worse_accuracy_ratio": float(np.mean(acc_delta >= -0.005)),
        },
        "eligible": bool(
            len(blocks) >= 8
            and (ll_rel >= 0.03 or delta["accuracy"] >= 0.01)
            and br_rel >= 0.01
            and float(np.mean(acc_delta >= -0.005)) >= 0.70
        ),
        "final_prequential_state": {
            "ema": {
                expert: float(value)
                for expert, value in state["ema"].items()
            },
            "seen": int(state["seen"]),
        },
        "blocks_detail": blocks,
    }


def _evaluate_holdout(
    rows: list[dict[str, Any]],
    horizon: str,
    development_state: dict[str, Any] | None,
) -> dict[str, Any]:
    split = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    development = rows[:split]
    holdout = rows[split:]
    gap = int(PURGE_BARS[horizon] + EMBARGO_BARS[horizon])

    meta_end = len(development) - gap
    meta_start = max(MIN_TRAIN, meta_end - META_BLOCK)
    base_train = development[:meta_start]
    meta = development[meta_start:meta_end]

    if len(base_train) < MIN_TRAIN or len(meta) < MIN_META_ROWS or len(holdout) < 200:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_holdout_training",
        }

    meta_models = _fit(base_train)
    meta_probs = _probs(meta_models, meta, horizon)
    threshold = _uncertainty_gate(_uncertainty(meta_probs))
    reliability_models = _fit_reliability_models(meta_probs, meta)

    hold_models = _fit(development)
    hold_probs = _probs(hold_models, holdout, horizon)
    hold_reliability = _reliability_prob(hold_probs, holdout, reliability_models)

    state = _new_state()
    if development_state:
        state["ema"].update(
            {
                expert: float(value)
                for expert, value in development_state["ema"].items()
            }
        )
        state["seen"] = int(development_state["seen"])

    routed, weights = _route(hold_probs, hold_reliability, state)
    uncertainty = _uncertainty(hold_probs)
    gate = uncertainty >= threshold

    baseline = hold_probs["production"]
    final = baseline.copy()
    final[gate] = routed[gate]

    y = [r["y"] for r in holdout]
    base_m = metrics(y, baseline)
    cand_m = metrics(y, final)

    high_base = None
    high_cand = None
    if gate.any():
        gate_idx = np.flatnonzero(gate)
        y_gate = [y[i] for i in gate_idx]
        high_base = metrics(y_gate, baseline[gate])
        high_cand = metrics(y_gate, final[gate])

    return {
        "status": "OK",
        "n": len(holdout),
        "gate_n": int(gate.sum()),
        "gate_rate": float(gate.mean()),
        "uncertainty_threshold": threshold,
        "baseline": base_m,
        "candidate": cand_m,
        "delta": {
            key: float(cand_m[key] - base_m[key])
            for key in ("accuracy", "logloss", "brier", "ece")
        },
        "high_uncertainty": {
            "baseline": high_base,
            "candidate": high_cand,
        },
        "mean_weights": {
            expert: float(weights[:, j].mean())
            for j, expert in enumerate(EXPERTS)
        },
        "used_development_prequential_state": development_state is not None,
    }


def evaluate(horizon: str) -> dict[str, Any]:
    rows = load_archive_research_rows(horizon, MAX_ROWS)
    if len(rows) < MIN_TRAIN + META_BLOCK + TEST_BLOCK + 200:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_archive_rows",
            "n": len(rows),
        }

    dev_end = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    development = rows[:dev_end]
    dev = _evaluate_development(development, horizon)
    state = dev.get("final_prequential_state") if isinstance(dev, dict) else None
    hold = _evaluate_holdout(rows, horizon, state)

    return {
        "status": "OK",
        "schema_version": 2,
        "research_only": True,
        "production_changed": False,
        "final_holdout_protected": True,
        "final_holdout_used_for_selection": False,
        "strict_point_in_time_archive_replay": False,
        "horizon": horizon,
        "n": len(rows),
        "development_n": len(development),
        "final_holdout_n": len(rows) - dev_end,
        "champion_baseline": "models/{horizon}.joblib frozen artifact",
        "routing_policy": (
            "prequential expert reliability + prior settled-loss state; "
            "activate only above prior-meta 75th-percentile uncertainty; "
            "otherwise retain production Champion"
        ),
        "development": dev,
        "final_holdout": hold,
    }


def main() -> None:
    payload = {
        "schema_version": 2,
        "research_only": True,
        "production_changed": False,
        "horizons": {horizon: evaluate(horizon) for horizon in HORIZONS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
