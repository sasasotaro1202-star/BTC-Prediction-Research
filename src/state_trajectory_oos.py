"""PIT-safe research evaluator for direct multi-horizon BTC state trajectories.

This module predicts future latent market-state clusters directly from the
current feature vector. It is research-only: it never changes Production,
the model registry, live prediction state, or the frozen holdout.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

HORIZON_NAMES = {1: "5m", 2: "10m", 3: "15m", 6: "30m", 12: "60m"}
DEFAULT_STEPS = tuple(HORIZON_NAMES)
BASE_INTERVAL_MINUTES = 5
DEFAULT_N_CLUSTERS = 6
DEFAULT_MIN_TRAIN = 400
DEFAULT_TEST_BLOCK = 25
DEFAULT_MIN_OOS = 200
FINAL_HOLDOUT_FRAC = 0.20


def _parse_time(value: Any) -> datetime:
    text = str(value).replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise ValueError("trajectory_timestamp_must_be_timezone_aware")
    return dt


def _finite_vector(row: dict[str, Any]) -> np.ndarray | None:
    try:
        x = np.asarray(row["x"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        return None
    return x


def _normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            created = _parse_time(row["created"])
        except (KeyError, TypeError, ValueError):
            continue
        x = _finite_vector(row)
        if x is None:
            continue
        normalized.append({
            **row,
            "_created_dt": created,
            "_x": x,
        })
    normalized.sort(key=lambda item: (item["_created_dt"], str(item.get("id", ""))))
    return normalized


def _build_pairs(rows: list[dict[str, Any]], steps: int) -> list[dict[str, Any]]:
    """Build exact-interval pairs without bridging missing 5-minute observations."""
    if steps < 1:
        raise ValueError("steps_must_be_positive")
    normalized = _normalize_rows(rows)
    by_time = {
        item["_created_dt"]: idx
        for idx, item in enumerate(normalized)
    }
    pairs: list[dict[str, Any]] = []
    for start_idx, start in enumerate(normalized):
        valid = True
        target_idx = None
        for offset in range(1, steps + 1):
            target_time = start["_created_dt"] + timedelta(minutes=BASE_INTERVAL_MINUTES * offset)
            idx = by_time.get(target_time)
            if idx is None:
                valid = False
                break
            target_idx = idx
        if not valid or target_idx is None:
            continue
        target = normalized[target_idx]
        pairs.append({
            "from_index": start_idx,
            "to_index": target_idx,
            "created": start["_created_dt"].isoformat(),
            "target_created": target["_created_dt"].isoformat(),
            "elapsed_minutes": int(
                (target["_created_dt"] - start["_created_dt"]).total_seconds() / 60
            ),
            "x": start["_x"].tolist(),
            "target_x": target["_x"].tolist(),
        })
    return pairs


def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(labels), n_classes), dtype=float)
    if len(labels):
        out[np.arange(len(labels)), labels.astype(int)] = 1.0
    return out


def _metrics(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> dict[str, float]:
    if len(y_true) == 0:
        raise ValueError("trajectory_empty_evaluation")
    probs = np.clip(np.asarray(probs, dtype=float), 1e-8, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = (prediction == y_true).astype(float)
    ece = 0.0
    for lo, hi in zip(np.linspace(0.0, 1.0, 11)[:-1], np.linspace(0.0, 1.0, 11)[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1.0 else confidence <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    truth = _one_hot(y_true, n_classes)
    brier = float(np.mean(np.sum((probs - truth) ** 2, axis=1)))
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "logloss": float(log_loss(y_true, probs, labels=list(range(n_classes)))),
        "brier": brier,
        "ece": ece,
    }


def _fit_state_space(rows: list[dict[str, Any]], n_clusters: int, fit_rows: int):
    if len(rows) < fit_rows or fit_rows < 2:
        return None
    reference = np.asarray([row["_x"] for row in rows[:fit_rows]], dtype=float)
    actual_clusters = min(int(n_clusters), len(reference))
    if actual_clusters < 2:
        return None
    scaler = StandardScaler()
    reference_scaled = scaler.fit_transform(reference)
    clusterer = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    clusterer.fit(reference_scaled)
    return scaler, clusterer, actual_clusters


def _fit_fold(train_pairs: list[dict[str, Any]], state_space):
    if len(train_pairs) < 2 or state_space is None:
        return None
    scaler, clusterer, n_classes = state_space
    x_train = np.asarray([p["x"] for p in train_pairs], dtype=float)
    target_x_train = np.asarray([p["target_x"] for p in train_pairs], dtype=float)
    target_state = clusterer.predict(scaler.transform(target_x_train))
    model = RandomForestClassifier(
        n_estimators=240,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(scaler.transform(x_train), target_state)
    return scaler, clusterer, model, n_classes


def _predict(fitted, pairs: list[dict[str, Any]]):
    scaler, clusterer, model, n_classes = fitted
    x = np.asarray([p["x"] for p in pairs], dtype=float)
    target_x = np.asarray([p["target_x"] for p in pairs], dtype=float)
    current_state = clusterer.predict(scaler.transform(x))
    future_state = clusterer.predict(scaler.transform(target_x))
    raw = model.predict_proba(scaler.transform(x))
    probs = np.zeros((len(pairs), n_classes), dtype=float)
    for col, cls in enumerate(model.classes_.astype(int)):
        probs[:, cls] = raw[:, col]
    persistence = _one_hot(current_state, n_classes)
    return future_state.astype(int), probs, persistence


def _evaluate_step(
    rows: list[dict[str, Any]],
    *,
    steps: int,
    state_space,
    test_block: int,
    min_train: int,
    min_oos: int,
) -> dict[str, Any]:
    pairs = _build_pairs(rows, steps)
    if len(pairs) < min_train + min_oos:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_contiguous_pairs",
            "pair_n": len(pairs),
            "oos_n": 0,
            "final_holdout_n": 0,
        }

    split = int(len(pairs) * (1.0 - FINAL_HOLDOUT_FRAC))
    development = pairs[:split]
    holdout = pairs[split:]

    oos_y: list[int] = []
    oos_candidate: list[list[float]] = []
    oos_persistence: list[list[float]] = []

    for test_start in range(min_train, len(development), test_block):
        train_end = max(0, test_start - steps)
        train_pairs = development[:train_end]
        test_pairs = development[test_start:min(test_start + test_block, len(development))]
        if len(train_pairs) < min_train or not test_pairs:
            continue
        fitted = _fit_fold(train_pairs, state_space)
        if fitted is None:
            continue
        y, candidate, persistence = _predict(fitted, test_pairs)
        oos_y.extend(y.tolist())
        oos_candidate.extend(candidate.tolist())
        oos_persistence.extend(persistence.tolist())

    if len(oos_y) < min_oos:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_prequential_oos_rows",
            "pair_n": len(pairs),
            "oos_n": len(oos_y),
            "final_holdout_n": len(holdout),
        }

    final_train = development[:-steps] if len(development) > steps else []
    fitted_final = _fit_fold(final_train, state_space)
    if fitted_final is None:
        return {
            "status": "DEFERRED",
            "reason": "final_training_fit_unavailable",
            "pair_n": len(pairs),
            "oos_n": len(oos_y),
            "final_holdout_n": len(holdout),
        }

    hold_y, hold_candidate, hold_persistence = _predict(fitted_final, holdout)

    n_classes = int(fitted_final[-1])
    oos_metrics = {
        "candidate": _metrics(np.asarray(oos_y, dtype=int), np.asarray(oos_candidate), n_classes),
        "persistence": _metrics(np.asarray(oos_y, dtype=int), np.asarray(oos_persistence), n_classes),
    }
    hold_metrics = {
        "candidate": _metrics(hold_y, hold_candidate, n_classes),
        "persistence": _metrics(hold_y, hold_persistence, n_classes),
    }

    return {
        "status": "OK",
        "pair_n": len(pairs),
        "oos_n": len(oos_y),
        "final_holdout_n": len(holdout),
        "state_count": n_classes,
        "state_label_method": "train-fold-kmeans",
        "evaluation": {
            "purge_steps": steps,
            "test_block": test_block,
            "final_holdout_fraction": FINAL_HOLDOUT_FRAC,
        },
        "oos": oos_metrics,
        "oos_delta_candidate_minus_persistence": {
            key: oos_metrics["candidate"][key] - oos_metrics["persistence"][key]
            for key in ("accuracy", "logloss", "brier", "ece")
        },
        "final_holdout": hold_metrics,
        "selection_uses_future_test_outcomes": False,
        "final_holdout_used_for_selection": False,
    }


def build_trajectory_oos(
    rows: list[dict[str, Any]],
    *,
    steps: tuple[int, ...] = DEFAULT_STEPS,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    min_train: int = DEFAULT_MIN_TRAIN,
    test_block: int = DEFAULT_TEST_BLOCK,
    min_oos: int = DEFAULT_MIN_OOS,
) -> dict[str, Any]:
    if not rows:
        return {
            "status": "DEFERRED",
            "reason": "no_pit_safe_rows",
            "research_only": True,
            "production_changed": False,
            "evaluation_mode": "direct_multi_horizon",
            "horizons": {},
        }
    if any(step not in HORIZON_NAMES for step in steps):
        raise ValueError("unsupported_trajectory_step")
    if n_clusters < 2 or min_train < 2 or test_block < 1 or min_oos < 1:
        raise ValueError("invalid_trajectory_parameters")

    normalized = _normalize_rows(rows)
    if not normalized:
        return {
            "status": "DEFERRED",
            "reason": "no_valid_timestamped_feature_rows",
            "research_only": True,
            "production_changed": False,
            "evaluation_mode": "direct_multi_horizon",
            "horizons": {},
        }

    state_space = _fit_state_space(normalized, n_clusters, min_train)
    if state_space is None:
        return {
            "status": "DEFERRED",
            "reason": "state_vocabulary_fit_unavailable",
            "research_only": True,
            "production_changed": False,
            "evaluation_mode": "direct_multi_horizon",
            "horizons": {},
        }

    horizons = {}
    for step in steps:
        horizons[HORIZON_NAMES[step]] = _evaluate_step(
            normalized,
            steps=step,
            state_space=state_space,
            min_train=min_train,
            test_block=test_block,
            min_oos=min_oos,
        )

    ok_count = sum(item.get("status") == "OK" for item in horizons.values())
    return {
        "status": "OK" if ok_count else "DEFERRED",
        "reason": "" if ok_count else "insufficient_contiguous_live_history",
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "evaluation_mode": "direct_multi_horizon",
        "input_interval_minutes": BASE_INTERVAL_MINUTES,
        "horizons": horizons,
        "state_representation": "unsupervised_feature-state_clusters",
        "state_definition": "frozen_initial_training_window",
        "state_vocabulary_fit_rows": min_train,
        "future_error_control": "direct_each_horizon_with_fold_purge",
    }
