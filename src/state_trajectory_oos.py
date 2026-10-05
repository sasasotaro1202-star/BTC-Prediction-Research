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
    timestamps = [item["_created_dt"] for item in normalized]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError("duplicate_trajectory_timestamp")
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


def _fit_state_space(train_rows: list[dict[str, Any]], n_clusters: int):
    if len(train_rows) < max(2, n_clusters):
        return None
    x = np.asarray([row["x"] for row in train_rows], dtype=float)
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    k = min(int(n_clusters), len(train_rows))
    if k < 2:
        return None
    clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusterer.fit(xs)
    return scaler, clusterer, k


def _fit_transition_model(train_pairs: list[dict[str, Any]], state_space):
    if state_space is None or len(train_pairs) < 2:
        return None
    scaler, clusterer, n_classes = state_space
    x = np.asarray([pair["x"] for pair in train_pairs], dtype=float)
    target_x = np.asarray([pair["target_x"] for pair in train_pairs], dtype=float)
    target_state = clusterer.predict(scaler.transform(target_x))
    model = RandomForestClassifier(
        n_estimators=240,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(scaler.transform(x), target_state)
    return scaler, clusterer, model, n_classes


def _predict(pairs: list[dict[str, Any]], fitted):
    scaler, clusterer, model, n_classes = fitted
    x = np.asarray([pair["x"] for pair in pairs], dtype=float)
    target_x = np.asarray([pair["target_x"] for pair in pairs], dtype=float)
    current_state = clusterer.predict(scaler.transform(x))
    future_state = clusterer.predict(scaler.transform(target_x))
    raw = np.asarray(model.predict_proba(scaler.transform(x)), dtype=float)
    probs = np.zeros((len(pairs), n_classes), dtype=float)
    for col, cls in enumerate(np.asarray(model.classes_, dtype=int)):
        probs[:, cls] = raw[:, col]
    probs = np.clip(probs, 1e-8, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    persistence = np.zeros_like(probs)
    persistence[np.arange(len(pairs)), current_state] = 1.0
    return future_state.astype(int), probs, persistence


def _metrics(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if len(y_true) == 0:
        raise ValueError("trajectory_empty_evaluation")
    probs = np.clip(probs, 1e-8, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    conf = probs.max(axis=1)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    one = np.zeros((len(y_true), n_classes), dtype=float)
    one[np.arange(len(y_true)), y_true] = 1.0
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "logloss": float(log_loss(y_true, probs, labels=list(range(n_classes)))),
        "brier": float(np.mean(np.sum((probs - one) ** 2, axis=1))),
        "ece": float(ece),
    }


def _evaluate_step(
    rows: list[dict[str, Any]],
    *,
    steps: int,
    n_clusters: int,
    min_train: int,
    test_block: int,
    min_oos: int,
) -> dict[str, Any]:
    pairs = _build_pairs(rows, steps)
    if len(pairs) < min_train + min_oos:
        return {"status": "DEFERRED", "reason": "insufficient_contiguous_pairs", "pair_n": len(pairs), "oos_n": 0, "final_holdout_n": 0}

    split = int(len(pairs) * (1.0 - FINAL_HOLDOUT_FRAC))
    development = pairs[:split]
    holdout = pairs[split:]
    if len(development) < min_train or not holdout:
        return {"status": "DEFERRED", "reason": "insufficient_development_holdout_split", "pair_n": len(pairs), "oos_n": 0, "final_holdout_n": len(holdout)}

    # Fit the latent-state vocabulary once using only the initial
    # chronological training window available before the first scored fold.
    # Keeping it frozen makes state IDs comparable across folds while preserving
    # the PIT boundary. Scored and frozen-holdout observations are excluded.
    initial_train_pairs = development[:max(0, min_train - steps)]
    initial_state_row_end = max(
        (int(pair["to_index"]) for pair in initial_train_pairs), default=-1
    ) + 1
    initial_state_rows = rows[:initial_state_row_end]
    state_space = _fit_state_space(initial_state_rows, n_clusters)
    if state_space is None:
        return {
            "status": "DEFERRED",
            "reason": "initial_state_space_fit_unavailable",
            "pair_n": len(pairs),
            "oos_n": 0,
            "final_holdout_n": len(holdout),
        }

    oos_y: list[int] = []
    oos_candidate: list[list[float]] = []
    oos_persistence: list[list[float]] = []
    fold_rows: list[dict[str, Any]] = []

    for test_start in range(min_train, len(development), test_block):
        test_end = min(test_start + test_block, len(development))
        train_pairs = development[: max(0, test_start - steps)]
        test_pairs = development[test_start:test_end]
        if len(train_pairs) < min_train or not test_pairs:
            continue

        fitted = _fit_transition_model(train_pairs, state_space)
        if fitted is None:
            continue

        y, candidate, persistence = _predict(test_pairs, fitted)
        oos_y.extend(y.tolist())
        oos_candidate.extend(candidate.tolist())
        oos_persistence.extend(persistence.tolist())
        fold_rows.append({
            "test_start": test_start,
            "test_end": test_end,
            "train_pair_n": len(train_pairs),
            "state_fit_row_n": len(initial_state_rows),
            "state_fit_latest_created": (
                initial_state_rows[-1]["created"].isoformat() if initial_state_rows else None
            ),
            "state_fit_scope": "initial_training_window_only",
            "test_pair_first_created": test_pairs[0]["created"],
            "purge_steps": steps,
        })

    if len(oos_y) < min_oos:
        return {"status": "DEFERRED", "reason": "insufficient_prequential_oos_rows", "pair_n": len(pairs), "oos_n": len(oos_y), "final_holdout_n": len(holdout), "folds": fold_rows}

    final_train_pairs = development[:-steps] if len(development) > steps else []
    final_fitted = _fit_transition_model(final_train_pairs, state_space)
    if final_fitted is None:
        return {"status": "DEFERRED", "reason": "final_training_fit_unavailable", "pair_n": len(pairs), "oos_n": len(oos_y), "final_holdout_n": len(holdout), "folds": fold_rows}

    hold_y, hold_candidate, hold_persistence = _predict(holdout, final_fitted)
    n_classes = int(final_fitted[-1])
    oos_metrics = {
        "candidate": _metrics(np.asarray(oos_y, dtype=int), np.asarray(oos_candidate), n_classes),
        "persistence": _metrics(np.asarray(oos_y, dtype=int), np.asarray(oos_persistence), n_classes),
    }
    hold_metrics = {
        "candidate": _metrics(hold_y, hold_candidate, n_classes),
        "persistence": _metrics(hold_y, hold_persistence, n_classes),
    }
    oos_delta = {k: oos_metrics["candidate"][k] - oos_metrics["persistence"][k] for k in ("accuracy", "logloss", "brier", "ece")}
    hold_delta = {k: hold_metrics["candidate"][k] - hold_metrics["persistence"][k] for k in ("accuracy", "logloss", "brier", "ece")}

    return {
        "status": "OK",
        "pair_n": len(pairs),
        "oos_n": len(oos_y),
        "final_holdout_n": len(holdout),
        "state_count": n_classes,
        "state_label_method": "initial-training-window-kmeans",
        "evaluation": {"purge_steps": steps, "test_block": test_block, "final_holdout_fraction": FINAL_HOLDOUT_FRAC},
        "oos": oos_metrics,
        "oos_delta_candidate_minus_persistence": oos_delta,
        "final_holdout": hold_metrics,
        "final_holdout_delta_candidate_minus_persistence": hold_delta,
        "folds": fold_rows,
        "future_rows_used_for_state_space_fit": False,
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
        return {"status": "DEFERRED", "reason": "no_pit_safe_rows", "research_only": True, "production_changed": False, "promotion_allowed": False, "horizons": {}}
    if any(step not in HORIZON_NAMES for step in steps):
        raise ValueError("unsupported_trajectory_step")
    if n_clusters < 2 or min_train < 2 or test_block < 1 or min_oos < 1:
        raise ValueError("invalid_trajectory_parameters")

    normalized = _normalize_rows(rows)
    if not normalized:
        return {"status": "DEFERRED", "reason": "no_valid_timestamped_feature_rows", "research_only": True, "production_changed": False, "promotion_allowed": False, "horizons": {}}

    horizons = {
        HORIZON_NAMES[step]: _evaluate_step(
            normalized,
            steps=step,
            n_clusters=n_clusters,
            min_train=min_train,
            test_block=test_block,
            min_oos=min_oos,
        )
        for step in steps
    }
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
        "state_definition": "initial-training-window-frozen",
        "future_error_control": "direct_each_horizon_with_fold_purge",
    }

def main():
    import json
    from pathlib import Path
    from model_compare import load_primary_production_strict_rows

    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "historical_research" / "state_trajectory_oos.json"
    rows = load_primary_production_strict_rows("5m")
    payload = build_trajectory_oos(rows)
    payload.update({
        "schema_version": 1,
        "input_scope": "strict_primary_5m_pit_rows",
        "source_row_count": len(rows),
        "promotion_effect": "none",
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload.get("status"),
        "source_row_count": len(rows),
        "horizons": {h: v.get("status") for h, v in payload.get("horizons", {}).items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()