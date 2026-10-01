"""Research-only prequential expert mixture for prediction-error risk.

The candidate risk estimators (global, case_memory, meta) are combined instead
of winner-take-all routing. Expert weights are updated from matured historical
losses at fixed refresh boundaries, then frozen for the next prediction block.
This borrows the online/local-performance idea from adaptive dynamic model
selection while preserving the project's PIT and research-only boundaries.
"""
from __future__ import annotations


try:
    from experience_pit_scope import load_strict_primary_rows
except ModuleNotFoundError:
    from src.experience_pit_scope import load_strict_primary_rows
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer

from experience_case_adaptive_controller_oos import (
    CLASSES,
    _adjust_probabilities,
    _meta_features,
    _parse_ts,
    _probabilities,
    _safe01,
)
from experience_predictability_router_oos import (
    _baseline_error,
    _case_key,
    _memory_add,
    _memory_risk,
    _memory_state,
    _fit_meta_model,
    _predict_meta,
)

try:
    from db import DB, init_db
except ModuleNotFoundError:
    from src.db import DB, init_db

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "experience_failure_expert_mix_oos.json"

CANDIDATES = ("global", "case_memory", "meta", "temporal_memory")
MIN_TRAIN = 140
VALIDATION_SIZE = 60
MIX_REFRESH = 20
ETA = 1.0
WEIGHT_FLOOR = 0.05
TEMPORAL_K = 25
MIN_TEMPORAL_TRAIN = 60
BLOCK_SIZE = 40


def _eligible_prior(rows: list[Any], current: Any) -> list[Any]:
    prediction_time = _parse_ts(current["created_at_utc"])
    out: list[Any] = []
    for row in rows:
        try:
            created = _parse_ts(row["created_at_utc"])
            settled = _parse_ts(row["settled_at_utc"])
        except (TypeError, ValueError):
            continue
        if created < prediction_time and settled < prediction_time:
            out.append(row)
    return out


def _binary_logloss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(labels) == 0:
        return float("nan")
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0 - 1e-9)
    return float(-np.mean(np.where(labels == 1, np.log(probabilities), np.log(1.0 - probabilities))))


def _sequential_validation_risks(
    train_rows: list[Any],
    validation_rows: list[Any],
) -> dict[str, np.ndarray]:
    ordered_rows = sorted(
        list(train_rows) + list(validation_rows),
        key=lambda row: (
            _parse_ts(row["settled_at_utc"]),
            _parse_ts(row["created_at_utc"]),
            int(row["experience_id"]),
        ),
    )
    train_ids = {int(row["experience_id"]) for row in train_rows}
    validation_sorted = sorted(
        validation_rows,
        key=lambda x: (_parse_ts(x["created_at_utc"]), int(x["experience_id"])),
    )
    memory_state = _memory_state([])
    added_ids: set[int] = set()
    global_values: list[float] = []
    memory_values: list[float] = []

    for row in validation_sorted:
        prediction_time = _parse_ts(row["created_at_utc"])
        for candidate in ordered_rows:
            cid = int(candidate["experience_id"])
            if cid in added_ids or cid == int(row["experience_id"]):
                continue
            created = _parse_ts(candidate["created_at_utc"])
            settled = _parse_ts(candidate["settled_at_utc"])
            if created < prediction_time and settled < prediction_time:
                _memory_add(memory_state, candidate)
                added_ids.add(cid)
        global_values.append(
            _safe01(
                (memory_state["errors"] + 1.0)
                / (memory_state["total"] + 2.0)
            )
        )
        memory_values.append(_memory_risk(memory_state, row))

    ordered_validation = sorted(
        validation_rows,
        key=lambda x: (_parse_ts(x["created_at_utc"]), int(x["experience_id"])),
    )
    order_index = {
        int(row["experience_id"]): i for i, row in enumerate(ordered_validation)
    }
    return {
        "global": np.asarray(
            [global_values[order_index[int(row["experience_id"])] ] for row in validation_rows],
            dtype=float,
        ),
        "case_memory": np.asarray(
            [memory_values[order_index[int(row["experience_id"])] ] for row in validation_rows],
            dtype=float,
        ),
    }



def _temporal_features(rows: list[Any]) -> list[dict[str, float | str]]:
    features: list[dict[str, float | str]] = []
    for row in rows:
        item = dict(_meta_features(row))
        probability = _probabilities(row)
        item.update(
            {
                "p_down": float(probability[0]),
                "p_flat": float(probability[1]),
                "p_up": float(probability[2]),
            }
        )
        features.append(item)
    return features


def _fit_temporal_memory(train_rows: list[Any]) -> Any:
    if len(train_rows) < MIN_TEMPORAL_TRAIN:
        return None
    try:
        vectorizer = DictVectorizer(sparse=False)
        matrix = np.asarray(
            vectorizer.fit_transform(_temporal_features(train_rows)),
            dtype=float,
        )
        if matrix.ndim != 2 or len(matrix) < MIN_TEMPORAL_TRAIN:
            return None
        median = np.nanmedian(matrix, axis=0)
        scale = np.nanmedian(np.abs(matrix - median), axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
        normalized = np.nan_to_num(
            (matrix - median) / scale,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        labels = np.asarray(
            [1 - int(row["correct"]) for row in train_rows],
            dtype=float,
        )
        return vectorizer, normalized, labels
    except (TypeError, ValueError, FloatingPointError):
        return None


def _predict_temporal_memory(
    bundle: Any,
    rows: list[Any],
    k: int = TEMPORAL_K,
) -> np.ndarray:
    if bundle is None or not rows:
        return np.full(len(rows), 0.5, dtype=float)
    try:
        vectorizer, train_matrix, labels = bundle
        current = np.asarray(
            vectorizer.transform(_temporal_features(rows)),
            dtype=float,
        )
        current = np.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0)
        if train_matrix.ndim != 2 or len(train_matrix) == 0:
            return np.full(len(rows), 0.5, dtype=float)
        k = max(3, min(int(k), len(train_matrix)))
        output: list[float] = []
        n_train = len(train_matrix)
        for query in current:
            distances = np.sqrt(
                np.mean((train_matrix - query[None, :]) ** 2, axis=1)
            )
            idx = np.argpartition(distances, k - 1)[:k]
            d = distances[idx]
            temperature = max(float(np.median(d)) + 0.25, 0.25)
            similarity = np.exp(-d / temperature)
            ages = (n_train - 1) - idx
            recency = np.exp(-ages / 120.0)
            weights = similarity * recency
            if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
                weights = np.ones_like(weights)
            weights /= float(weights.sum())
            weighted_error = float(np.dot(weights, labels[idx]))
            output.append((weighted_error * len(idx) + 1.0) / (len(idx) + 2.0))
        return np.asarray([_safe01(v) for v in output], dtype=float)
    except (TypeError, ValueError, FloatingPointError):
        return np.full(len(rows), 0.5, dtype=float)



def _validation_candidate_losses(
    matured: list[Any],
    current: Any,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    eligible = _eligible_prior(matured, current)
    if len(eligible) < MIN_TRAIN + VALIDATION_SIZE:
        return (
            {name: float("nan") for name in CANDIDATES},
            {name: np.asarray([], dtype=float) for name in CANDIDATES},
        )

    validation = eligible[-VALIDATION_SIZE:]
    validation_start = _parse_ts(validation[0]["created_at_utc"])
    train = [
        row
        for row in eligible[:-VALIDATION_SIZE]
        if _parse_ts(row["created_at_utc"]) < validation_start
        and _parse_ts(row["settled_at_utc"]) < validation_start
    ]
    if len(train) < MIN_TRAIN:
        return (
            {name: float("nan") for name in CANDIDATES},
            {name: np.asarray([], dtype=float) for name in CANDIDATES},
        )

    labels = np.asarray(
        [1 - int(row["correct"]) for row in validation],
        dtype=int,
    )
    candidate_values = _sequential_validation_risks(train, validation)
    meta_bundle, meta_default = _fit_meta_model(train)
    candidate_values["meta"] = _predict_meta(meta_bundle, validation, meta_default)
    temporal_bundle = _fit_temporal_memory(train)
    candidate_values["temporal_memory"] = _predict_temporal_memory(
        temporal_bundle,
        validation,
    )
    if temporal_bundle is None:
        candidate_values["temporal_memory"][:] = _baseline_error(train)
    losses = {
        name: _binary_logloss(labels, candidate_values[name])
        for name in CANDIDATES
    }
    return losses, candidate_values


def _update_weights(
    weights: dict[str, float],
    losses: dict[str, float],
) -> dict[str, float]:
    updated = {
        name: float(weights.get(name, 1.0 / len(CANDIDATES)))
        for name in CANDIDATES
    }
    total_initial = sum(updated.values())
    if total_initial <= 0.0 or not math.isfinite(total_initial):
        updated = {name: 1.0 for name in CANDIDATES}
    finite_losses = {
        name: float(value)
        for name, value in losses.items()
        if name in CANDIDATES and math.isfinite(float(value))
    }
    if not finite_losses:
        return dict(weights)

    for name in CANDIDATES:
        if name not in finite_losses:
            continue
        updated[name] *= math.exp(-ETA * min(10.0, max(0.0, finite_losses[name])))
    total = sum(updated.values())
    if total <= 0.0 or not math.isfinite(total):
        updated = {name: 1.0 for name in CANDIDATES}
        total = 3.0
    normalized = {name: updated[name] / total for name in CANDIDATES}

    # Keep every expert alive so a temporarily weak expert can recover.
    floor = WEIGHT_FLOOR / len(CANDIDATES)
    residual = {
        name: max(0.0, normalized[name] - floor)
        for name in CANDIDATES
    }
    residual_total = sum(residual.values())
    remaining = max(0.0, 1.0 - floor * len(CANDIDATES))
    if residual_total <= 0.0 or not math.isfinite(residual_total):
        return {name: 1.0 / len(CANDIDATES) for name in CANDIDATES}
    return {
        name: floor + remaining * residual[name] / residual_total
        for name in CANDIDATES
    }


def _mixed_risk(
    candidate_risks: dict[str, float],
    weights: dict[str, float],
) -> float:
    return _safe01(
        sum(
            float(weights.get(name, 0.0)) * _safe01(float(candidate_risks[name]))
            for name in CANDIDATES
        )
    )


def _apply_risk(
    base: np.ndarray,
    risk: float,
    baseline_error: float,
) -> tuple[np.ndarray, str]:
    adjusted, action, _ = _adjust_probabilities(base, risk, baseline_error)
    return adjusted, action



def _risk_coverage(error_labels: np.ndarray, risk: np.ndarray, coverages=(0.90, 0.80, 0.70, 0.60, 0.50)) -> dict[str, dict[str, float | int]]:
    error_labels = np.asarray(error_labels, dtype=int)
    risk = np.asarray(risk, dtype=float)
    if len(error_labels) == 0:
        return {}
    order = np.argsort(risk, kind="mergesort")
    out: dict[str, dict[str, float | int]] = {}
    for coverage in coverages:
        k = max(1, int(round(len(error_labels) * float(coverage))))
        kept = error_labels[order[:k]]
        out[f"{float(coverage):.2f}"] = {
            "coverage": float(k / len(error_labels)),
            "error_rate": float(np.mean(kept)),
            "accuracy": float(1.0 - np.mean(kept)),
            "n": int(k),
        }
    return out


def _aurc(error_labels: np.ndarray, risk: np.ndarray) -> float:
    error_labels = np.asarray(error_labels, dtype=int)
    risk = np.asarray(risk, dtype=float)
    order = np.argsort(risk, kind="mergesort")
    kept = error_labels[order]
    cumulative = np.cumsum(kept) / np.arange(1, len(kept) + 1)
    coverage = np.arange(1, len(kept) + 1) / len(kept)
    if len(cumulative) <= 1:
        return float(cumulative[0]) if len(cumulative) else float("nan")
    return float(
        np.sum((cumulative[:-1] + cumulative[1:]) * np.diff(coverage) * 0.5)
    )



def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    pred = np.argmax(probabilities, axis=1)
    one = np.eye(3, dtype=float)[labels]
    return {
        "accuracy": float(np.mean(pred == labels)),
        "logloss": float(-np.mean(np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-9, 1.0)))),
        "brier": float(np.mean(np.sum((probabilities - one) ** 2, axis=1))),
    }


def evaluate_horizon(rows: list[Any], horizon: str) -> dict[str, Any]:
    ordered = sorted(
        [row for row in rows if str(row["horizon"]) == horizon],
        key=lambda row: (
            _parse_ts(row["created_at_utc"]),
            _parse_ts(row["settled_at_utc"]),
            int(row["experience_id"]),
        ),
    )
    if len(ordered) <= MIN_TRAIN + VALIDATION_SIZE:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_experience_rows",
            "n": len(ordered),
            "research_only": True,
            "production_changed": False,
            "promotion_evidence_eligible": False,
        }

    weights = {name: 1.0 / len(CANDIDATES) for name in CANDIDATES}
    labels: list[int] = []
    mixed_risks: list[float] = []
    base_probs: list[np.ndarray] = []
    global_probs: list[np.ndarray] = []
    mix_probs: list[np.ndarray] = []
    actions: list[str] = []
    weight_trace: list[dict[str, Any]] = []
    validation_trace: list[dict[str, Any]] = []
    pit_excluded = 0
    refresh_count = 0
    deferred = 0

    for block_start in range(MIN_TRAIN + VALIDATION_SIZE, len(ordered), MIX_REFRESH):
        block_end = min(len(ordered), block_start + MIX_REFRESH)
        first_current = ordered[block_start]
        matured = _eligible_prior(ordered[:block_start], first_current)
        pit_excluded += max(0, block_start - len(matured))
        if len(matured) < MIN_TRAIN + VALIDATION_SIZE:
            deferred += block_end - block_start
            continue

        losses, _ = _validation_candidate_losses(matured, first_current)
        before = dict(weights)
        weights = _update_weights(weights, losses)
        refresh_count += 1
        weight_trace.append({
            "block_start": int(block_start),
            "weights_before": before,
            "weights_after": dict(weights),
            "validation_logloss": losses,
        })
        validation_trace.append({
            "block_start": int(block_start),
            "validation_logloss": losses,
        })

        memory_state = _memory_state(matured)
        seen_ids = {int(row["experience_id"]) for row in matured}
        meta_bundle, meta_default = _fit_meta_model(matured)
        temporal_bundle = _fit_temporal_memory(matured)
        block_rows = ordered[block_start:block_end]
        frozen_meta = _predict_meta(meta_bundle, block_rows, meta_default)
        temporal_default = _baseline_error(matured)
        frozen_temporal = _predict_temporal_memory(
            temporal_bundle,
            block_rows,
        )
        if temporal_bundle is None:
            frozen_temporal[:] = temporal_default
        frozen_index = {int(row["experience_id"]): i for i, row in enumerate(block_rows)}

        for index in range(block_start, block_end):
            current = ordered[index]
            current_matured = _eligible_prior(ordered[:index], current)
            new_rows = [
                row for row in current_matured
                if int(row["experience_id"]) not in seen_ids
            ]
            for row in sorted(
                new_rows,
                key=lambda x: (_parse_ts(x["settled_at_utc"]), int(x["experience_id"])),
            ):
                _memory_add(memory_state, row)
                seen_ids.add(int(row["experience_id"]))

            baseline = _safe01(
                (memory_state["errors"] + 1.0)
                / (memory_state["total"] + 2.0)
            )
            candidate_risks = {
                "global": baseline,
                "case_memory": _memory_risk(memory_state, current),
                "meta": float(
                    frozen_meta[frozen_index[int(current["experience_id"])]]
                ),
                "temporal_memory": float(
                    frozen_temporal[frozen_index[int(current["experience_id"])]]
                ),
            }
            mixed = _mixed_risk(candidate_risks, weights)
            base = _probabilities(current)
            mixed_probability, action = _apply_risk(base, mixed, baseline)

            actual = str(current["actual_direction"])
            if actual not in CLASSES:
                continue
            labels.append(CLASSES.index(actual))
            base_probs.append(base)
            global_probs.append(
                _apply_risk(base, baseline, baseline)[0]
            )
            mix_probs.append(mixed_probability)
            mixed_risks.append(mixed)
            actions.append(action)

    if not labels:
        return {
            "status": "DEFERRED",
            "reason": "no_pit_valid_prequential_cases",
            "n": len(ordered),
            "research_only": True,
            "production_changed": False,
            "promotion_evidence_eligible": False,
        }

    y = np.asarray(labels, dtype=int)
    base = np.vstack(base_probs)
    global_adjusted = np.vstack(global_probs)
    mix = np.vstack(mix_probs)
    overall_base = _metrics(y, base)
    overall_global = _metrics(y, global_adjusted)
    overall_mix = _metrics(y, mix)
    mixed_risk = np.asarray(mixed_risks, dtype=float)
    base_error_labels = (np.argmax(base, axis=1) != y).astype(int)
    risk_cov = _risk_coverage(base_error_labels, mixed_risk)
    aurc_value = _aurc(base_error_labels, mixed_risk)
    baseline_risk = 1.0 - np.max(base, axis=1)
    baseline_rc = _risk_coverage(base_error_labels, baseline_risk)
    baseline_aurc = _aurc(base_error_labels, baseline_risk)

    blocks: list[dict[str, Any]] = []
    for start in range(0, len(y), BLOCK_SIZE):
        end = min(len(y), start + BLOCK_SIZE)
        if end - start < 20:
            continue
        base_m = _metrics(y[start:end], base[start:end])
        global_m = _metrics(y[start:end], global_adjusted[start:end])
        mix_m = _metrics(y[start:end], mix[start:end])
        blocks.append({
            "index": len(blocks),
            "start": int(start),
            "end": int(end),
            "n": int(end - start),
            "base": base_m,
            "global_adjusted": global_m,
            "expert_mix": mix_m,
            "delta_mix_minus_global": {
                "accuracy": mix_m["accuracy"] - global_m["accuracy"],
                "logloss": mix_m["logloss"] - global_m["logloss"],
                "brier": mix_m["brier"] - global_m["brier"],
            },
        })

    return {
        "status": "OK",
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "horizon": horizon,
        "n": len(ordered),
        "prequential_test_rows": len(y),
        "pit_violation_count": 0,
        "pit_excluded_candidate_count": int(pit_excluded),
        "deferred_cases": int(deferred),
        "mix_refresh_count": int(refresh_count),
        "mix_refresh_size": MIX_REFRESH,
        "eta": ETA,
        "weight_floor": WEIGHT_FLOOR,
            "temporal_k": TEMPORAL_K,
            "min_temporal_train": MIN_TEMPORAL_TRAIN,
        "candidate_sources": CANDIDATES,
        "weight_trace": weight_trace,
        "validation_trace": validation_trace,
        "base": overall_base,
        "global_adjusted": overall_global,
        "expert_mix": overall_mix,
        "delta_mix_minus_global": {
            "accuracy": overall_mix["accuracy"] - overall_global["accuracy"],
            "logloss": overall_mix["logloss"] - overall_global["logloss"],
            "brier": overall_mix["brier"] - overall_global["brier"],
        },
        "risk_selective_diagnostics": {
            "risk_target": "base_prediction_error",
            "risk_coverage": risk_cov,
            "aurc": aurc_value,
            "baseline_confidence_risk_coverage": baseline_rc,
            "baseline_confidence_aurc": baseline_aurc,
            "aurc_improvement": float(baseline_aurc - aurc_value) if np.isfinite(aurc_value) and np.isfinite(baseline_aurc) else None,
        },
        "action_counts": {
            name: int(actions.count(name))
            for name in sorted(set(actions))
        },
    }


def load_rows_with_pit_scope() -> tuple[list[Any], dict[str, Any]]:
    return load_strict_primary_rows(DB)


def load_rows() -> list[Any]:
    rows, _ = load_rows_with_pit_scope()
    return rows


def build() -> dict[str, Any]:
    rows, pit_scope = load_rows_with_pit_scope()
    result = {
        horizon: evaluate_horizon(rows, horizon)
        for horizon in ("5m", "10m")
    }
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_changed": False,
        "strict_pit_scope": True,
        "pit_scope": pit_scope,
        "promotion_evidence_eligible": False,
        "description": (
            "Prequential expert mixture: exponentially updates weights of "
            "global, case-memory and meta predictability estimators using only "
            "matured historical losses."
        ),
        "config": {
            "candidates": CANDIDATES,
            "min_train": MIN_TRAIN,
            "validation_size": VALIDATION_SIZE,
            "mix_refresh": MIX_REFRESH,
            "eta": ETA,
            "weight_floor": WEIGHT_FLOOR,
        },
        "horizons": result,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False))
