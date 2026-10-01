"""Research-only prequential router for prediction-error risk estimation.

The router treats the choice of predictability estimator as a first-class policy.
At each refresh boundary it compares three causal estimators on an earlier
validation slice of matured experience:
  global      - global historical error rate;
  case_memory - hierarchical case-conditional error memory;
  meta        - logistic prediction-error model.

The selected estimator is kept for ROUTER_REFRESH prediction cases before the
next source-selection refresh. This explicit model lifetime prevents expensive
retraining on every single row while preserving the causal boundary.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from experience_case_adaptive_controller_oos import (
    CLASSES,
    MIN_CASE_SUPPORT,
    MIN_TRAIN as CASE_MIN_TRAIN,
    _adjust_probabilities,
    _baseline_error,
    _case_key,
    _meta_features,
    _parse_ts,
    _probabilities,
    _safe01,
)

try:
    from db import DB, init_db
except ModuleNotFoundError:
    from src.db import DB, init_db

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "experience_predictability_router_oos.json"

MIN_TRAIN = 140
VALIDATION_SIZE = 60
TIE_EPS = 0.002
MIN_RELATIVE_GAIN = 0.01
ROUTER_REFRESH = 20
MIN_CONSECUTIVE_SELECTIONS = 2
MAX_SHRINK = 0.35
ABSTAIN_THRESHOLD = 0.80
CANDIDATES = ("global", "case_memory", "meta")


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


def _binary_logloss(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0 - 1e-9)
    y = np.asarray(y, dtype=int)
    return float(-np.mean(np.where(y == 1, np.log(p), np.log(1.0 - p))))


def _new_memory_state() -> dict[str, Any]:
    return {
        "total": 0,
        "errors": 0,
        "levels": [dict() for _ in range(6)],
    }


def _memory_add(state: dict[str, Any], row: Any) -> None:
    key = _case_key(row)
    state["total"] += 1
    state["errors"] += 1 - int(row["correct"])
    for length in range(1, 7):
        prefix = key[:length]
        n, errors = state["levels"][length - 1].get(prefix, (0, 0))
        state["levels"][length - 1][prefix] = (n + 1, errors + 1 - int(row["correct"]))


def _memory_risk(state: dict[str, Any], row: Any, shrinkage: float = 20.0) -> float:
    total = int(state["total"])
    errors = int(state["errors"])
    global_error = _safe01((errors + 1.0) / (total + 2.0))
    target = _case_key(row)
    for length in (6, 5, 4, 3, 2, 1):
        group = state["levels"][length - 1].get(target[:length])
        if group is None:
            continue
        n, group_errors = group
        rate = (group_errors + 1.0) / (n + 2.0)
        weight = n / (n + float(shrinkage))
        estimate = weight * rate + (1.0 - weight) * global_error
        if n >= MIN_CASE_SUPPORT or length == 1:
            return _safe01(estimate)
    return global_error


def _memory_state(rows: list[Any]) -> dict[str, Any]:
    state = _new_memory_state()
    for row in rows:
        _memory_add(state, row)
    return state


def _fit_meta_model(train_rows: list[Any]) -> tuple[tuple[DictVectorizer, LogisticRegression] | None, float]:
    baseline = _baseline_error(train_rows)
    if len(train_rows) < CASE_MIN_TRAIN:
        return None, baseline
    try:
        x_train = [dict(_meta_features(row)) for row in train_rows]
        y = np.asarray([1 - int(row["correct"]) for row in train_rows], dtype=int)
        if len(np.unique(y)) < 2:
            return None, baseline
        vectorizer = DictVectorizer(sparse=True)
        x = vectorizer.fit_transform(x_train)
        model = LogisticRegression(
            C=0.5,
            class_weight=None,
            max_iter=1000,
            random_state=42,
        )
        model.fit(x, y)
        if 1 not in {int(c) for c in model.classes_}:
            return None, baseline
        return (vectorizer, model), baseline
    except (TypeError, ValueError, FloatingPointError):
        return None, baseline


def _predict_meta(model_bundle: tuple[DictVectorizer, LogisticRegression] | None, rows: list[Any], default: float) -> np.ndarray:
    if not rows or model_bundle is None:
        return np.full(len(rows), default, dtype=float)
    vectorizer, model = model_bundle
    try:
        x = vectorizer.transform([dict(_meta_features(row)) for row in rows])
        class_index = {int(c): i for i, c in enumerate(model.classes_)}
        if 1 not in class_index:
            return np.full(len(rows), default, dtype=float)
        return np.asarray(model.predict_proba(x)[:, class_index[1]], dtype=float)
    except (TypeError, ValueError, FloatingPointError):
        return np.full(len(rows), default, dtype=float)


def _meta_predictions(train_rows: list[Any], prediction_rows: list[Any]) -> np.ndarray:
    bundle, default = _fit_meta_model(train_rows)
    return _predict_meta(bundle, prediction_rows, default)


def _risk(candidate: str, train_rows: list[Any], row: Any) -> float:
    if candidate == "global":
        return _baseline_error(train_rows)
    if candidate == "case_memory":
        return _memory_risk(_memory_state(train_rows), row)
    if candidate == "meta":
        return float(_meta_predictions(train_rows, [row])[0])
    raise ValueError(f"unknown_candidate:{candidate}")


def _choose_source(matured: list[Any], current: Any) -> tuple[str, dict[str, float], int]:
    eligible = _eligible_prior(matured, current)
    if len(eligible) < MIN_TRAIN + VALIDATION_SIZE:
        return "global", {name: float("nan") for name in CANDIDATES}, 0

    validation = eligible[-VALIDATION_SIZE:]
    validation_start = _parse_ts(validation[0]["created_at_utc"])
    train = [
        row for row in eligible[:-VALIDATION_SIZE]
        if _parse_ts(row["created_at_utc"]) < validation_start
        and _parse_ts(row["settled_at_utc"]) < validation_start
    ]
    if len(train) < MIN_TRAIN:
        return "global", {name: float("nan") for name in CANDIDATES}, 0

    labels = np.asarray([1 - int(row["correct"]) for row in validation], dtype=int)
    meta_bundle, meta_default = _fit_meta_model(train)
    meta_values = _predict_meta(meta_bundle, validation, meta_default)

    memory_state = _new_memory_state()
    settled_train = sorted(train, key=lambda row: (_parse_ts(row["settled_at_utc"]), int(row["experience_id"])))
    ptr = 0
    memory_values: list[float] = []
    global_values: list[float] = []
    for row in sorted(validation, key=lambda x: (_parse_ts(x["created_at_utc"]), int(x["experience_id"]))):
        prediction_time = _parse_ts(row["created_at_utc"])
        while ptr < len(settled_train):
            candidate = settled_train[ptr]
            created = _parse_ts(candidate["created_at_utc"])
            settled = _parse_ts(candidate["settled_at_utc"])
            if not (created < prediction_time and settled < prediction_time):
                break
            _memory_add(memory_state, candidate)
            ptr += 1
        global_values.append(_safe01((memory_state["errors"] + 1.0) / (memory_state["total"] + 2.0)))
        memory_values.append(_memory_risk(memory_state, row))

    # Restore validation order after the causal state pass.
    order_index = {int(row["experience_id"]): i for i, row in enumerate(sorted(validation, key=lambda x: (_parse_ts(x["created_at_utc"]), int(x["experience_id"]))))}
    ordered_global = np.asarray([global_values[order_index[int(row["experience_id"])] ] for row in validation], dtype=float)
    ordered_memory = np.asarray([memory_values[order_index[int(row["experience_id"])] ] for row in validation], dtype=float)
    candidate_values = {
        "global": ordered_global,
        "case_memory": ordered_memory,
        "meta": meta_values,
    }
    scores = {name: _binary_logloss(labels, candidate_values[name]) for name in CANDIDATES}
    best = _select_source_from_scores(scores)
    return best, scores, len(validation)



def _select_source_from_scores(scores: dict[str, float]) -> str:
    """Select a risk source only when its validation gain is material."""
    best = min(CANDIDATES, key=lambda c: (scores[c], CANDIDATES.index(c)))
    global_score = float(scores["global"])
    if best != "global":
        required = global_score * (1.0 - MIN_RELATIVE_GAIN)
        if scores[best] >= required or scores[best] >= global_score - TIE_EPS:
            return "global"
    return best



def _mll(labels: np.ndarray, p: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    p = np.asarray(p, dtype=float)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(labels)), labels], 1e-9, 1.0))))


def _brier(labels: np.ndarray, p: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    p = np.asarray(p, dtype=float)
    one = np.eye(3, dtype=float)[labels]
    return float(np.mean(np.sum((p - one) ** 2, axis=1)))


def _risk_adjust(base: np.ndarray, risk: float, baseline_error: float) -> tuple[np.ndarray, str]:
    adjusted, action, _ = _adjust_probabilities(base, risk, baseline_error)
    return adjusted, action


def evaluate_horizon(rows: list[Any], horizon: str) -> dict[str, Any]:
    ordered = sorted(
        [r for r in rows if str(r["horizon"]) == horizon],
        key=lambda r: (_parse_ts(r["created_at_utc"]), _parse_ts(r["settled_at_utc"]), int(r["experience_id"])),
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

    y_values: list[int] = []
    base_probs: list[np.ndarray] = []
    adjusted_probs: list[np.ndarray] = []
    risks: list[float] = []
    actions: list[str] = []
    sources: list[str] = []
    case_keys: list[tuple[str, str, str, str, str, str]] = []
    source_validation_logloss: dict[str, list[float]] = {c: [] for c in CANDIDATES}
    pit_excluded_candidate_count = 0
    deferred_cases = 0
    refresh_count = 0
    source_changes = 0
    previous_source: str | None = None
    previous_candidate: str | None = None
    candidate_streak = 0
    raw_source_counts: dict[str, int] = {c: 0 for c in CANDIDATES}

    for block_start in range(MIN_TRAIN + VALIDATION_SIZE, len(ordered), ROUTER_REFRESH):
        block_end = min(len(ordered), block_start + ROUTER_REFRESH)
        first_current = ordered[block_start]
        matured = _eligible_prior(ordered[:block_start], first_current)
        pit_excluded_candidate_count += max(0, block_start - len(matured))
        if len(matured) < MIN_TRAIN + VALIDATION_SIZE:
            deferred_cases += block_end - block_start
            continue

        source, validation_scores, validation_n = _choose_source(matured, first_current)
        refresh_count += 1
        for name, value in validation_scores.items():
            if math.isfinite(value):
                source_validation_logloss[name].append(float(value))
        if previous_source is not None and source != previous_source:
            source_changes += 1
        previous_source = source

        memory_state = _memory_state(matured)
        seen_ids = {int(row["experience_id"]) for row in matured}
        meta_bundle = None
        meta_default = _baseline_error(matured)
        if source == "meta":
            meta_bundle, meta_default = _fit_meta_model(matured)

        block_rows = ordered[block_start:block_end]
        frozen_meta = _predict_meta(meta_bundle, block_rows, meta_default)
        frozen_index = {int(row["experience_id"]): i for i, row in enumerate(block_rows)}

        for local_idx, index in enumerate(range(block_start, block_end)):
            current = ordered[index]
            current_matured = _eligible_prior(ordered[:index], current)
            new_rows = [
                row for row in current_matured
                if int(row["experience_id"]) not in seen_ids
            ]
            for row in sorted(new_rows, key=lambda x: (_parse_ts(x["settled_at_utc"]), int(x["experience_id"]))):
                _memory_add(memory_state, row)
                seen_ids.add(int(row["experience_id"]))

            baseline = _safe01((memory_state["errors"] + 1.0) / (memory_state["total"] + 2.0))
            if source == "global":
                risk = baseline
            elif source == "case_memory":
                risk = _memory_risk(memory_state, current)
            else:
                risk = float(frozen_meta[frozen_index[int(current["experience_id"])]])
            risk = _safe01(risk)

            base = _probabilities(current)
            adjusted, action = _risk_adjust(base, risk, baseline)
            actual = str(current["actual_direction"])
            y_values.append(CLASSES.index(actual))
            base_probs.append(base)
            adjusted_probs.append(adjusted)
            risks.append(risk)
            actions.append(action)
            sources.append(source)
            case_keys.append(_case_key(current))

    if not y_values:
        return {
            "status": "DEFERRED",
            "reason": "no_pit_valid_prequential_cases",
            "n": len(ordered),
            "research_only": True,
            "production_changed": False,
            "promotion_evidence_eligible": False,
        }

    y = np.asarray(y_values, dtype=int)
    base = np.vstack(base_probs)
    adjusted = np.vstack(adjusted_probs)
    risk_arr = np.asarray(risks, dtype=float)
    base_pred = np.argmax(base, axis=1)
    adjusted_pred = np.argmax(adjusted, axis=1)
    error_labels = (base_pred != y).astype(int)
    coverage_mask = np.asarray(actions) != "ABSTAIN"

    block_size = 40
    blocks: list[dict[str, Any]] = []
    for start in range(0, len(y), block_size):
        end = min(len(y), start + block_size)
        if end - start < 20:
            continue
        sl = slice(start, end)
        block_y = y[sl]
        block_base = base[sl]
        block_adjusted = adjusted[sl]
        bp = np.argmax(block_base, axis=1)
        ap = np.argmax(block_adjusted, axis=1)
        blocks.append({
            "index": len(blocks),
            "start": start,
            "end": end,
            "n": end - start,
            "baseline_accuracy": float(np.mean(bp == block_y)),
            "routed_accuracy": float(np.mean(ap == block_y)),
            "baseline_logloss": _mll(block_y, block_base),
            "routed_logloss": _mll(block_y, block_adjusted),
            "baseline_brier": _brier(block_y, block_base),
            "routed_brier": _brier(block_y, block_adjusted),
            "delta_accuracy": float(np.mean(ap == block_y) - np.mean(bp == block_y)),
            "delta_logloss": float(_mll(block_y, block_adjusted) - _mll(block_y, block_base)),
            "delta_brier": float(_brier(block_y, block_adjusted) - _brier(block_y, block_base)),
            "source_counts": {name: sources[start:end].count(name) for name in CANDIDATES},
        })

    case_groups: dict[str, dict[str, Any]] = {}
    action_array = np.asarray(actions)
    for key in sorted(set(case_keys)):
        mask = np.asarray([k == key for k in case_keys], dtype=bool)
        if int(mask.sum()) < MIN_CASE_SUPPORT:
            continue
        label = "|".join(key)
        case_groups[label] = {
            "n": int(mask.sum()),
            "base_accuracy": float(np.mean(base_pred[mask] == y[mask])),
            "adjusted_accuracy": float(np.mean(adjusted_pred[mask] == y[mask])),
            "base_logloss": _mll(y[mask], base[mask]),
            "adjusted_logloss": _mll(y[mask], adjusted[mask]),
            "mean_risk": float(np.mean(risk_arr[mask])),
            "observed_error_rate": float(np.mean(error_labels[mask])),
            "abstain_rate": float(np.mean(action_array[mask] == "ABSTAIN")),
            "source_case": {
                "horizon": key[0],
                "regime": key[1],
                "predicted_direction": key[2],
                "confidence_bucket": key[3],
                "production_mode": key[4],
                "information_state": key[5],
            },
        }

    source_counts = {name: int(sources.count(name)) for name in CANDIDATES}
    return {
        "status": "OK",
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "horizon": horizon,
        "n": len(ordered),
        "prequential_test_rows": len(y),
        "learning_boundary": "prior_matured_experience_only_with_explicit_router_refresh",
        "pit_violation_count": 0,
        "pit_excluded_candidate_count": int(pit_excluded_candidate_count),
        "deferred_cases": int(deferred_cases),
        "router_refresh_count": int(refresh_count),
        "router_refresh_size": ROUTER_REFRESH,
        "min_consecutive_selections": MIN_CONSECUTIVE_SELECTIONS,
        "min_relative_gain": MIN_RELATIVE_GAIN,
        "source_changes": int(source_changes),
        "candidate_sources": CANDIDATES,
        "selection_rule": f"lowest_validation_logloss_requires_{MIN_RELATIVE_GAIN:.3f}_relative_gain_and_global_fallback_within_{TIE_EPS}",
        "source_counts": source_counts,
        "raw_source_counts": raw_source_counts,
        "chronological_blocks": blocks,
        "block_size": block_size,
        "mean_validation_logloss_by_source": {
            name: (float(np.mean(values)) if values else None)
            for name, values in source_validation_logloss.items()
        },
        "baseline": {
            "accuracy": float(np.mean(base_pred == y)),
            "logloss": _mll(y, base),
            "brier": _brier(y, base),
        },
        "routed": {
            "accuracy": float(np.mean(adjusted_pred == y)),
            "logloss": _mll(y, adjusted),
            "brier": _brier(y, adjusted),
        },
        "delta_routed_minus_baseline": {
            "accuracy": float(np.mean(adjusted_pred == y) - np.mean(base_pred == y)),
            "logloss": float(_mll(y, adjusted) - _mll(y, base)),
            "brier": float(_brier(y, adjusted) - _brier(y, base)),
        },
        "selective": {
            "coverage": float(np.mean(coverage_mask)),
            "abstain_rate": float(1.0 - np.mean(coverage_mask)),
            "accuracy_on_covered": (
                float(np.mean(adjusted_pred[coverage_mask] == y[coverage_mask]))
                if np.any(coverage_mask) else None
            ),
        },
        "predictability": {
            "risk_target": "base_prediction_error",
            "logloss": _binary_logloss(error_labels, risk_arr),
            "brier": float(np.mean((risk_arr - error_labels) ** 2)),
            "auc": float(roc_auc_score(error_labels, risk_arr)) if len(np.unique(error_labels)) == 2 else None,
            "mean_predicted_risk": float(np.mean(risk_arr)),
            "observed_error_rate": float(np.mean(error_labels)),
        },
        "case_group_metrics": case_groups,
    }


def load_rows() -> list[Any]:
    init_db()
    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row
        return list(con.execute(
            """SELECT * FROM experience_ledger
               WHERE actual_direction IN ('DOWN','FLAT','UP')
                 AND settled_at_utc IS NOT NULL
               ORDER BY settled_at_utc, experience_id"""
        ).fetchall())


def build() -> dict[str, Any]:
    rows = load_rows()
    payload = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "description": (
            "Efficient prequential predictability router with explicit model lifetime, "
            "causal validation, hierarchical case memory, and chronological diagnostics."
        ),
        "config": {
            "min_train": MIN_TRAIN,
            "validation_size": VALIDATION_SIZE,
            "router_refresh": ROUTER_REFRESH,
            "tie_eps": TIE_EPS,
        },
        "horizons": {h: evaluate_horizon(rows, h) for h in ("5m", "10m")},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False))