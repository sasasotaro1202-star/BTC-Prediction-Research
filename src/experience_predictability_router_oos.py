"""Research-only prequential router for prediction-error risk estimation.

Instead of fixing one predictability estimator, this candidate chooses among:
- a causal global error baseline;
- hierarchical case-memory error risk;
- a prequential logistic meta-model.

The estimator is selected using only an earlier validation slice of already
matured experiences, then refit/recomputed on all matured experiences before the
current prediction. Production remains untouched.
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
    _fit_meta_probability,
    _hierarchical_prior,
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


def _meta_predictions(train_rows: list[Any], prediction_rows: list[Any]) -> np.ndarray:
    baseline = _baseline_error(train_rows)
    if not prediction_rows or len(train_rows) < CASE_MIN_TRAIN:
        return np.full(len(prediction_rows), baseline, dtype=float)
    try:
        x_train_dict = [
            {
                k: v
                for k, v in _meta_features(row).items()
            }
            for row in train_rows
        ]
        x_pred_dict = [
            {
                k: v
                for k, v in _meta_features(row).items()
            }
            for row in prediction_rows
        ]
        y = np.asarray([1 - int(row["correct"]) for row in train_rows], dtype=int)
        if len(np.unique(y)) < 2:
            return np.full(len(prediction_rows), baseline, dtype=float)
        vectorizer = DictVectorizer(sparse=True)
        x_train = vectorizer.fit_transform(x_train_dict)
        x_pred = vectorizer.transform(x_pred_dict)
        model = LogisticRegression(
            C=0.5,
            class_weight=None,
            max_iter=1000,
            random_state=42,
        )
        model.fit(x_train, y)
        class_index = {int(cls): idx for idx, cls in enumerate(model.classes_)}
        if 1 not in class_index:
            return np.full(len(prediction_rows), baseline, dtype=float)
        return np.asarray(
            model.predict_proba(x_pred)[:, class_index[1]],
            dtype=float,
        )
    except (TypeError, ValueError, FloatingPointError):
        return np.full(len(prediction_rows), baseline, dtype=float)


def _risk(candidate: str, train_rows: list[Any], row: Any) -> float:
    if candidate == "global":
        return _baseline_error(train_rows)
    if candidate == "case_memory":
        return _hierarchical_prior(train_rows, row)
    if candidate == "meta":
        return _fit_meta_probability(train_rows, row)[0]
    raise ValueError(f"unknown_candidate:{candidate}")


def _binary_logloss(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0 - 1e-9)
    y = np.asarray(y, dtype=int)
    return float(-np.mean(np.where(y == 1, np.log(p), np.log(1.0 - p))))


def _choose_source(prior: list[Any], current: Any) -> tuple[str, dict[str, float], int]:
    eligible = _eligible_prior(prior, current)
    if len(eligible) < MIN_TRAIN + VALIDATION_SIZE:
        return "global", {"global": float("nan"), "case_memory": float("nan"), "meta": float("nan")}, 0

    train = eligible[:-VALIDATION_SIZE]
    validation = eligible[-VALIDATION_SIZE:]
    if len(train) < MIN_TRAIN:
        return "global", {"global": float("nan"), "case_memory": float("nan"), "meta": float("nan")}, 0

    labels = np.asarray(
        [1 - int(row["correct"]) for row in validation],
        dtype=int,
    )
    scores: dict[str, float] = {}
    global_values = np.full(len(validation), _baseline_error(train), dtype=float)
    memory_values = np.asarray(
        [_hierarchical_prior(train, row) for row in validation],
        dtype=float,
    )
    meta_values = _meta_predictions(train, validation)
    candidate_values = {
        "global": global_values,
        "case_memory": memory_values,
        "meta": meta_values,
    }
    for candidate in CANDIDATES:
        scores[candidate] = _binary_logloss(labels, candidate_values[candidate])

    best = min(CANDIDATES, key=lambda c: (scores[c], CANDIDATES.index(c)))
    global_score = scores["global"]
    if best != "global" and scores[best] >= global_score - TIE_EPS:
        best = "global"
    return best, scores, len(validation)


def evaluate_horizon(rows: list[Any], horizon: str) -> dict[str, Any]:
    ordered = sorted(
        [r for r in rows if str(r["horizon"]) == horizon],
        key=lambda r: (
            _parse_ts(r["created_at_utc"]),
            _parse_ts(r["settled_at_utc"]),
            int(r["experience_id"]),
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

    y_values: list[int] = []
    base_probs: list[np.ndarray] = []
    adjusted_probs: list[np.ndarray] = []
    risks: list[float] = []
    actions: list[str] = []
    sources: list[str] = []
    case_keys: list[tuple[str, str, str, str, str]] = []
    source_validation_logloss: dict[str, list[float]] = {c: [] for c in CANDIDATES}
    pit_excluded_candidate_count = 0
    deferred_cases = 0

    for idx in range(MIN_TRAIN + VALIDATION_SIZE, len(ordered)):
        current = ordered[idx]
        prior_candidates = ordered[:idx]
        prior = _eligible_prior(prior_candidates, current)
        pit_excluded_candidate_count += max(0, len(prior_candidates) - len(prior))
        if len(prior) < MIN_TRAIN + VALIDATION_SIZE:
            deferred_cases += 1
            continue

        source, scores, validation_n = _choose_source(prior, current)
        if validation_n:
            for name, value in scores.items():
                if math.isfinite(value):
                    source_validation_logloss[name].append(value)

        risk = _risk(source, prior, current)
        risk = _safe01(risk)
        base = _probabilities(current)
        baseline = _baseline_error(prior)
        adjusted, action, _ = _adjust_probabilities(base, risk, baseline)

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
    coverage = float(np.mean(coverage_mask))
    # Use the same multiclass metrics as the controller.
    def mll(prob: np.ndarray, labels: np.ndarray) -> float:
        return float(
            -np.mean(np.log(np.clip(prob[np.arange(len(labels)), labels], 1e-9, 1.0)))
        )

    def brier(prob: np.ndarray, labels: np.ndarray) -> float:
        one = np.eye(3, dtype=float)[labels]
        return float(np.mean(np.sum((prob - one) ** 2, axis=1)))

    source_counts = {name: int(sources.count(name)) for name in CANDIDATES}
    predictability = {
        "risk_target": "base_prediction_error",
        "logloss": _binary_logloss(error_labels, risk_arr),
        "brier": float(np.mean((risk_arr - error_labels) ** 2)),
        "auc": (
            float(roc_auc_score(error_labels, risk_arr))
            if len(np.unique(error_labels)) == 2 else None
        ),
        "mean_predicted_risk": float(np.mean(risk_arr)),
        "observed_error_rate": float(np.mean(error_labels)),
    }

    case_groups: dict[str, dict[str, Any]] = {}
    for key in sorted(set(case_keys)):
        mask = np.asarray([k == key for k in case_keys], dtype=bool)
        if int(mask.sum()) < MIN_CASE_SUPPORT:
            continue
        label = "|".join(key)
        case_groups[label] = {
            "n": int(mask.sum()),
            "base_accuracy": float(np.mean(base_pred[mask] == y[mask])),
            "adjusted_accuracy": float(np.mean(adjusted_pred[mask] == y[mask])),
            "base_logloss": mll(base[mask]),
            "adjusted_logloss": mll(adjusted[mask]),
            "mean_risk": float(np.mean(risk_arr[mask])),
            "observed_error_rate": float(np.mean(error_labels[mask])),
            "abstain_rate": float(np.mean(np.asarray(actions)[mask] == "ABSTAIN")),
        }

    return {
        "status": "OK",
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "horizon": horizon,
        "n": len(ordered),
        "prequential_test_rows": int(len(y)),
        "learning_boundary": "router_selected_only_from_validation_slices_of_prior_matured_experiences",
        "pit_violation_count": 0,
        "pit_excluded_candidate_count": int(pit_excluded_candidate_count),
        "deferred_cases": int(deferred_cases),
        "candidate_sources": CANDIDATES,
        "selection_rule": f"lowest_validation_logloss_with_global_fallback_within_{TIE_EPS}",
        "source_counts": source_counts,
        "mean_validation_logloss_by_source": {
            name: (
                float(np.mean(values)) if values else None
            )
            for name, values in source_validation_logloss.items()
        },
        "baseline": {
            "accuracy": float(np.mean(base_pred == y)),
            "logloss": mll(base, y),
            "brier": brier(base, y),
        },
        "routed": {
            "accuracy": float(np.mean(adjusted_pred == y)),
            "logloss": mll(adjusted, y),
            "brier": brier(adjusted, y),
        },
        "delta_routed_minus_baseline": {
            "accuracy": float(np.mean(adjusted_pred == y) - np.mean(base_pred == y)),
            "logloss": float(mll(adjusted, y) - mll(base, y)),
            "brier": float(brier(adjusted, y) - brier(base, y)),
        },
        "selective": {
            "coverage": coverage,
            "abstain_rate": float(1.0 - coverage),
            "accuracy_on_covered": (
                float(np.mean(adjusted_pred[coverage_mask] == y[coverage_mask]))
                if np.any(coverage_mask) else None
            ),
        },
        "predictability": predictability,
        "case_group_metrics": case_groups,
    }


def load_rows() -> list[Any]:
    init_db()
    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row
        return list(
            con.execute(
                """SELECT *
                   FROM experience_ledger
                   WHERE actual_direction IN ('DOWN','FLAT','UP')
                     AND settled_at_utc IS NOT NULL
                   ORDER BY settled_at_utc, experience_id"""
            ).fetchall()
        )


def build() -> dict[str, Any]:
    rows = load_rows()
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "description": (
            "Prequential predictability router selecting the safest error-risk "
            "estimator from global, case-memory, and meta-model candidates."
        ),
        "horizons": {h: evaluate_horizon(rows, h) for h in ("5m", "10m")},
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
