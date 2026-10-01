"""Research-only failure-recurrence detector for matured prediction experience.

The detector asks whether failures are locally recurrent within the same prediction
case. A case is defined by horizon, regime, predicted direction, confidence bucket,
and production mode. For every test prediction, the detector uses only experiences
whose created_at_utc and settled_at_utc are strictly earlier than the prediction.

It produces recurrence features and evaluates a compact calibrated logistic model
against a causal global-error baseline. It does not alter production or create
promotion evidence.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from experience_case_adaptive_controller_oos import (
    CLASSES,
    _case_key,
    _parse_ts,
    _probabilities,
    _safe01,
)

try:
    from db import DB, init_db
except ModuleNotFoundError:
    from src.db import DB, init_db

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "experience_failure_recurrence_oos.json"

MIN_TRAIN = 100
RECENT_MATCHES = 20
RECENT_WINDOW = 10
MIN_CASE_SUPPORT = 5
RECURRENCE_REFRESH = 20


def _eligible_prior(ordered: list[Any], current: Any) -> list[Any]:
    prediction_time = _parse_ts(current["created_at_utc"])
    out: list[Any] = []
    for row in ordered:
        try:
            created = _parse_ts(row["created_at_utc"])
            settled = _parse_ts(row["settled_at_utc"])
        except (TypeError, ValueError):
            continue
        if created < prediction_time and settled < prediction_time:
            out.append(row)
    return out


def _baseline_error(prior: list[Any]) -> float:
    if not prior:
        return 0.5
    errors = sum(1 - int(row["correct"]) for row in prior)
    return _safe01((errors + 1.0) / (len(prior) + 2.0))


def _recurrence_features(prior: list[Any], current: Any) -> dict[str, float]:
    """Build recurrence features from matured experiences only.

    The helper itself enforces the PIT boundary so callers cannot accidentally
    bypass the causal filter by passing an unfiltered row collection.
    """
    eligible = _eligible_prior(prior, current)
    key = _case_key(current)
    matching = [row for row in eligible if _case_key(row) == key]
    matching.sort(key=lambda row: (_parse_ts(row["settled_at_utc"]), int(row["experience_id"])))
    recent = matching[-RECENT_MATCHES:]
    tail = recent[-RECENT_WINDOW:]

    if recent:
        recent_error_rate = float(np.mean([1 - int(row["correct"]) for row in recent]))
        tail_error_rate = float(np.mean([1 - int(row["correct"]) for row in tail]))
        streak = 0
        for row in reversed(recent):
            if int(row["correct"]) == 0:
                streak += 1
            else:
                break
        now = _parse_ts(current["created_at_utc"])
        last_failure = None
        for row in reversed(recent):
            if int(row["correct"]) == 0:
                last_failure = _parse_ts(row["settled_at_utc"])
                break
        hours_since_failure = (
            max(0.0, (now - last_failure).total_seconds() / 3600.0)
            if last_failure is not None
            else 168.0
        )
    else:
        recent_error_rate = 0.0
        tail_error_rate = 0.0
        streak = 0
        hours_since_failure = 168.0

    return {
        "case_support": float(len(matching)),
        "recent_error_rate": recent_error_rate,
        "tail_error_rate": tail_error_rate,
        "error_streak": float(streak),
        "failure_recency_decay": float(math.exp(-hours_since_failure / 6.0)),
        "case_has_failure": float(any(int(row["correct"]) == 0 for row in recent)),
        "support_log": float(math.log1p(len(matching))),
    }


def _feature_row(features: dict[str, float]) -> list[float]:
    names = (
        "recent_error_rate",
        "tail_error_rate",
        "error_streak",
        "failure_recency_decay",
        "case_has_failure",
        "support_log",
    )
    return [float(features[name]) for name in names]


def _fit_prequential_model(
    prior: list[Any],
) -> tuple[LogisticRegression | None, float]:
    """Fit one causal recurrence model for a bounded model lifetime."""
    baseline = _baseline_error(prior)
    if len(prior) < MIN_TRAIN:
        return None, baseline

    rows = []
    y = []
    for end in range(MIN_TRAIN, len(prior)):
        sub = prior[:end]
        row = prior[end]
        feat = _recurrence_features(sub, row)
        rows.append(_feature_row(feat))
        y.append(1 - int(row["correct"]))

    if len(rows) < 30 or len(set(y)) < 2:
        return None, baseline

    try:
        model = LogisticRegression(
            C=0.5,
            class_weight=None,
            max_iter=1000,
            random_state=42,
        )
        model.fit(np.asarray(rows, dtype=float), np.asarray(y, dtype=int))
        if 1 not in {int(cls) for cls in model.classes_}:
            return None, baseline
        return model, baseline
    except (TypeError, ValueError, FloatingPointError):
        return None, baseline


def _predict_prequential_model(
    model: LogisticRegression | None,
    prior: list[Any],
    current_rows: list[Any],
    default: float,
) -> np.ndarray:
    if model is None:
        return np.full(len(current_rows), default, dtype=float)
    try:
        class_index = {int(cls): idx for idx, cls in enumerate(model.classes_)}
        if 1 not in class_index:
            return np.full(len(current_rows), default, dtype=float)
        features = [
            _feature_row(_recurrence_features(prior, row))
            for row in current_rows
        ]
        return np.asarray(
            model.predict_proba(np.asarray(features, dtype=float))[
                :, class_index[1]
            ],
            dtype=float,
        )
    except (TypeError, ValueError, FloatingPointError):
        return np.full(len(current_rows), default, dtype=float)


def _fit_prequential(prior: list[Any], current: Any) -> tuple[float, bool]:
    if len(prior) < MIN_TRAIN:
        return _baseline_error(prior), False

    rows = []
    y = []
    for end in range(MIN_TRAIN, len(prior)):
        sub = prior[:end]
        row = prior[end]
        feat = _recurrence_features(sub, row)
        rows.append(_feature_row(feat))
        y.append(1 - int(row["correct"]))

    if len(rows) < 30 or len(set(y)) < 2:
        return _baseline_error(prior), False

    try:
        model = LogisticRegression(
            C=0.5,
            class_weight=None,
            max_iter=1000,
            random_state=42,
        )
        model.fit(np.asarray(rows, dtype=float), np.asarray(y, dtype=int))
        index = {int(cls): i for i, cls in enumerate(model.classes_)}
        if 1 not in index:
            return _baseline_error(prior), False
        p = float(model.predict_proba(np.asarray([_feature_row(_recurrence_features(prior, current))]))[0, index[1]])
        return _safe01(p), True
    except (TypeError, ValueError, FloatingPointError):
        return _baseline_error(prior), False


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(np.where(y == 1, p, 1.0 - p), 1e-9, 1.0))))


def evaluate_horizon(rows: list[Any], horizon: str) -> dict[str, Any]:
    ordered = sorted(
        [row for row in rows if str(row["horizon"]) == horizon],
        key=lambda row: (
            _parse_ts(row["created_at_utc"]),
            _parse_ts(row["settled_at_utc"]),
            int(row["experience_id"]),
        ),
    )
    if len(ordered) <= MIN_TRAIN:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_experience_rows",
            "n": len(ordered),
            "promotion_evidence_eligible": False,
        }

    y_values: list[int] = []
    p_model: list[float] = []
    p_base: list[float] = []
    feature_records: list[dict[str, float]] = []
    fitted = 0
    deferred_cases = 0
    pit_excluded_candidate_count = 0
    refresh_count = 0

    for block_start in range(MIN_TRAIN, len(ordered), RECURRENCE_REFRESH):
        block_end = min(len(ordered), block_start + RECURRENCE_REFRESH)
        first_current = ordered[block_start]
        prior_candidates = ordered[:block_start]
        prior = _eligible_prior(prior_candidates, first_current)
        pit_excluded_candidate_count += max(
            0, len(prior_candidates) - len(prior)
        )
        if len(prior) < MIN_TRAIN:
            deferred_cases += block_end - block_start
            continue

        model, baseline = _fit_prequential_model(prior)
        refresh_count += 1
        if model is not None:
            fitted += 1

        block_rows = ordered[block_start:block_end]
        model_predictions = _predict_prequential_model(
            model,
            prior,
            block_rows,
            baseline,
        )

        for offset, index in enumerate(range(block_start, block_end)):
            current = ordered[index]
            current_prior = _eligible_prior(
                ordered[:index],
                current,
            )
            current_baseline = _baseline_error(current_prior)
            if model is None:
                probability = current_baseline
            else:
                probability = float(model_predictions[offset])
            probability = _safe01(probability)
            feat = _recurrence_features(current_prior, current)
            y_values.append(1 - int(current["correct"]))
            p_model.append(probability)
            p_base.append(current_baseline)
            feature_records.append(feat)

    if not y_values:
        return {
            "status": "DEFERRED",
            "reason": "no_pit_valid_prequential_cases",
            "n": len(ordered),
            "promotion_evidence_eligible": False,
        }

    y = np.asarray(y_values, dtype=int)
    pm = np.asarray(p_model, dtype=float)
    pb = np.asarray(p_base, dtype=float)

    model_ll = _logloss(y, pm)
    base_ll = _logloss(y, pb)
    auc = float(roc_auc_score(y, pm)) if len(np.unique(y)) == 2 else None

    case_support = np.asarray(
        [x["case_support"] for x in feature_records],
        dtype=float,
    )
    recent_error = np.asarray(
        [x["recent_error_rate"] for x in feature_records],
        dtype=float,
    )
    streak = np.asarray(
        [x["error_streak"] for x in feature_records],
        dtype=float,
    )
    recurrence_cases = case_support >= MIN_CASE_SUPPORT

    return {
        "status": "OK",
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "horizon": horizon,
        "n": len(ordered),
        "prequential_test_rows": int(len(y)),
        "learning_boundary": (
            "only_experiences_with_created_at_utc_and_settled_at_utc_"
            "strictly_before_current_prediction_time"
        ),
        "pit_violation_count": 0,
        "pit_excluded_candidate_count": int(
            pit_excluded_candidate_count
        ),
        "deferred_cases": int(deferred_cases),
        "model_fit_count": int(fitted),
        "model_refresh_count": int(refresh_count),
        "model_refresh_size": RECURRENCE_REFRESH,
        "features": (
            "same_case_recent_error_rate",
            "same_case_tail_error_rate",
            "same_case_error_streak",
            "same_case_failure_recency_decay",
            "same_case_has_failure",
            "same_case_support",
        ),
        "model": {
            "type": "prequential_logistic_regression",
            "target": "prediction_error",
            "trained_only_on_prior_settled_experiences": True,
            "bounded_model_lifetime": True,
        },
        "baseline": {
            "logloss": base_ll,
        },
        "recurrence_model": {
            "logloss": model_ll,
            "auc": auc,
        },
        "delta_logloss_model_minus_baseline": float(
            model_ll - base_ll
        ),
        "diagnostics": {
            "cases_with_support_floor": int(np.sum(recurrence_cases)),
            "share_with_support_floor": float(np.mean(recurrence_cases)),
            "mean_case_support": float(np.mean(case_support)),
            "mean_recent_error_rate": float(np.mean(recent_error)),
            "mean_error_streak": float(np.mean(streak)),
        },
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
        "description": "Prequential failure-recurrence features within matched prediction cases.",
        "horizons": {h: evaluate_horizon(rows, h) for h in ("5m", "10m")},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False))
