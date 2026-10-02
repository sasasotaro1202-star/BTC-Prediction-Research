"""Research-only PIT-safe calibration of case-predictability risk.

The raw case-memory error probability is calibrated from previously evaluated
prediction cases only when both their prediction and outcome-settlement times
are strictly before the current prediction time. Calibration is diagnostic and
never mutates production state.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

try:
    from experience_pit_scope import load_strict_verified_rows
except ModuleNotFoundError:
    from src.experience_pit_scope import load_strict_verified_rows

from experience_case_adaptive_controller_oos import (
    _adjust_probabilities,
    _baseline_error,
    _parse_ts,
    _probabilities,
    _safe01,
)
from experience_predictability_router_oos import _memory_risk, _memory_state

try:
    from db import DB, init_db
except ModuleNotFoundError:
    from src.db import DB, init_db

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "experience_predictability_calibration_oos.json"

MIN_TRAIN = 100
MIN_CALIBRATION = 30
MAX_CALIBRATION = 120
BLOCK_SIZE = 40
BOOTSTRAP_SAMPLES = 2000
EPS = 1e-9


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


def _fit_calibrator(
    history: list[tuple[datetime, datetime, float, int]],
    current_time: datetime,
    current_risk: float,
) -> tuple[float, str, int]:
    eligible = [
        item
        for item in history
        if item[0] < current_time and item[1] < current_time
    ]
    eligible.sort(key=lambda item: item[0])
    if len(eligible) > MAX_CALIBRATION:
        eligible = eligible[-MAX_CALIBRATION:]
    if len(eligible) < MIN_CALIBRATION:
        return float(_safe01(current_risk)), "FALLBACK_RAW", len(eligible)

    x = np.asarray([[float(item[2])] for item in eligible], dtype=float)
    y = np.asarray([int(item[3]) for item in eligible], dtype=int)
    if len(np.unique(y)) < 2:
        return float(_safe01(current_risk)), "FALLBACK_RAW_SINGLE_CLASS", len(eligible)
    try:
        model = LogisticRegression(C=0.5, max_iter=1000, random_state=42042)
        model.fit(x, y)
        class_index = {int(cls): i for i, cls in enumerate(model.classes_)}
        if 1 not in class_index:
            return float(_safe01(current_risk)), "FALLBACK_RAW_NO_ERROR_CLASS", len(eligible)
        value = float(
            model.predict_proba(np.asarray([[float(current_risk)]], dtype=float))[0, class_index[1]]
        )
        return float(np.clip(value, 0.01, 0.99)), "FITTED_PRIOR_ONLY_LOGISTIC", len(eligible)
    except (TypeError, ValueError, FloatingPointError):
        return float(_safe01(current_risk)), "FALLBACK_RAW_FIT_ERROR", len(eligible)


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        hi = p <= edges[i + 1] if i == bins - 1 else p < edges[i + 1]
        mask = (p >= edges[i]) & hi
        if np.any(mask):
            total += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(total)


def _risk_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int | None]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    result: dict[str, float | int | None] = {
        "n": int(len(y)),
        "brier": float(np.mean((p - y) ** 2)) if len(y) else None,
        "ece": _ece(y, p) if len(y) else None,
        "mean_predicted_risk": float(np.mean(p)) if len(y) else None,
        "observed_error_rate": float(np.mean(y)) if len(y) else None,
    }
    if len(np.unique(y)) >= 2:
        result["auc"] = float(roc_auc_score(y, p))
    else:
        result["auc"] = None
    return result


def _prediction_metrics(y_idx: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_idx = np.asarray(y_idx, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    return {
        "accuracy": float(np.mean(np.argmax(p, axis=1) == y_idx)),
        "logloss": float(
            -np.mean(np.log(np.clip(p[np.arange(len(y_idx)), y_idx], EPS, 1.0)))
        ),
        "brier": float(
            np.mean(np.sum((p - np.eye(3, dtype=float)[y_idx]) ** 2, axis=1))
        ),
    }


def _paired_bootstrap(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        mean = float(values.mean()) if len(values) else float("nan")
        return {"mean": mean, "low": mean, "high": mean}
    rng = np.random.default_rng(42042)
    idx = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
    }


def _risk_coverage(errors: np.ndarray, risk: np.ndarray) -> dict[str, dict[str, float | int]]:
    errors = np.asarray(errors, dtype=int)
    risk = np.asarray(risk, dtype=float)
    if len(errors) == 0:
        return {}
    order = np.argsort(risk, kind="mergesort")
    result: dict[str, dict[str, float | int]] = {}
    for coverage in (0.90, 0.80, 0.70, 0.60, 0.50):
        k = max(1, int(round(len(errors) * coverage)))
        kept = errors[order[:k]]
        result[f"{coverage:.2f}"] = {
            "coverage": float(k / len(errors)),
            "error_rate": float(np.mean(kept)),
            "accuracy": float(1.0 - np.mean(kept)),
            "n": int(k),
        }
    return result


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
            "research_only": True,
            "production_changed": False,
            "promotion_evidence_eligible": False,
        }

    risk_history: list[tuple[datetime, datetime, float, int]] = []
    raw_risks: list[float] = []
    calibrated_risks: list[float] = []
    raw_probs: list[np.ndarray] = []
    calibrated_probs: list[np.ndarray] = []
    labels: list[int] = []
    methods: dict[str, int] = {}
    calibration_counts: list[int] = []
    pit_excluded = 0

    for index in range(MIN_TRAIN, len(ordered)):
        current = ordered[index]
        current_time = _parse_ts(current["created_at_utc"])
        matured = _eligible_prior(ordered[:index], current)
        pit_excluded += max(0, index - len(matured))
        if len(matured) < MIN_TRAIN:
            continue

        memory = _memory_state(matured)
        raw_risk = _safe01(_memory_risk(memory, current))
        baseline_error = _baseline_error(matured)
        calibrated_risk, method, count = _fit_calibrator(
            risk_history, current_time, raw_risk
        )
        methods[method] = methods.get(method, 0) + 1
        calibration_counts.append(int(count))

        base = _probabilities(current)
        raw_probability, _ = _adjust_probabilities(
            base, raw_risk, baseline_error
        )
        calibrated_probability, _ = _adjust_probabilities(
            base, calibrated_risk, baseline_error
        )

        actual = str(current["actual_direction"])
        if actual not in ("DOWN", "FLAT", "UP"):
            continue
        labels.append(("DOWN", "FLAT", "UP").index(actual))
        raw_risks.append(raw_risk)
        calibrated_risks.append(calibrated_risk)
        raw_probs.append(raw_probability)
        calibrated_probs.append(calibrated_probability)

        error = int(np.argmax(base) != labels[-1])
        settled = _parse_ts(current["settled_at_utc"])
        risk_history.append((current_time, settled, raw_risk, error))

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
    raw_risk_arr = np.asarray(raw_risks, dtype=float)
    calibrated_risk_arr = np.asarray(calibrated_risks, dtype=float)
    raw = np.vstack(raw_probs)
    calibrated = np.vstack(calibrated_probs)
    error_labels = (np.argmax(raw, axis=1) != y).astype(int)

    raw_pred = _prediction_metrics(y, raw)
    cal_pred = _prediction_metrics(y, calibrated)
    raw_risk_metrics = _risk_metrics(error_labels, raw_risk_arr)
    cal_risk_metrics = _risk_metrics(error_labels, calibrated_risk_arr)

    ll_delta = (
        -np.log(np.clip(calibrated[np.arange(len(y)), y], EPS, 1.0))
        + np.log(np.clip(raw[np.arange(len(y)), y], EPS, 1.0))
    )
    one_hot = np.eye(3, dtype=float)[y]
    brier_delta = np.sum((calibrated - one_hot) ** 2, axis=1) - np.sum(
        (raw - one_hot) ** 2, axis=1
    )

    blocks: list[dict[str, Any]] = []
    for start in range(0, len(y), BLOCK_SIZE):
        end = min(len(y), start + BLOCK_SIZE)
        if end - start < 20:
            continue
        bm_raw = _prediction_metrics(y[start:end], raw[start:end])
        bm_cal = _prediction_metrics(y[start:end], calibrated[start:end])
        blocks.append(
            {
                "index": len(blocks),
                "start": int(start),
                "end": int(end),
                "n": int(end - start),
                "raw": bm_raw,
                "calibrated": bm_cal,
                "delta_calibrated_minus_raw": {
                    key: float(bm_cal[key] - bm_raw[key])
                    for key in ("accuracy", "logloss", "brier")
                },
            }
        )

    coverage_raw = _risk_coverage(error_labels, raw_risk_arr)
    coverage_cal = _risk_coverage(error_labels, calibrated_risk_arr)

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "horizon": horizon,
        "n": len(ordered),
        "prequential_test_rows": len(y),
        "pit_violation_count": 0,
        "pit_excluded_candidate_count": int(pit_excluded),
        "min_train": MIN_TRAIN,
        "min_calibration": MIN_CALIBRATION,
        "max_calibration": MAX_CALIBRATION,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "calibration_method_counts": methods,
        "mean_calibration_support": float(np.mean(calibration_counts)) if calibration_counts else 0.0,
        "raw_prediction_metrics": raw_pred,
        "calibrated_prediction_metrics": cal_pred,
        "delta_calibrated_minus_raw": {
            key: float(cal_pred[key] - raw_pred[key])
            for key in ("accuracy", "logloss", "brier")
        },
        "risk_metrics": {
            "raw": raw_risk_metrics,
            "calibrated": cal_risk_metrics,
            "delta_calibrated_minus_raw": {
                "brier": float(cal_risk_metrics["brier"] - raw_risk_metrics["brier"]),
                "ece": float(cal_risk_metrics["ece"] - raw_risk_metrics["ece"]),
                "auc": (
                    float(cal_risk_metrics["auc"] - raw_risk_metrics["auc"])
                    if cal_risk_metrics["auc"] is not None and raw_risk_metrics["auc"] is not None
                    else None
                ),
            },
            "raw_coverage": coverage_raw,
            "calibrated_coverage": coverage_cal,
        },
        "paired_bootstrap": {
            "logloss_delta_calibrated_minus_raw": _paired_bootstrap(ll_delta),
            "brier_delta_calibrated_minus_raw": _paired_bootstrap(brier_delta),
        },
        "chronological_blocks": blocks,
        "calibration_contract": {
            "prior_prediction_time_strictly_earlier": True,
            "prior_outcome_settled_strictly_before_current_prediction": True,
            "current_outcome_excluded": True,
            "same_prediction_time_excluded": True,
            "bounded_history": True,
            "fallback_when_insufficient_history": True,
            "production_changed": False,
            "promotion_allowed": False,
        },
    }


def build() -> dict[str, Any]:
    rows, pit_scope = load_strict_verified_rows(DB)
    results = {h: evaluate_horizon(rows, h) for h in ("5m", "10m")}
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_changed": False,
        "strict_pit_scope": True,
        "research_population": "all_verified_production_sources",
        "promotion_primary_scope": "binance_futures_only",
        "pit_scope": pit_scope,
        "promotion_evidence_eligible": False,
        "description": (
            "PIT-safe prequential calibration of case-memory prediction-error "
            "risk using only prior matured outcomes."
        ),
        "config": {
            "min_train": MIN_TRAIN,
            "min_calibration": MIN_CALIBRATION,
            "max_calibration": MAX_CALIBRATION,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "horizons": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False))
