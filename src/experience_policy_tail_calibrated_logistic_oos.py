"""Research-only temporal tail-calibrated logistic challenger.

For each outer OOS case, the most recent 20% of prior settled experiences are
reserved for sigmoid calibration of a base logistic model trained on the older
80%. The final base model is then refit on all prior settled experiences before
scoring the outer test case. Settlement timestamps are never split across any
boundary.
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

try:
    from experience_policy_oos import (
        _baseline,
        _features,
        _metrics,
        _parse_ts,
        _safe_normalize_probability,
    )
    from db import DB
except ModuleNotFoundError:
    from src.experience_policy_oos import (
        _baseline,
        _features,
        _metrics,
        _parse_ts,
        _safe_normalize_probability,
    )
    from src.db import DB

try:
    from sklearn.feature_extraction import DictVectorizer
except ImportError:
    DictVectorizer = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "experience_policy_tail_calibrated_logistic_oos.json"
CONFIG = ROOT / "config" / "EXPERIENCE_POLICY_TAIL_CALIBRATED_LOGISTIC_OOS.json"

DEFAULT_CONFIG = {
    "schema_version": 1,
    "min_train_rows": 100,
    "test_block_rows": 1,
    "calibration_tail_fraction": 0.20,
    "min_calibration_rows": 20,
    "model_c": 0.5,
    "calibrator_c": 1.0,
    "thresholds": [0.55, 0.60, 0.65, 0.70],
    "max_report_cases": 100,
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    if not path.is_file():
        return dict(DEFAULT_CONFIG)
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return dict(DEFAULT_CONFIG)
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(obj)
        return cfg
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return dict(DEFAULT_CONFIG)


def _logit(values: list[float]) -> np.ndarray:
    p = np.asarray([_safe_normalize_probability(v) for v in values], dtype=float)
    return np.log(p / (1.0 - p))


def _vectorize(train_rows: list[Any], test_rows: list[Any]):
    if DictVectorizer is None:
        raise RuntimeError("sklearn DictVectorizer unavailable")
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform([_features(row) for row in train_rows])
    x_test = vectorizer.transform([_features(row) for row in test_rows])
    return vectorizer, x_train, x_test


def _fit_base(rows: list[Any], model_c: float):
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform([_features(row) for row in rows])
    y_train = np.asarray([1 - int(row["correct"]) for row in rows], dtype=int)
    if len(np.unique(y_train)) < 2:
        return None, None, None
    model = LogisticRegression(
        C=float(model_c),
        class_weight=None,
        max_iter=1000,
        random_state=42,
    )
    model.fit(x_train, y_train)
    return vectorizer, model, y_train


def _probability(vectorizer, model, rows: list[Any]) -> list[float]:
    x = vectorizer.transform([_features(row) for row in rows])
    class_index = {int(c): i for i, c in enumerate(model.classes_)}
    if 1 not in class_index:
        return [0.5] * len(rows)
    return [
        _safe_normalize_probability(float(p[class_index[1]]))
        for p in model.predict_proba(x)
    ]


def _fit_tail_calibrated(
    train_rows: list[Any],
    test_rows: list[Any],
    *,
    model_c: float,
    calibrator_c: float,
    tail_fraction: float,
    min_calibration_rows: int,
) -> tuple[list[float], bool, bool]:
    # Chronological split inside the already-prior training set.
    tail_fraction = min(0.5, max(0.05, float(tail_fraction)))
    split = max(1, min(len(train_rows) - 1, int(round(len(train_rows) * (1.0 - tail_fraction)))))
    while (
        split < len(train_rows)
        and split > 0
        and str(train_rows[split]["settled_at_utc"]) == str(train_rows[split - 1]["settled_at_utc"])
    ):
        split += 1
    fit_rows = train_rows[:split]
    calibration_rows = train_rows[split:]
    if len(fit_rows) < 20 or len(calibration_rows) < int(min_calibration_rows):
        vectorizer, model, _ = _fit_base(train_rows, model_c)
        if model is None:
            return [_baseline(train_rows)] * len(test_rows), False, True
        return _probability(vectorizer, model, test_rows), True, True

    fit_vectorizer, fit_model, _ = _fit_base(fit_rows, model_c)
    if fit_model is None:
        return [_baseline(train_rows)] * len(test_rows), False, True

    calibration_raw = _probability(fit_vectorizer, fit_model, calibration_rows)
    y_calibration = np.asarray([1 - int(row["correct"]) for row in calibration_rows], dtype=int)
    if len(np.unique(y_calibration)) < 2:
        final_vectorizer, final_model, _ = _fit_base(train_rows, model_c)
        if final_model is None:
            return [_baseline(train_rows)] * len(test_rows), False, True
        return _probability(final_vectorizer, final_model, test_rows), True, True

    calibrator = LogisticRegression(
        C=float(calibrator_c),
        class_weight=None,
        max_iter=1000,
        random_state=42,
    )
    calibrator.fit(_logit(calibration_raw).reshape(-1, 1), y_calibration)

    final_vectorizer, final_model, _ = _fit_base(train_rows, model_c)
    if final_model is None:
        return [_baseline(train_rows)] * len(test_rows), False, True
    raw_test = _probability(final_vectorizer, final_model, test_rows)
    class_index = {int(c): i for i, c in enumerate(calibrator.classes_)}
    if 1 not in class_index:
        return [_baseline(train_rows)] * len(test_rows), True, True
    calibrated = calibrator.predict_proba(_logit(raw_test).reshape(-1, 1))
    return [
        _safe_normalize_probability(float(p[class_index[1]]))
        for p in calibrated
    ], True, False


def evaluate_rows(
    rows: list[Any],
    *,
    min_train_rows: int = 100,
    test_block_rows: int = 1,
    calibration_tail_fraction: float = 0.20,
    min_calibration_rows: int = 20,
    model_c: float = 0.5,
    calibrator_c: float = 1.0,
    thresholds: tuple[float, ...] = (0.55, 0.60, 0.65, 0.70),
    max_report_cases: int = 100,
) -> dict[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (_parse_ts(row["settled_at_utc"]), int(row["experience_id"])),
    )
    if len(ordered) <= min_train_rows:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_settled_experience",
            "rows": len(ordered),
            "min_train_rows": int(min_train_rows),
            "research_only": True,
            "production_changed": False,
            "promotion_evidence_eligible": False,
        }

    y_true: list[int] = []
    probabilities: list[float] = []
    baselines: list[float] = []
    cases: list[dict[str, Any]] = []
    calibration_applied = 0
    calibration_fallback = 0

    start = int(min_train_rows)
    while (
        start > 0
        and start < len(ordered)
        and str(ordered[start]["settled_at_utc"]) == str(ordered[start - 1]["settled_at_utc"])
    ):
        start += 1
    while start < len(ordered):
        train_rows = ordered[:start]
        stop = min(len(ordered), start + max(1, int(test_block_rows)))
        while (
            stop < len(ordered)
            and str(ordered[stop]["settled_at_utc"]) == str(ordered[stop - 1]["settled_at_utc"])
        ):
            stop += 1
        test_rows = ordered[start:stop]
        if not test_rows:
            break
        p, fitted, fallback = _fit_tail_calibrated(
            train_rows,
            test_rows,
            model_c=model_c,
            calibrator_c=calibrator_c,
            tail_fraction=calibration_tail_fraction,
            min_calibration_rows=min_calibration_rows,
        )
        if fitted:
            calibration_applied += 1
        if fallback:
            calibration_fallback += 1
        baseline = _baseline(train_rows)
        for row, probability in zip(test_rows, p, strict=False):
            error = 1 - int(row["correct"])
            y_true.append(error)
            probabilities.append(float(probability))
            baselines.append(float(baseline))
            cases.append({
                "experience_id": int(row["experience_id"]),
                "prediction_id": int(row["prediction_id"]),
                "horizon": str(row["horizon"]),
                "settled_at_utc": str(row["settled_at_utc"]),
                "predicted_direction": str(row["predicted_direction"]),
                "correct": int(row["correct"]),
                "learned_error_probability": float(probability),
                "baseline_error_probability": float(baseline),
                "calibration_fallback": bool(fallback),
            })
        start = stop

    candidate = _metrics(y_true, probabilities)
    baseline = _metrics(y_true, baselines)
    threshold_metrics: dict[str, dict[str, float | int | None]] = {}
    total_errors = sum(y_true)
    all_accuracy = float(sum(1 - value for value in y_true) / len(y_true)) if y_true else 0.0
    for threshold in thresholds:
        keep = [i for i, p in enumerate(probabilities) if p < float(threshold)]
        abstain = [i for i, p in enumerate(probabilities) if p >= float(threshold)]
        kept_correct = sum(1 - y_true[i] for i in keep)
        abstain_errors = sum(y_true[i] for i in abstain)
        kept_accuracy = float(kept_correct / len(keep)) if keep else None
        threshold_metrics[f"{float(threshold):.2f}"] = {
            "threshold_error_probability": float(threshold),
            "coverage": float(len(keep) / len(y_true)) if y_true else 0.0,
            "kept_n": int(len(keep)),
            "kept_accuracy": kept_accuracy,
            "selective_accuracy_gain_vs_all": (
                float(kept_accuracy - all_accuracy) if kept_accuracy is not None else None
            ),
            "abstain_n": int(len(abstain)),
            "abstain_error_rate": float(abstain_errors / len(abstain)) if abstain else None,
            "abstain_error_capture_rate": float(abstain_errors / total_errors) if total_errors else None,
        }

    high_risk = sorted(
        cases,
        key=lambda row: (-float(row["learned_error_probability"]), row["settled_at_utc"], row["experience_id"]),
    )[:max(1, int(max_report_cases))]
    return {
        "status": "OK",
        "rows": len(ordered),
        "prequential_test_rows": len(y_true),
        "min_train_rows": int(min_train_rows),
        "test_block_rows": int(max(1, int(test_block_rows))),
        "model": {
            "type": "logistic_regression",
            "class_weight": None,
            "probability_calibration": "temporal_tail_sigmoid",
            "calibration_tail_fraction": float(calibration_tail_fraction),
            "min_calibration_rows": int(min_calibration_rows),
            "trained_only_on_prior_settled_experiences": True,
            "same_settlement_timestamp_isolation": True,
            "calibration_applied_count": int(calibration_applied),
            "calibration_fallback_count": int(calibration_fallback),
        },
        "meta_error_probability": candidate,
        "baseline_error_probability": baseline,
        "relative_logloss_gain_vs_baseline": float(
            (baseline["logloss"] - candidate["logloss"]) / max(abs(baseline["logloss"]), 1e-12)
        ),
        "relative_brier_gain_vs_baseline": float(
            (baseline["brier"] - candidate["brier"]) / max(abs(baseline["brier"]), 1e-12)
        ),
        "threshold_policy_candidates": threshold_metrics,
        "high_risk_cases_latest": high_risk,
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
    }


def build(
    db_path: Path = DB,
    *,
    config_path: Path = CONFIG,
    output_path: Path = OUT,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    rows_by_horizon: dict[str, list[Any]] = {"5m": [], "10m": []}
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """SELECT * FROM experience_ledger
               WHERE horizon IN ('5m','10m')
               ORDER BY settled_at_utc, experience_id"""
        ).fetchall()
    for row in rows:
        rows_by_horizon[str(row["horizon"])].append(row)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "candidate": "temporal_tail_calibrated_unweighted_logistic",
        "learning_boundary": (
            "outer test cases use only prior settled experiences; the prior set "
            "is split chronologically at a settlement boundary for calibration; "
            "same-settlement outcomes cannot train one another"
        ),
        "hypothesis": (
            "temporal sigmoid calibration may improve probability quality while "
            "preserving the useful error-risk ranking seen in the unweighted challenger"
        ),
        "config": cfg,
        "horizons": {},
    }
    thresholds = tuple(float(x) for x in cfg["thresholds"])
    for horizon, horizon_rows in rows_by_horizon.items():
        payload["horizons"][horizon] = evaluate_rows(
            horizon_rows,
            min_train_rows=int(cfg["min_train_rows"]),
            test_block_rows=int(cfg["test_block_rows"]),
            calibration_tail_fraction=float(cfg["calibration_tail_fraction"]),
            min_calibration_rows=int(cfg["min_calibration_rows"]),
            model_c=float(cfg["model_c"]),
            calibrator_c=float(cfg["calibrator_c"]),
            thresholds=thresholds,
            max_report_cases=int(cfg["max_report_cases"]),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False))
