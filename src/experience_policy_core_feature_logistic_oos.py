"""Research-only core-feature experience-risk logistic challenger.

Uses only compact, prediction-time features that are less coupled to specific
model versions and source/runtime error strings: confidence, entropy, margin,
time-of-day, regime, and predicted direction. Class weighting is disabled so
the empirical error prevalence remains the probability prior.
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
from sklearn.metrics import brier_score_loss, log_loss

try:
    from experience_policy_oos import _baseline, _metrics, _parse_ts, _safe_normalize_probability
    from db import DB
except ModuleNotFoundError:
    from src.experience_policy_oos import _baseline, _metrics, _parse_ts, _safe_normalize_probability
    from src.db import DB

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "experience_policy_core_feature_logistic_oos.json"
CONFIG = ROOT / "config" / "EXPERIENCE_POLICY_CORE_FEATURE_LOGISTIC_OOS.json"

DEFAULT_CONFIG = {
    "schema_version": 1,
    "min_train_rows": 100,
    "test_block_rows": 1,
    "model_c": 0.5,
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


def _core_features(row: Any) -> dict[str, float | str]:
    created = _parse_ts(row["created_at_utc"])
    hour = created.hour + created.minute / 60.0
    return {
        "confidence": float(row["confidence"]),
        "entropy": float(row["entropy"]),
        "margin": float(row["margin"]),
        "hour_sin": math.sin(2.0 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2.0 * math.pi * hour / 24.0),
        "regime": str(row["regime"] or "UNKNOWN"),
        "predicted_direction": str(row["predicted_direction"]),
    }


def _fit_predict(train_rows: list[Any], test_rows: list[Any], model_c: float):
    baseline = _baseline(train_rows)
    try:
        vectorizer = DictVectorizer(sparse=True)
        x_train = vectorizer.fit_transform([_core_features(row) for row in train_rows])
        x_test = vectorizer.transform([_core_features(row) for row in test_rows])
        y_train = np.asarray([1 - int(row["correct"]) for row in train_rows], dtype=int)
        if len(np.unique(y_train)) < 2:
            return [baseline] * len(test_rows), False, []
        model = LogisticRegression(
            C=float(model_c),
            class_weight=None,
            max_iter=1000,
            random_state=42,
        )
        model.fit(x_train, y_train)
        p = model.predict_proba(x_test)
        class_index = {int(c): i for i, c in enumerate(model.classes_)}
        probabilities = [
            _safe_normalize_probability(float(row[class_index[1]]))
            if 1 in class_index else baseline
            for row in p
        ]
        names = list(vectorizer.get_feature_names_out())
        coef = model.coef_[0]
        ranked = sorted(zip(names, coef, strict=False), key=lambda x: abs(float(x[1])), reverse=True)
        drivers = [
            f"{name}={'+' if value >= 0 else ''}{float(value):.6f}"
            for name, value in ranked[:12]
        ]
        return probabilities, True, drivers
    except (ValueError, TypeError, FloatingPointError):
        return [baseline] * len(test_rows), False, []


def evaluate_rows(
    rows: list[Any],
    *,
    min_train_rows: int = 100,
    test_block_rows: int = 1,
    model_c: float = 0.5,
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
    probs: list[float] = []
    baselines: list[float] = []
    cases: list[dict[str, Any]] = []
    model_fit_count = 0
    fallback_count = 0
    latest_drivers: list[str] = []

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

        probabilities, fitted, drivers = _fit_predict(train_rows, test_rows, model_c)
        if fitted:
            model_fit_count += 1
            latest_drivers = drivers
        else:
            fallback_count += 1
        baseline = _baseline(train_rows)
        for row, p in zip(test_rows, probabilities, strict=False):
            error = 1 - int(row["correct"])
            y_true.append(error)
            probs.append(float(p))
            baselines.append(float(baseline))
            cases.append({
                "experience_id": int(row["experience_id"]),
                "prediction_id": int(row["prediction_id"]),
                "horizon": str(row["horizon"]),
                "settled_at_utc": str(row["settled_at_utc"]),
                "predicted_direction": str(row["predicted_direction"]),
                "correct": int(row["correct"]),
                "learned_error_probability": float(p),
                "baseline_error_probability": float(baseline),
            })
        start = stop

    candidate = _metrics(y_true, probs)
    baseline = _metrics(y_true, baselines)
    return {
        "status": "OK",
        "rows": len(ordered),
        "prequential_test_rows": len(y_true),
        "min_train_rows": int(min_train_rows),
        "test_block_rows": int(max(1, int(test_block_rows))),
        "model": {
            "type": "logistic_regression",
            "target": "prediction_error",
            "feature_profile": "core_prediction_time",
            "class_weight": None,
            "trained_only_on_prior_settled_experiences": True,
            "same_settlement_timestamp_isolation": True,
            "model_fit_count": int(model_fit_count),
            "fallback_count": int(fallback_count),
            "latest_feature_drivers": latest_drivers,
        },
        "meta_error_probability": candidate,
        "baseline_error_probability": baseline,
        "relative_logloss_gain_vs_baseline": float(
            (baseline["logloss"] - candidate["logloss"]) / max(abs(baseline["logloss"]), 1e-12)
        ),
        "relative_brier_gain_vs_baseline": float(
            (baseline["brier"] - candidate["brier"]) / max(abs(baseline["brier"]), 1e-12)
        ),
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "high_risk_cases_latest": sorted(
            cases,
            key=lambda row: (-float(row["learned_error_probability"]), row["settled_at_utc"], row["experience_id"]),
        )[:max(1, int(max_report_cases))],
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
        "candidate": "core_feature_unweighted_logistic",
        "learning_boundary": (
            "outer test cases use only prior settled experiences; same-settlement "
            "cases are evaluated as one batch and cannot train one another"
        ),
        "feature_policy": [
            "confidence",
            "entropy",
            "margin",
            "hour_sin",
            "hour_cos",
            "regime",
            "predicted_direction",
        ],
        "excluded_features": [
            "model_version",
            "production_mode",
            "warning_flags",
            "data_quality_flags",
            "source",
            "actual_price",
            "actual_direction",
            "correct",
            "settled_at_utc",
        ],
        "hypothesis": (
            "a compact feature set may generalize better than source/runtime-"
            "specific categorical features"
        ),
        "config": cfg,
        "horizons": {},
    }
    for horizon, horizon_rows in rows_by_horizon.items():
        payload["horizons"][horizon] = evaluate_rows(
            horizon_rows,
            min_train_rows=int(cfg["min_train_rows"]),
            test_block_rows=int(cfg["test_block_rows"]),
            model_c=float(cfg["model_c"]),
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
