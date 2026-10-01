"""Research-only prequential learning from settled BTC experience.

try:
    from experience_pit_scope import load_strict_primary_rows
except ModuleNotFoundError:
    from src.experience_pit_scope import load_strict_primary_rows
The learner predicts the probability that the next prediction will be wrong.
For every evaluated experience, training data contains only earlier settled
experiences. The learned risk score can later inform abstention/confidence
policies, but this module never mutates production state.
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
    from db import DB, init_db
except ModuleNotFoundError:
    from src.db import DB, init_db

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "experience_policy_oos.json"
CONFIG = ROOT / "config" / "EXPERIENCE_POLICY_OOS.json"

DEFAULT_CONFIG = {
    "schema_version": 1,
    "min_train_rows": 100,
    "thresholds": [0.55, 0.60, 0.65, 0.70],
    "max_report_cases": 100,
    "model_c": 0.5,
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config(path: Path = CONFIG) -> dict[str, Any]:
    if not path.is_file():
        return dict(DEFAULT_CONFIG)
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return dict(DEFAULT_CONFIG)
        out = dict(DEFAULT_CONFIG)
        out.update(obj)
        return out
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return dict(DEFAULT_CONFIG)


def _parse_ts(value: Any) -> datetime:
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


def _flag_tokens(value: Any, prefix: str) -> list[str]:
    if value in (None, "", "[]", "null"):
        return []
    try:
        obj = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return [prefix + ":invalid_json"]
    if isinstance(obj, list):
        return [prefix + ":" + str(item) for item in obj if item not in (None, "")]
    return [prefix + ":present"]


def _features(row: Any) -> dict[str, float | str]:
    """Prediction-time features only; no realized outcome fields are included."""
    created = _parse_ts(row["created_at_utc"])
    hour = created.hour + created.minute / 60.0
    confidence = float(row["confidence"])
    entropy = float(row["entropy"])
    margin = float(row["margin"])
    values: dict[str, float | str] = {
        "confidence": confidence,
        "entropy": entropy,
        "margin": margin,
        "hour_sin": math.sin(2.0 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2.0 * math.pi * hour / 24.0),
        "warning_count": float(len(_flag_tokens(row["warning_flags"], "warning"))),
        "quality_flag_count": float(len(_flag_tokens(row["data_quality_flags"], "quality"))),
        "regime": str(row["regime"] or "UNKNOWN"),
        "predicted_direction": str(row["predicted_direction"]),
        "production_mode": str(row["production_mode"] or "UNKNOWN"),
        "model_version": str(row["model_version"] or "UNKNOWN"),
    }
    for token in _flag_tokens(row["warning_flags"], "warning"):
        values["flag:" + token] = 1.0
    for token in _flag_tokens(row["data_quality_flags"], "quality"):
        values["flag:" + token] = 1.0
    return values


def _safe_normalize_probability(value: float) -> float:
    return float(min(1.0 - 1e-6, max(1e-6, value)))


def _baseline(train_rows: list[Any]) -> float:
    errors = sum(int(row["correct"] == 0) for row in train_rows)
    n = len(train_rows)
    return _safe_normalize_probability((errors + 1.0) / (n + 2.0))


def _confidence_bucket(confidence: float) -> str:
    if confidence < 0.40:
        return "0.33-0.40"
    if confidence < 0.50:
        return "0.40-0.50"
    if confidence < 0.60:
        return "0.50-0.60"
    if confidence < 0.70:
        return "0.60-0.70"
    return "0.70+"


def _hierarchical_key(row: Any, level: int) -> tuple[str, ...]:
    base = (
        str(row["regime"] or "UNKNOWN"),
        str(row["predicted_direction"]),
        _confidence_bucket(float(row["confidence"])),
        str(row["production_mode"] or "UNKNOWN"),
    )
    if level == 1:
        return base
    if level == 2:
        return base[:2]
    return base[:1]


def _hierarchical_memory_predict(
    train_rows: list[Any],
    test_rows: list[Any],
    shrinkage: float = 20.0,
) -> list[float]:
    """Estimate next-case error probability using only prior settled rows."""
    global_error = _baseline(train_rows)
    grouped: dict[tuple[str, tuple[str, ...]], list[int]] = {}
    for level in (1, 2, 3):
        for row in train_rows:
            grouped.setdefault((str(level), _hierarchical_key(row, level)), []).append(
                1 - int(row["correct"])
            )
    probabilities: list[float] = []
    for row in test_rows:
        chosen = global_error
        for level in (1, 2, 3):
            values = grouped.get((str(level), _hierarchical_key(row, level)), [])
            if not values:
                continue
            n = len(values)
            group_rate = (sum(values) + 1.0) / (n + 2.0)
            weight = n / (n + float(shrinkage))
            chosen = weight * group_rate + (1.0 - weight) * global_error
            if n >= 5:
                break
        probabilities.append(_safe_normalize_probability(chosen))
    return probabilities


def _metrics(y_true: list[int], probabilities: list[float]) -> dict[str, float]:
    if not y_true:
        return {"n": 0, "logloss": math.nan, "brier": math.nan, "error_rate": math.nan}
    y = np.asarray(y_true, dtype=int)
    p = np.asarray([_safe_normalize_probability(v) for v in probabilities], dtype=float)
    return {
        "n": int(len(y)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "error_rate": float(np.mean(y)),
    }


def _fit_predict(
    train_rows: list[Any],
    test_rows: list[Any],
    model_c: float,
    class_weight: str | None = "balanced",
) -> tuple[list[float], bool, list[str]]:
    baseline = _baseline(train_rows)
    try:
        vectorizer = DictVectorizer(sparse=True)
        x_train = vectorizer.fit_transform([_features(row) for row in train_rows])
        x_test = vectorizer.transform([_features(row) for row in test_rows])
        y_train = np.asarray([1 - int(row["correct"]) for row in train_rows], dtype=int)
        if len(np.unique(y_train)) < 2:
            return [baseline] * len(test_rows), False, []
        model = LogisticRegression(
            C=float(model_c),
            class_weight=class_weight,
            max_iter=1000,
            random_state=42,
        )
        model.fit(x_train, y_train)
        p = model.predict_proba(x_test)
        class_index = {int(c): i for i, c in enumerate(model.classes_)}
        probabilities = [
            _safe_normalize_probability(float(row[class_index[1]]))
            if 1 in class_index
            else baseline
            for row in p
        ]
        names = list(vectorizer.get_feature_names_out())
        coef = model.coef_[0]
        ranked = sorted(
            zip(names, coef, strict=False),
            key=lambda x: abs(float(x[1])),
            reverse=True,
        )
        drivers = [
            f"{name}={'+' if value >= 0 else ''}{float(value):.6f}"
            for name, value in ranked[:12]
        ]
        return probabilities, True, drivers
    except (ValueError, TypeError, FloatingPointError):
        return [baseline] * len(test_rows), False, []


def prequential_evaluate(
    rows: list[Any],
    *,
    min_train_rows: int = 100,
    thresholds: tuple[float, ...] = (0.55, 0.60, 0.65, 0.70),
    block_size: int = 1,
    model_c: float = 0.5,
    max_report_cases: int = 100,
    class_weight: str | None = "balanced",
) -> dict[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (
            _parse_ts(row["settled_at_utc"]),
            int(row["experience_id"]),
        ),
    )
    if len(ordered) <= min_train_rows:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_settled_experience",
            "rows": len(ordered),
            "min_train_rows": min_train_rows,
        }

    y_meta: list[int] = []
    p_meta: list[float] = []
    p_baseline: list[float] = []
    cases: list[dict[str, Any]] = []
    model_fit_count = 0
    fallback_count = 0
    latest_drivers: list[str] = []
    p_memory: list[float] = []

    start = int(min_train_rows)
    while start > 0 and start < len(ordered) and str(ordered[start]["settled_at_utc"]) == str(ordered[start - 1]["settled_at_utc"]):
        start += 1
    while start < len(ordered):
        # Cases sharing one settlement timestamp become one evaluation batch.
        # No outcome settled at the same instant can train another case.
        stop = min(len(ordered), start + max(1, int(block_size)))
        # Never split one settlement timestamp across train/test or test/test.
        while (
            stop < len(ordered)
            and str(ordered[stop]["settled_at_utc"]) == str(ordered[stop - 1]["settled_at_utc"])
        ):
            stop += 1
        train_rows = ordered[:start]
        test_rows = ordered[start:stop]
        if not test_rows:
            break
        probabilities, fitted, drivers = _fit_predict(
            train_rows,
            test_rows,
            model_c,
            class_weight=class_weight,
        )
        p_memory.extend(_hierarchical_memory_predict(train_rows, test_rows))
        if fitted:
            model_fit_count += 1
            latest_drivers = drivers
        else:
            fallback_count += 1
        baseline = _baseline(train_rows)
        for row, p in zip(test_rows, probabilities, strict=False):
            error = 1 - int(row["correct"])
            y_meta.append(error)
            p_meta.append(p)
            p_baseline.append(baseline)
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
    meta = _metrics(y_meta, p_meta)
    baseline_metrics = _metrics(y_meta, p_baseline)
    memory_metrics = _metrics(y_meta, p_memory)
    threshold_metrics: dict[str, dict[str, float | int | None]] = {}
    total_errors = sum(y_meta)
    all_accuracy = float(sum(1 - value for value in y_meta) / len(y_meta)) if y_meta else 0.0
    for threshold in thresholds:
        keep = [idx for idx, p in enumerate(p_meta) if p < float(threshold)]
        abstain = [idx for idx, p in enumerate(p_meta) if p >= float(threshold)]
        kept_correct = sum(1 - y_meta[idx] for idx in keep)
        abstain_errors = sum(y_meta[idx] for idx in abstain)
        kept_accuracy = float(kept_correct / len(keep)) if keep else None
        threshold_metrics[f"{float(threshold):.2f}"] = {
            "threshold_error_probability": float(threshold),
            "coverage": float(len(keep) / len(y_meta)) if y_meta else 0.0,
            "kept_n": int(len(keep)),
            "kept_accuracy": kept_accuracy,
            "selective_accuracy_gain_vs_all": (
                float(kept_accuracy - all_accuracy) if kept_accuracy is not None else None
            ),
            "abstain_n": int(len(abstain)),
            "abstain_error_rate": float(abstain_errors / len(abstain)) if abstain else None,
            "abstain_error_capture_rate": (
                float(abstain_errors / total_errors) if total_errors else None
            ),
        }

    high_risk = sorted(
        cases,
        key=lambda row: (-float(row["learned_error_probability"]), row["settled_at_utc"], row["experience_id"]),
    )[:max(1, int(max_report_cases))]
    return {
        "status": "OK",
        "rows": len(ordered),
        "prequential_test_rows": len(y_meta),
        "min_train_rows": int(min_train_rows),
        "block_size": int(block_size),
        "model": {
            "type": "logistic_regression",
            "target": "prediction_error",
            "class_weight": class_weight,
            "trained_only_on_prior_settled_experiences": True,
            "model_fit_count": int(model_fit_count),
            "fallback_count": int(fallback_count),
            "latest_feature_drivers": latest_drivers,
        },
        "meta_error_probability": meta,
        "baseline_error_probability": baseline_metrics,
        "hierarchical_experience_memory": memory_metrics,
        "delta_logloss_meta_minus_baseline": float(meta["logloss"] - baseline_metrics["logloss"]),
        "delta_logloss_memory_minus_baseline": float(memory_metrics["logloss"] - baseline_metrics["logloss"]),
        "delta_brier_memory_minus_baseline": float(memory_metrics["brier"] - baseline_metrics["brier"]),
        "delta_brier_meta_minus_baseline": float(meta["brier"] - baseline_metrics["brier"]),
        "threshold_policy_candidates": threshold_metrics,
        "high_risk_cases_latest": high_risk,
    }


def build(
    db_path: Path = DB,
    *,
    config_path: Path = CONFIG,
    output_path: Path = OUT,
) -> dict[str, Any]:
    cfg = _config(config_path)
    rows, pit_scope = load_strict_primary_rows(db_path)
    rows_by_horizon: dict[str, list[Any]] = {"5m": [], "10m": []}
    for row in rows:
        rows_by_horizon[str(row["horizon"])].append(row)


    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "research_only": True,
        "production_changed": False,
        "strict_pit_scope": True,
        "pit_scope": pit_scope,
        "promotion_evidence_eligible": False,
        "experience_source": "experience_ledger",
        "learning_boundary": "each test case is scored using only prior_settled_experiences (earlier settled experiences); no current or future outcome is used",
        "config": cfg,
        "horizons": {},
    }
    for horizon, rows in rows_by_horizon.items():
        result = prequential_evaluate(
            rows,
            min_train_rows=int(cfg["min_train_rows"]),
            thresholds=tuple(float(x) for x in cfg["thresholds"]),
            block_size=1,
            model_c=float(cfg["model_c"]),
            max_report_cases=int(cfg["max_report_cases"]),
        )
        payload["horizons"][horizon] = result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    report = build()
    print(json.dumps(report, ensure_ascii=False))
