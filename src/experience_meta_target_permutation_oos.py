"""Research-only adversarial target-permutation audit for experience meta-risk.

The base production prediction is untouched. This audit tests the error-risk
meta layer itself: fit the same prediction-error model on chronological, PIT-valid
training cases, then compare it with identical models trained on shuffled error
targets. A suspicious real-vs-null result triggers investigation; it never permits
selection or production promotion.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

try:
    from experience_pit_scope import load_strict_verified_rows
except ModuleNotFoundError:
    from src.experience_pit_scope import load_strict_verified_rows

from experience_case_adaptive_controller_oos import _meta_features, _parse_ts

try:
    from db import DB, init_db
except ModuleNotFoundError:
    from src.db import DB, init_db

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "experience_meta_target_permutation_oos.json"

MIN_TRAIN = 140
VALIDATION_SIZE = 60
MIN_CLASS_COUNT = 5
SEEDS = (7, 19, 43, 71, 101, 137)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int | None]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0 - 1e-9)
    if len(y) == 0:
        return {"n": 0, "logloss": None, "brier": None, "accuracy": None, "auc": None}
    idx = np.arange(len(y))
    pred = (p >= 0.5).astype(int)
    result: dict[str, float | int | None] = {
        "n": int(len(y)),
        "logloss": float(-np.mean(np.where(y == 1, np.log(p), np.log(1.0 - p)))),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean(pred == y)),
        "auc": None,
    }
    if len(np.unique(y)) >= 2:
        result["auc"] = float(roc_auc_score(y, p))
    return result


def _fit_predict(
    train_rows: list[Any],
    eval_rows: list[Any],
    target_values: np.ndarray,
) -> np.ndarray | None:
    if len(train_rows) != len(target_values):
        raise ValueError("meta target length mismatch")
    if len(train_rows) < MIN_TRAIN or len(eval_rows) == 0:
        return None
    if np.sum(target_values == 0) < MIN_CLASS_COUNT or np.sum(target_values == 1) < MIN_CLASS_COUNT:
        return None
    try:
        vectorizer = DictVectorizer(sparse=True)
        x_train = vectorizer.fit_transform([dict(_meta_features(row)) for row in train_rows])
        x_eval = vectorizer.transform([dict(_meta_features(row)) for row in eval_rows])
        model = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
        model.fit(x_train, target_values.astype(int))
        class_index = {int(c): i for i, c in enumerate(model.classes_)}
        if 1 not in class_index:
            return None
        return np.asarray(model.predict_proba(x_eval)[:, class_index[1]], dtype=float)
    except (TypeError, ValueError, FloatingPointError):
        return None


def _eligible_train(rows: list[Any], cutoff: datetime) -> list[Any]:
    selected: list[Any] = []
    for row in rows:
        try:
            created = _parse_ts(row["created_at_utc"])
            settled = _parse_ts(row["settled_at_utc"])
        except (TypeError, ValueError):
            continue
        if created < cutoff and settled < cutoff:
            selected.append(row)
    return selected


def audit_horizon(rows: list[Any], horizon: str) -> dict[str, Any]:
    ordered = sorted(
        [r for r in rows if str(r["horizon"]) == horizon],
        key=lambda r: (_parse_ts(r["created_at_utc"]), int(r["experience_id"])),
    )
    if len(ordered) < MIN_TRAIN + VALIDATION_SIZE:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_experience_rows",
            "n": int(len(ordered)),
            "research_only": True,
            "production_changed": False,
            "selection_allowed": False,
        }

    validation = ordered[-VALIDATION_SIZE:]
    cutoff = _parse_ts(validation[0]["created_at_utc"])
    train = _eligible_train(ordered[:-VALIDATION_SIZE], cutoff)
    if len(train) < MIN_TRAIN:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_pit_valid_training_rows",
            "n": int(len(ordered)),
            "train_n": int(len(train)),
            "validation_n": int(len(validation)),
            "research_only": True,
            "production_changed": False,
            "selection_allowed": False,
        }

    train_y = np.asarray([1 - int(r["correct"]) for r in train], dtype=int)
    eval_y = np.asarray([1 - int(r["correct"]) for r in validation], dtype=int)

    real_p = _fit_predict(train, validation, train_y)
    if real_p is None:
        return {
            "status": "DEFERRED",
            "reason": "meta_model_fit_unavailable",
            "n": int(len(ordered)),
            "train_n": int(len(train)),
            "validation_n": int(len(validation)),
            "research_only": True,
            "production_changed": False,
            "selection_allowed": False,
        }

    null_metrics: list[dict[str, float | int | None]] = []
    for seed in SEEDS:
        shuffled = np.random.default_rng(seed).permutation(train_y)
        null_p = _fit_predict(train, validation, shuffled)
        if null_p is None:
            continue
        null_metrics.append(_metrics(eval_y, null_p))

    if len(null_metrics) < 3:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_valid_permutations",
            "n": int(len(ordered)),
            "train_n": int(len(train)),
            "validation_n": int(len(validation)),
            "permutations": int(len(null_metrics)),
            "research_only": True,
            "production_changed": False,
            "selection_allowed": False,
        }

    real = _metrics(eval_y, real_p)
    null_ll = np.asarray([float(x["logloss"]) for x in null_metrics], dtype=float)
    null_br = np.asarray([float(x["brier"]) for x in null_metrics], dtype=float)
    null_acc = np.asarray([float(x["accuracy"]) for x in null_metrics], dtype=float)
    null_auc = np.asarray(
        [float(x["auc"]) for x in null_metrics if x["auc"] is not None],
        dtype=float,
    )

    separation = {
        "logloss_null_mean_minus_real": float(null_ll.mean() - float(real["logloss"])),
        "brier_null_mean_minus_real": float(null_br.mean() - float(real["brier"])),
        "accuracy_real_minus_null_mean": float(float(real["accuracy"]) - null_acc.mean()),
        "auc_real_minus_null_mean": (
            float(float(real["auc"]) - null_auc.mean())
            if real["auc"] is not None and len(null_auc)
            else None
        ),
    }
    strong = (
        float(real["logloss"]) < float(np.quantile(null_ll, 0.10))
        and float(real["brier"]) < float(np.quantile(null_br, 0.10))
        and float(real["accuracy"]) > float(np.quantile(null_acc, 0.90))
    )
    suspicious = (
        float(real["logloss"]) >= float(null_ll.mean() - 0.01)
        or float(real["brier"]) >= float(null_br.mean() - 0.01)
        or float(real["accuracy"]) <= float(null_acc.mean() + 0.02)
    )
    status = "SEPARATED" if strong and not suspicious else ("SUSPICIOUS" if suspicious else "NO_SEPARATION")

    return {
        "status": status,
        "risk_flag": bool(suspicious),
        "n": int(len(ordered)),
        "train_n": int(len(train)),
        "validation_n": int(len(validation)),
        "train_cutoff_utc": cutoff.isoformat(),
        "seeds": list(SEEDS),
        "real": real,
        "null_summary": {
            "permutations": int(len(null_metrics)),
            "logloss_mean": float(null_ll.mean()),
            "logloss_q10": float(np.quantile(null_ll, 0.10)),
            "logloss_q90": float(np.quantile(null_ll, 0.90)),
            "brier_mean": float(null_br.mean()),
            "brier_q10": float(np.quantile(null_br, 0.10)),
            "brier_q90": float(np.quantile(null_br, 0.90)),
            "accuracy_mean": float(null_acc.mean()),
            "accuracy_q10": float(np.quantile(null_acc, 0.10)),
            "accuracy_q90": float(np.quantile(null_acc, 0.90)),
        },
        "separation": separation,
        "pit": {
            "train_created_strictly_before_validation": True,
            "train_settled_strictly_before_validation": True,
            "validation_outcomes_used_only_after_model_fitted": True,
        },
        "research_only": True,
        "production_changed": False,
        "selection_allowed": False,
        "interpretation": "A suspicious result triggers leakage/feature review; it does not prove leakage by itself.",
    }


def build() -> dict[str, Any]:
    rows, pit_scope = load_strict_verified_rows(DB)
    results = {h: audit_horizon(rows, h) for h in ("5m", "10m")}
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_changed": False,
        "strict_pit_scope": True,
        "research_population": "all_verified_production_sources",
        "promotion_primary_scope": "binance_futures_only",
        "pit_scope": pit_scope,
        "promotion_evidence_eligible": False,
        "description": "Adversarial target-permutation audit of the prediction-error meta layer.",
        "config": {
            "min_train": MIN_TRAIN,
            "validation_size": VALIDATION_SIZE,
            "min_class_count": MIN_CLASS_COUNT,
            "seeds": list(SEEDS),
        },
        "horizons": results,
    }


def main() -> None:
    init_db()
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
