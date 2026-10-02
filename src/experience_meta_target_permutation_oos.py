"""Research-only adversarial target-permutation audit for experience meta-risk.

The base production prediction is untouched. This audit tests the error-risk
meta layer itself: fit the same prediction-error model on chronological, PIT-valid
training cases, then compare it with identical models trained on shuffled error
targets. A suspicious real-vs-null result triggers investigation; it never permits
selection or production promotion.
"""
from __future__ import annotations

import json
import math
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
VALIDATION_SIZE = 40
MIN_CLASS_COUNT = 5
MIN_FOLDS = 2
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
    """Run a multi-block chronological permutation audit of the meta error target.

    Each validation block is scored only after fitting on rows whose prediction
    and settlement timestamps are strictly before that block starts. The same
    chronological folds are replayed with multiple shuffled error targets so the
    audit can distinguish persistent signal from a single-window coincidence.
    """
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

    fold_results: list[dict[str, Any]] = []
    all_real_y: list[np.ndarray] = []
    all_real_p: list[np.ndarray] = []
    fold_null_metrics: list[dict[str, Any]] = []

    # Leave a causal training prefix before the first validation block.
    first_start = MIN_TRAIN + VALIDATION_SIZE
    for start in range(first_start, len(ordered), VALIDATION_SIZE):
        validation = ordered[start:start + VALIDATION_SIZE]
        if len(validation) < MIN_CLASS_COUNT * 2:
            break
        cutoff = _parse_ts(validation[0]["created_at_utc"])
        train = _eligible_train(ordered[:start], cutoff)
        if len(train) < MIN_TRAIN:
            continue

        train_y = np.asarray([1 - int(r["correct"]) for r in train], dtype=int)
        eval_y = np.asarray([1 - int(r["correct"]) for r in validation], dtype=int)
        real_p = _fit_predict(train, validation, train_y)
        if real_p is None:
            continue

        null_metrics: list[dict[str, float | int | None]] = []
        for seed in SEEDS:
            shuffled = np.random.default_rng(seed).permutation(train_y)
            null_p = _fit_predict(train, validation, shuffled)
            if null_p is None:
                continue
            null_metrics.append(_metrics(eval_y, null_p))
        if len(null_metrics) < 3:
            continue

        real = _metrics(eval_y, real_p)
        all_real_y.append(eval_y)
        all_real_p.append(real_p)
        fold_null_metrics.append({
            "logloss_mean": float(np.mean([float(x["logloss"]) for x in null_metrics])),
            "brier_mean": float(np.mean([float(x["brier"]) for x in null_metrics])),
            "accuracy_mean": float(np.mean([float(x["accuracy"]) for x in null_metrics])),
            "auc_mean": (
                float(np.mean([float(x["auc"]) for x in null_metrics if x["auc"] is not None]))
                if any(x["auc"] is not None for x in null_metrics)
                else None
            ),
            "permutations": int(len(null_metrics)),
        })
        fold_results.append({
            "validation_start_utc": validation[0]["created_at_utc"],
            "validation_end_utc": validation[-1]["created_at_utc"],
            "train_n": int(len(train)),
            "validation_n": int(len(validation)),
            "real": real,
            "null_summary": fold_null_metrics[-1],
            "separation": {
                "logloss_null_mean_minus_real": float(fold_null_metrics[-1]["logloss_mean"] - float(real["logloss"])),
                "brier_null_mean_minus_real": float(fold_null_metrics[-1]["brier_mean"] - float(real["brier"])),
                "accuracy_real_minus_null_mean": float(float(real["accuracy"]) - fold_null_metrics[-1]["accuracy_mean"]),
            },
            "pit": {
                "train_created_strictly_before_validation": True,
                "train_settled_strictly_before_validation": True,
                "validation_outcomes_used_only_after_model_fitted": True,
            },
        })

    if len(fold_results) < MIN_FOLDS or not all_real_y:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_valid_permutation_folds",
            "n": int(len(ordered)),
            "observed_folds": int(len(fold_results)),
            "minimum_folds": MIN_FOLDS,
            "research_only": True,
            "production_changed": False,
            "selection_allowed": False,
        }

    eval_y_all = np.concatenate(all_real_y)
    real_p_all = np.concatenate(all_real_p)
    real = _metrics(eval_y_all, real_p_all)

    null_ll = np.asarray([float(x["logloss_mean"]) for x in fold_null_metrics], dtype=float)
    null_br = np.asarray([float(x["brier_mean"]) for x in fold_null_metrics], dtype=float)
    null_acc = np.asarray([float(x["accuracy_mean"]) for x in fold_null_metrics], dtype=float)
    null_auc = np.asarray(
        [float(x["auc_mean"]) for x in fold_null_metrics if x["auc_mean"] is not None],
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
    fold_consistency = {
        "logloss_better_folds": int(sum(
            float(f["real"]["logloss"]) < float(f["null_summary"]["logloss_mean"])
            for f in fold_results
        )),
        "brier_better_folds": int(sum(
            float(f["real"]["brier"]) < float(f["null_summary"]["brier_mean"])
            for f in fold_results
        )),
        "accuracy_better_folds": int(sum(
            float(f["real"]["accuracy"]) > float(f["null_summary"]["accuracy_mean"])
            for f in fold_results
        )),
        "folds": int(len(fold_results)),
    }
    strong = (
        separation["logloss_null_mean_minus_real"] > 0.01
        and separation["brier_null_mean_minus_real"] > 0.005
        and separation["accuracy_real_minus_null_mean"] > 0.03
        and fold_consistency["logloss_better_folds"] >= max(2, int(math.ceil(len(fold_results) * 0.67)))
        and fold_consistency["brier_better_folds"] >= max(2, int(math.ceil(len(fold_results) * 0.67)))
        and fold_consistency["accuracy_better_folds"] >= max(2, int(math.ceil(len(fold_results) * 0.67)))
    )
    suspicious = (
        separation["logloss_null_mean_minus_real"] <= -0.01
        or separation["brier_null_mean_minus_real"] <= -0.01
        or separation["accuracy_real_minus_null_mean"] <= -0.02
        or fold_consistency["logloss_better_folds"] < len(fold_results) // 2
    )
    status = "SEPARATED" if strong and not suspicious else ("SUSPICIOUS" if suspicious else "NO_SEPARATION")

    return {
        "status": status,
        "risk_flag": bool(suspicious),
        "n": int(len(ordered)),
        "train_n": int(fold_results[0]["train_n"]),
        "validation_n": int(sum(int(f["validation_n"]) for f in fold_results)),
        "folds": fold_results,
        "fold_consistency": fold_consistency,
        "seeds": list(SEEDS),
        "real": real,
        "null_summary": {
            "permutations_per_fold": int(min(int(x["permutations"]) for x in fold_null_metrics)),
            "folds": int(len(fold_results)),
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
            "every_validation_fold_training_created_strictly_before_validation": True,
            "every_validation_fold_training_settled_strictly_before_validation": True,
            "validation_outcomes_used_only_after_model_fitted": True,
        },
        "research_only": True,
        "production_changed": False,
        "selection_allowed": False,
        "interpretation": "A suspicious result triggers leakage/feature review; it does not prove leakage by itself. Multi-block consistency is an evidence-quality check, not a promotion gate.",
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
            "min_folds": MIN_FOLDS,
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
