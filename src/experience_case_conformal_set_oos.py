"""Research-only case-conditional conformal prediction sets.

For each prediction, calibration scores are computed only from experiences whose
prediction and outcome were both completed strictly before the current prediction.
A hierarchical case pool is used when the exact case is sparse.

The output is a prediction set {class}, {class_a, class_b}, or {DOWN, FLAT, UP}.
No production artifact or promotion evidence is modified.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experience_case_adaptive_controller_oos import CLASSES, _case_key, _parse_ts, _probabilities

try:
    from db import DB, init_db
except ModuleNotFoundError:
    from src.db import DB, init_db

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "experience_case_conformal_set_oos.json"

MIN_TRAIN = 100
MIN_CALIBRATION_SUPPORT = 20
ALPHA_GRID = (0.10, 0.20, 0.30)


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


def _aps_score(row: Any) -> float:
    p = _probabilities(row)
    actual = str(row["actual_direction"])
    idx = CLASSES.index(actual)
    order = np.argsort(-p, kind="stable")
    cumulative = 0.0
    for class_idx in order:
        cumulative += float(p[class_idx])
        if int(class_idx) == idx:
            return float(np.clip(cumulative, 0.0, 1.0))
    raise ValueError("actual_direction_not_in_class_set")


def _calibration_pool(prior: list[Any], current: Any) -> tuple[list[Any], str]:
    key = _case_key(current)
    levels = (
        (key, "exact_case"),
        (key[:4], "without_production_mode"),
        (key[:3], "horizon_regime_direction"),
        (key[:2], "horizon_regime"),
        ((key[0],), "horizon"),
    )
    for level, name in levels:
        pool = [row for row in prior if _case_key(row)[:len(level)] == level]
        if len(pool) >= MIN_CALIBRATION_SUPPORT:
            return pool, name
    return prior, "global"


def _conformal_quantile(scores: list[float], alpha: float) -> float:
    if not scores:
        return 1.0
    ordered = np.sort(np.asarray(scores, dtype=float))
    n = len(ordered)
    rank = int(math.ceil((n + 1) * (1.0 - float(alpha)))) - 1
    rank = min(max(rank, 0), n - 1)
    return float(np.clip(ordered[rank], 0.0, 1.0))


def prediction_set(probabilities: Any, threshold: float) -> list[str]:
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or np.any(p < 0) or float(p.sum()) <= 0:
        raise ValueError("invalid_probability_vector")
    p = p / p.sum()
    order = np.argsort(-p, kind="stable")
    chosen: list[str] = []
    cumulative = 0.0
    for idx in order:
        chosen.append(CLASSES[int(idx)])
        cumulative += float(p[int(idx)])
        if cumulative >= float(threshold):
            break
    return chosen


def evaluate_horizon(rows: list[Any], horizon: str, alpha: float = 0.20) -> dict[str, Any]:
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

    outcomes: list[int] = []
    set_sizes: list[int] = []
    covered: list[int] = []
    singleton_correct: list[int] = []
    singleton_n = 0
    q_values: list[float] = []
    pool_levels: list[str] = []
    skipped = 0

    for idx in range(MIN_TRAIN, len(ordered)):
        current = ordered[idx]
        prior = _eligible_prior(ordered[:idx], current)
        if len(prior) < MIN_TRAIN:
            skipped += 1
            continue

        pool, level = _calibration_pool(prior, current)
        if len(pool) < MIN_CALIBRATION_SUPPORT:
            skipped += 1
            continue

        scores = [_aps_score(row) for row in pool]
        q = _conformal_quantile(scores, alpha)
        p = _probabilities(current)
        chosen = prediction_set(p, q)
        actual = str(current["actual_direction"])

        outcomes.append(CLASSES.index(actual))
        set_sizes.append(len(chosen))
        covered.append(int(actual in chosen))
        q_values.append(q)
        pool_levels.append(level)

        if len(chosen) == 1:
            singleton_n += 1
            singleton_correct.append(int(chosen[0] == actual))

    if not outcomes:
        return {
            "status": "DEFERRED",
            "reason": "no_pit_valid_calibration_cases",
            "n": len(ordered),
            "promotion_evidence_eligible": False,
        }

    sizes = np.asarray(set_sizes, dtype=float)
    covered_arr = np.asarray(covered, dtype=int)
    singleton = np.asarray(singleton_correct, dtype=int)

    return {
        "status": "OK",
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "horizon": horizon,
        "n": len(ordered),
        "prequential_test_rows": len(outcomes),
        "learning_boundary": "only_experiences_with_created_at_utc_and_settled_at_utc_strictly_before_current_prediction_time",
        "pit_violation_count": int(skipped),
        "alpha": float(alpha),
        "target_marginal_coverage": float(1.0 - alpha),
        "coverage": float(np.mean(covered_arr)),
        "coverage_gap": float(np.mean(covered_arr) - (1.0 - alpha)),
        "mean_set_size": float(np.mean(sizes)),
        "singleton_rate": float(np.mean(sizes == 1.0)),
        "singleton_accuracy": float(np.mean(singleton)) if singleton_n else None,
        "all_three_rate": float(np.mean(sizes == 3.0)),
        "mean_conformal_threshold": float(np.mean(q_values)),
        "calibration_pool_level_counts": {
            level: int(pool_levels.count(level))
            for level in sorted(set(pool_levels))
        },
        "model": {
            "type": "APS_conformal_prediction_set",
            "trained_only_on_prior_settled_experiences": True,
            "case_conditioned_calibration": True,
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
        "description": "PIT-safe case-conditional APS prediction sets for uncertainty-aware output.",
        "alphas": [float(x) for x in ALPHA_GRID],
        "horizons": {
            h: {
                f"alpha_{alpha:.2f}": evaluate_horizon(rows, h, alpha)
                for alpha in ALPHA_GRID
            }
            for h in ("5m", "10m")
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False))
