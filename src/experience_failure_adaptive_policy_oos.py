"""Research-only policy optimizer driven by matured failure experience.

This layer does not replace the prediction model. It learns how aggressively to
respond to predicted error risk by selecting the shrink/abstain policy from
past, already-settled cases only. The policy is chosen prequentially and remains
research-only until independent OOS/holdout promotion evidence exists.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experience_case_adaptive_controller_oos import (
    ABSTAIN_THRESHOLD,
    CLASSES,
    MAX_SHRINK,
    MIN_TRAIN,
    _baseline_error,
    _case_key,
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
OUT = ROOT / "data" / "historical_research" / "experience_failure_adaptive_policy_oos.json"

POLICY_GRID = tuple(
    (float(threshold), float(max_shrink))
    for threshold in (0.70, 0.75, 0.80, 0.85, 0.90)
    for max_shrink in (0.15, 0.25, 0.35)
)
VALIDATION_SIZE = 80
MIN_VALIDATION_SUPPORT = 30
MIN_CASE_VALIDATION_SUPPORT = 30
MIN_COVERAGE = 0.80
STABILITY_EPS = 0.002
BLOCK_SIZE = 40
RISK_REFRESH = 20


@dataclass(frozen=True)
class RiskRecord:
    experience_id: int
    horizon: str
    case_key: tuple[str, ...]
    created_at_utc: datetime
    settled_at_utc: datetime
    y_index: int
    base_probability: tuple[float, float, float]
    baseline_error: float
    error_risk: float


def _eligible_prior(ordered: list[Any], current: Any) -> list[Any]:
    prediction_time = _parse_ts(current["created_at_utc"])
    eligible: list[Any] = []
    for row in ordered:
        try:
            created = _parse_ts(row["created_at_utc"])
            settled = _parse_ts(row["settled_at_utc"])
        except (TypeError, ValueError):
            continue
        if created < prediction_time and settled < prediction_time:
            eligible.append(row)
    return eligible


def _apply_policy(
    base: np.ndarray,
    error_risk: float,
    baseline_error: float,
    threshold: float,
    max_shrink: float,
) -> tuple[np.ndarray, str, float]:
    probability = np.asarray(base, dtype=float)
    probability = np.clip(probability, 1e-9, 1.0)
    probability /= probability.sum()
    if error_risk >= threshold:
        return probability, "ABSTAIN", 0.0
    excess = max(0.0, float(error_risk) - float(baseline_error))
    shrink = float(
        np.clip(
            excess / max(1.0 - float(baseline_error), 1e-6),
            0.0,
            max_shrink,
        )
    )
    if shrink <= 0.0:
        return probability, "KEEP", 0.0
    uniform = np.full(3, 1.0 / 3.0, dtype=float)
    adjusted = (1.0 - shrink) * probability + shrink * uniform
    adjusted /= adjusted.sum()
    return adjusted, "SHRINK", shrink


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1.0))))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    one_hot = np.eye(3, dtype=float)[y]
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def _fit_risk_model(
    prior: list[Any],
) -> tuple[Any, float]:
    """Fit one error-risk meta model for a bounded model lifetime."""
    baseline = _baseline_error(prior)
    if len(prior) < MIN_TRAIN:
        return None, baseline
    try:
        from experience_predictability_router_oos import _fit_meta_model

        return _fit_meta_model(prior)
    except (ImportError, TypeError, ValueError, FloatingPointError):
        return None, baseline


def _predict_risk_model(
    model_bundle: Any,
    rows: list[Any],
    default: float,
) -> np.ndarray:
    if not rows or model_bundle is None:
        return np.full(len(rows), default, dtype=float)
    try:
        from experience_predictability_router_oos import _predict_meta

        return _predict_meta(model_bundle, rows, default)
    except (ImportError, TypeError, ValueError, FloatingPointError):
        return np.full(len(rows), default, dtype=float)


def _risk_trace(
    rows: list[Any],
    horizon: str,
) -> tuple[list[RiskRecord], int, int]:
    ordered = sorted(
        [row for row in rows if str(row["horizon"]) == horizon],
        key=lambda row: (
            _parse_ts(row["created_at_utc"]),
            _parse_ts(row["settled_at_utc"]),
            int(row["experience_id"]),
        ),
    )
    records: list[RiskRecord] = []
    pit_excluded = 0
    refresh_count = 0

    for block_start in range(MIN_TRAIN, len(ordered), RISK_REFRESH):
        block_end = min(len(ordered), block_start + RISK_REFRESH)
        first_current = ordered[block_start]
        candidates = ordered[:block_start]
        prior = _eligible_prior(candidates, first_current)
        pit_excluded += max(0, len(candidates) - len(prior))
        if len(prior) < MIN_TRAIN:
            continue

        model, default = _fit_risk_model(prior)
        refresh_count += 1
        block_rows = ordered[block_start:block_end]
        model_predictions = _predict_risk_model(
            model,
            block_rows,
            default,
        )

        for offset, index in enumerate(range(block_start, block_end)):
            current = ordered[index]
            current_prior = _eligible_prior(
                ordered[:index],
                current,
            )
            if len(current_prior) < MIN_TRAIN:
                continue

            base = _probabilities(current)
            baseline = _baseline_error(current_prior)
            memory_risk = _hierarchical_prior(
                current_prior,
                current,
            )
            meta_risk = (
                float(model_predictions[offset])
                if model is not None
                else baseline
            )
            risk = _safe01(
                0.65 * meta_risk + 0.35 * memory_risk
            )

            actual = str(current["actual_direction"])
            if actual not in CLASSES:
                continue
            records.append(
                RiskRecord(
                    experience_id=int(current["experience_id"]),
                    horizon=horizon,
                    case_key=_case_key(current),
                    created_at_utc=_parse_ts(current["created_at_utc"]),
                    settled_at_utc=_parse_ts(current["settled_at_utc"]),
                    y_index=CLASSES.index(actual),
                    base_probability=tuple(float(x) for x in base),
                    baseline_error=baseline,
                    error_risk=risk,
                )
            )
    return records, pit_excluded, refresh_count


def _eligible_risk_records(records: list[RiskRecord], current: RiskRecord) -> list[RiskRecord]:
    """Return risk records whose prediction and outcome both predate current."""
    return [
        record
        for record in records
        if record.created_at_utc < current.created_at_utc
        and record.settled_at_utc < current.created_at_utc
    ]


def _metrics(records: list[RiskRecord], policy: tuple[float, float]) -> dict[str, Any]:
    if not records:
        return {
            "n": 0,
            "coverage": None,
            "abstain_rate": None,
            "accuracy": None,
            "logloss": None,
            "brier": None,
            "covered_accuracy": None,
            "covered_logloss": None,
            "covered_brier": None,
        }
    probabilities = []
    labels = []
    actions = []
    for record in records:
        adjusted, action, _ = _apply_policy(
            np.asarray(record.base_probability),
            record.error_risk,
            record.baseline_error,
            policy[0],
            policy[1],
        )
        probabilities.append(adjusted)
        labels.append(record.y_index)
        actions.append(action)
    p = np.vstack(probabilities)
    y = np.asarray(labels, dtype=int)
    action_array = np.asarray(actions)
    covered = action_array != "ABSTAIN"
    predicted = np.argmax(p, axis=1)
    return {
        "n": int(len(records)),
        "coverage": float(np.mean(covered)),
        "abstain_rate": float(np.mean(~covered)),
        "accuracy": float(np.mean(predicted == y)),
        "logloss": _logloss(y, p),
        "brier": _brier(y, p),
        "covered_accuracy": (
            float(np.mean(predicted[covered] == y[covered]))
            if np.any(covered) else None
        ),
        "covered_logloss": (
            _logloss(y[covered], p[covered])
            if np.any(covered) else None
        ),
        "covered_brier": (
            _brier(y[covered], p[covered])
            if np.any(covered) else None
        ),
    }



def _policy_stable_against_fixed(
    validation: list[RiskRecord],
    policy: tuple[float, float],
) -> bool:
    """Require the candidate to remain close to fixed-policy performance in two time blocks."""
    if len(validation) < 2 * MIN_VALIDATION_SUPPORT:
        return True
    midpoint = len(validation) // 2
    fixed = (ABSTAIN_THRESHOLD, MAX_SHRINK)
    for subset in (validation[:midpoint], validation[midpoint:]):
        candidate_metrics = _metrics(subset, policy)
        fixed_metrics = _metrics(subset, fixed)
        candidate_ll = candidate_metrics["covered_logloss"]
        fixed_ll = fixed_metrics["covered_logloss"]
        candidate_cov = candidate_metrics["coverage"] or 0.0
        fixed_cov = fixed_metrics["coverage"] or 0.0
        if (
            candidate_ll is None
            or fixed_ll is None
            or candidate_cov < MIN_COVERAGE
            or candidate_ll > fixed_ll + STABILITY_EPS
            or candidate_cov + STABILITY_EPS < fixed_cov
        ):
            return False
    return True


def _choose_policy(validation: list[RiskRecord]) -> tuple[tuple[float, float], dict[str, Any]]:
    if len(validation) < MIN_VALIDATION_SUPPORT:
        return (
            (ABSTAIN_THRESHOLD, MAX_SHRINK),
            {
                "source": "fixed_fallback",
                "support": int(len(validation)),
                "reason": "insufficient_validation_support",
            },
        )

    eligible: list[tuple[tuple[float, float], dict[str, Any]]] = []
    stability_rejections = 0
    for policy in POLICY_GRID:
        metrics = _metrics(validation, policy)
        if metrics["coverage"] is None or metrics["coverage"] < MIN_COVERAGE:
            continue
        if not _policy_stable_against_fixed(validation, policy):
            stability_rejections += 1
            continue
        eligible.append((policy, metrics))

    if not eligible:
        # The fixed policy is itself part of POLICY_GRID. Fail closed if
        # numerical/pathological inputs prevent a stable candidate from surviving.
        fixed_policy = (ABSTAIN_THRESHOLD, MAX_SHRINK)
        fixed_metrics = _metrics(validation, fixed_policy)
        eligible = [(fixed_policy, fixed_metrics)]


    eligible.sort(
        key=lambda item: (
            float("inf") if item[1]["covered_logloss"] is None else item[1]["covered_logloss"],
            float("inf") if item[1]["covered_brier"] is None else item[1]["covered_brier"],
            -float(item[1]["coverage"] or 0.0),
            item[0],
        )
    )
    best_policy, best_metrics = eligible[0]
    fixed_metrics = _metrics(validation, (ABSTAIN_THRESHOLD, MAX_SHRINK))
    delta = None
    if fixed_metrics["covered_logloss"] is not None and best_metrics["covered_logloss"] is not None:
        delta = float(best_metrics["covered_logloss"] - fixed_metrics["covered_logloss"])
    return best_policy, {
        "source": "matured_failure_history",
        "support": int(len(validation)),
        "selected": {
            "abstain_threshold": best_policy[0],
            "max_shrink": best_policy[1],
        },
        "validation": best_metrics,
        "fixed_validation": fixed_metrics,
        "covered_logloss_delta_selected_minus_fixed": delta,
        "stability_epsilon": STABILITY_EPS,
        "stability_rejections": int(stability_rejections),
    }



def _choose_case_or_global_policy(
    validation: list[RiskRecord],
    current: RiskRecord,
) -> tuple[tuple[float, float], dict[str, Any]]:
    case_validation = [record for record in validation if record.case_key == current.case_key]
    if len(case_validation) >= MIN_CASE_VALIDATION_SUPPORT:
        policy, detail = _choose_policy(case_validation)
        detail = {
            **detail,
            "source": "case_matured_failure_history",
            "case_support": int(len(case_validation)),
            "global_validation_support": int(len(validation)),
        }
        return policy, detail

    policy, detail = _choose_policy(validation)
    detail = {
        **detail,
        "source": "matured_failure_history" if detail["source"] == "matured_failure_history" else detail["source"],
        "case_support": int(len(case_validation)),
        "global_validation_support": int(len(validation)),
    }
    return policy, detail


def _metrics_with_per_row_policies(
    records: list[RiskRecord],
    policies: list[tuple[float, float]],
) -> dict[str, Any]:
    if len(records) != len(policies):
        raise ValueError("records_and_policies_length_mismatch")
    probabilities = []
    labels = []
    actions = []
    for record, policy in zip(records, policies):
        adjusted, action, _ = _apply_policy(
            np.asarray(record.base_probability),
            record.error_risk,
            record.baseline_error,
            policy[0],
            policy[1],
        )
        probabilities.append(adjusted)
        labels.append(record.y_index)
        actions.append(action)
    p = np.vstack(probabilities)
    y = np.asarray(labels, dtype=int)
    covered = np.asarray(actions) != "ABSTAIN"
    predicted = np.argmax(p, axis=1)
    return {
        "n": int(len(records)),
        "coverage": float(np.mean(covered)),
        "abstain_rate": float(1.0 - np.mean(covered)),
        "accuracy": float(np.mean(predicted == y)),
        "logloss": _logloss(y, p),
        "brier": _brier(y, p),
        "covered_accuracy": (
            float(np.mean(predicted[covered] == y[covered]))
            if np.any(covered) else None
        ),
        "covered_logloss": (
            _logloss(y[covered], p[covered])
            if np.any(covered) else None
        ),
        "covered_brier": (
            _brier(y[covered], p[covered])
            if np.any(covered) else None
        ),
    }


def _evaluate_horizon(rows: list[Any], horizon: str) -> dict[str, Any]:
    records, pit_excluded, refresh_count = _risk_trace(rows, horizon)
    if not records:
        return {
            "status": "DEFERRED",
            "reason": "no_pit_valid_risk_trace",
            "n": len([r for r in rows if str(r["horizon"]) == horizon]),
            "promotion_evidence_eligible": False,
        }

    ordered = sorted(records, key=lambda r: (r.created_at_utc, r.experience_id))
    selected_records: list[RiskRecord] = []
    fixed_records: list[RiskRecord] = []
    selected_policies: list[tuple[float, float]] = []
    policy_sources: list[str] = []
    deferred = 0
    changed = 0

    for index, current in enumerate(ordered):
        prior = _eligible_risk_records(ordered[:index], current)
        if len(prior) < MIN_VALIDATION_SUPPORT:
            deferred += 1
            continue
        validation = prior[-VALIDATION_SIZE:]
        policy, selection = _choose_case_or_global_policy(validation, current)
        selected_policies.append(policy)
        policy_sources.append(str(selection["source"]))
        if policy != (ABSTAIN_THRESHOLD, MAX_SHRINK):
            changed += 1
        selected_records.append(current)
        fixed_records.append(current)

    if not selected_records:
        return {
            "status": "DEFERRED",
            "reason": "no_policy_selection_cases",
            "n": len(ordered),
            "promotion_evidence_eligible": False,
        }

    selected_metric = _metrics_with_per_row_policies(selected_records, selected_policies)
    fixed_metric = _metrics(fixed_records, (ABSTAIN_THRESHOLD, MAX_SHRINK))
    base_metric = _metrics(fixed_records, (0.999, 0.0))

    blocks = []
    block_size = max(BLOCK_SIZE, len(selected_records) // 6)
    for start in range(0, len(selected_records), block_size):
        block_records = selected_records[start : start + block_size]
        block_policies = selected_policies[start : start + block_size]
        if len(block_records) < 10:
            continue
        adaptive_block = _metrics_with_per_row_policies(block_records, block_policies)
        fixed_block = _metrics(block_records, (ABSTAIN_THRESHOLD, MAX_SHRINK))
        blocks.append(
            {
                "start_index": int(start),
                "n": int(len(block_records)),
                "adaptive": adaptive_block,
                "fixed": fixed_block,
                "delta_adaptive_minus_fixed": {
                    "logloss": float(adaptive_block["logloss"] - fixed_block["logloss"]),
                    "brier": float(adaptive_block["brier"] - fixed_block["brier"]),
                    "covered_logloss": (
                        float(adaptive_block["covered_logloss"] - fixed_block["covered_logloss"])
                        if adaptive_block["covered_logloss"] is not None
                        and fixed_block["covered_logloss"] is not None else None
                    ),
                    "covered_accuracy": (
                        float(adaptive_block["covered_accuracy"] - fixed_block["covered_accuracy"])
                        if adaptive_block["covered_accuracy"] is not None
                        and fixed_block["covered_accuracy"] is not None else None
                    ),
                },
            }
        )

    counts: dict[str, int] = {}
    for policy in selected_policies:
        name = f"t={policy[0]:.2f},s={policy[1]:.2f}"
        counts[name] = counts.get(name, 0) + 1

    return {
        "status": "OK",
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "horizon": horizon,
        "n": len(ordered),
        "prequential_test_rows": len(selected_records),
        "learning_boundary": (
            "policy_for_current_case_uses_only_risk_records_with_created_at_utc_and "
            "settled_at_utc_strictly_before_current_prediction_time; each risk record "
            "itself uses only matured outcomes before its own prediction"
        ),
        "pit_violation_count": 0,
        "pit_excluded_candidate_count": int(pit_excluded),
        "deferred_cases": int(deferred),
        "validation_size": VALIDATION_SIZE,
        "risk_model_refresh_count": int(refresh_count),
        "risk_model_refresh_size": RISK_REFRESH,
        "min_validation_support": MIN_VALIDATION_SUPPORT,
        "min_coverage_constraint": MIN_COVERAGE,
        "policy_grid_size": len(POLICY_GRID),
        "fixed_policy": {
            "abstain_threshold": ABSTAIN_THRESHOLD,
            "max_shrink": MAX_SHRINK,
        },
        "selected_policy_counts": counts,
        "changed_policy_rate": float(changed / len(selected_policies)),
        "policy_sources": {
            "case_matured_failure_history": int(policy_sources.count("case_matured_failure_history")),
            "matured_failure_history": int(policy_sources.count("matured_failure_history")),
            "fixed_fallback": int(policy_sources.count("fixed_fallback")),
        },
        "base": base_metric,
        "fixed_policy_metrics": fixed_metric,
        "adaptive_policy_metrics": selected_metric,
        "delta_adaptive_minus_fixed": {
            "accuracy": float(selected_metric["accuracy"] - fixed_metric["accuracy"]),
            "logloss": float(selected_metric["logloss"] - fixed_metric["logloss"]),
            "brier": float(selected_metric["brier"] - fixed_metric["brier"]),
            "covered_accuracy": (
                float(selected_metric["covered_accuracy"] - fixed_metric["covered_accuracy"])
                if selected_metric["covered_accuracy"] is not None
                and fixed_metric["covered_accuracy"] is not None else None
            ),
            "covered_logloss": (
                float(selected_metric["covered_logloss"] - fixed_metric["covered_logloss"])
                if selected_metric["covered_logloss"] is not None
                and fixed_metric["covered_logloss"] is not None else None
            ),
            "covered_brier": (
                float(selected_metric["covered_brier"] - fixed_metric["covered_brier"])
                if selected_metric["covered_brier"] is not None
                and fixed_metric["covered_brier"] is not None else None
            ),
        },
        "chronological_blocks": blocks,
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
    results = {h: _evaluate_horizon(rows, h) for h in ("5m", "10m")}
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "description": (
            "Prequential failure-history policy optimizer: historical mature "
            "failure experience tunes shrinkage/abstention aggressiveness."
        ),
        "config": {
            "policy_grid": [
                {"abstain_threshold": threshold, "max_shrink": shrink}
                for threshold, shrink in POLICY_GRID
            ],
            "validation_size": VALIDATION_SIZE,
            "min_validation_support": MIN_VALIDATION_SUPPORT,
            "min_coverage": MIN_COVERAGE,
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
