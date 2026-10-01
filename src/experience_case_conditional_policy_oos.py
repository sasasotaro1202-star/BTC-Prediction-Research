"""Research-only case-conditional policy challenger built on matured prediction experience.

This candidate extends the case-adaptive error-risk layer with a second question:
when a case is historically difficult, is the original probability distribution
systematically wrong in a case-specific way?

For each prediction, only experiences with both created_at_utc and settled_at_utc
strictly before the prediction time are eligible. A smoothed empirical outcome
distribution is estimated hierarchically from the matching case, then blended
conservatively with the current probability distribution. The resulting policy is
evaluated prequentially against the unchanged baseline.


from experience_pit_scope import load_strict_primary_rowsThis is not a production hook and does not create promotion evidence.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experience_case_adaptive_controller_oos import (
    CLASSES,
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
OUT = ROOT / "data" / "historical_research" / "experience_case_conditional_policy_oos.json"

MIN_TRAIN = 100
MIN_CASE_SUPPORT = 8
CASE_SHRINKAGE = 20.0
MAX_CORRECTION_BLEND = 0.35
CASE_RISK_THRESHOLD = 0.55


def _eligible_prior(ordered: list[Any], current: Any) -> list[Any]:
    """Fail-closed PIT filter for one prediction time."""
    prediction_time = _parse_ts(current["created_at_utc"])
    eligible: list[Any] = []
    for candidate in ordered:
        try:
            created = _parse_ts(candidate["created_at_utc"])
            settled = _parse_ts(candidate["settled_at_utc"])
        except (TypeError, ValueError):
            continue
        if created < prediction_time and settled < prediction_time:
            eligible.append(candidate)
    return eligible


def _case_outcome_prior(
    train_rows: list[Any],
    current: Any,
    *,
    shrinkage: float = CASE_SHRINKAGE,
) -> tuple[np.ndarray, int, str]:
    """Estimate P(actual class | current case) from matured experiences.

    The hierarchy starts with the full case key
    (horizon/regime/predicted_direction/confidence_bucket/production_mode) and
    backs off one dimension at a time. Laplace smoothing avoids zero-probability
    classes, while support-weighting prevents sparse cases from dominating.
    """
    eligible = _eligible_prior(train_rows, current)
    global_counts = np.asarray(
        [sum(str(r["actual_direction"]) == cls for r in eligible) for cls in CLASSES],
        dtype=float,
    )
    global_prior = (global_counts + 1.0) / (global_counts.sum() + len(CLASSES))

    target = _case_key(current)
    levels = (
        target,
        target[:5],
        target[:4],
        target[:3],
        target[:2],
        (target[0],),
    )
    for level in levels:
        group = [r for r in eligible if _case_key(r)[: len(level)] == level]
        if not group:
            continue
        counts = np.asarray(
            [sum(str(r["actual_direction"]) == cls for r in group) for cls in CLASSES],
            dtype=float,
        )
        n = int(len(group))
        empirical = (counts + 1.0) / (counts.sum() + len(CLASSES))
        weight = n / (n + float(shrinkage))
        posterior = weight * empirical + (1.0 - weight) * global_prior
        # The exact full case is useful only after the configured support floor.
        # Broader levels are allowed to rescue sparse cases.
        if n >= MIN_CASE_SUPPORT or level == (target[0],):
            return posterior / posterior.sum(), n, "/".join(level)

    return global_prior / global_prior.sum(), 0, "global"


def _case_correction(
    base: np.ndarray,
    case_prior: np.ndarray,
    *,
    support: int,
    error_risk: float,
    baseline_error: float,
) -> tuple[np.ndarray, float, str]:
    """Conservatively blend toward the case-specific empirical target distribution."""
    p = np.asarray(base, dtype=float)
    p = np.clip(p, 1e-9, 1.0)
    p /= p.sum()

    if support < MIN_CASE_SUPPORT or error_risk < CASE_RISK_THRESHOLD:
        return p, 0.0, "KEEP"

    support_weight = support / (support + CASE_SHRINKAGE)
    risk_weight = float(
        np.clip(
            (error_risk - max(CASE_RISK_THRESHOLD, baseline_error))
            / max(1.0 - max(CASE_RISK_THRESHOLD, baseline_error), 1e-6),
            0.0,
            1.0,
        )
    )
    blend = float(
        np.clip(MAX_CORRECTION_BLEND * support_weight * risk_weight, 0.0, MAX_CORRECTION_BLEND)
    )
    if blend <= 0.0:
        return p, 0.0, "KEEP"

    adjusted = (1.0 - blend) * p + blend * np.asarray(case_prior, dtype=float)
    adjusted = np.clip(adjusted, 1e-9, 1.0)
    adjusted /= adjusted.sum()
    return adjusted, blend, "CASE_CORRECTION"


def _logloss(y_idx: np.ndarray, p: np.ndarray) -> float:
    if len(y_idx) == 0:
        return float("nan")
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y_idx)), y_idx], 1e-9, 1.0))))


def _brier(y_idx: np.ndarray, p: np.ndarray) -> float:
    if len(y_idx) == 0:
        return float("nan")
    one = np.eye(3, dtype=float)[y_idx]
    return float(np.mean(np.sum((p - one) ** 2, axis=1)))


def _ece(y_idx: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    if len(y_idx) == 0:
        return float("nan")
    conf = np.max(p, axis=1)
    pred = np.argmax(p, axis=1)
    correct = (pred == y_idx).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        mask = (conf >= edges[i]) & (conf <= edges[i + 1] if i == bins - 1 else conf < edges[i + 1])
        if not np.any(mask):
            continue
        total += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return total


def _metrics(y_idx: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(np.mean(np.argmax(p, axis=1) == y_idx)),
        "logloss": _logloss(y_idx, p),
        "brier": _brier(y_idx, p),
        "ece": _ece(y_idx, p),
    }


def evaluate_horizon(rows: list[Any], horizon: str) -> dict[str, Any]:
    ordered = sorted(
        [r for r in rows if str(r["horizon"]) == horizon],
        key=lambda r: (
            _parse_ts(r["created_at_utc"]),
            _parse_ts(r["settled_at_utc"]),
            int(r["experience_id"]),
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
    base_probs: list[np.ndarray] = []
    selected_probs: list[np.ndarray] = []
    actions: list[str] = []
    supports: list[int] = []
    blends: list[float] = []
    risk_values: list[float] = []
    case_source_levels: list[str] = []
    deferred_cases = 0
    pit_excluded_candidate_count = 0

    for idx in range(MIN_TRAIN, len(ordered)):
        current = ordered[idx]
        prior_candidates = ordered[:idx]
        prior = _eligible_prior(prior_candidates, current)
        pit_excluded_candidate_count += max(0, len(prior_candidates) - len(prior))
        if len(prior) < MIN_TRAIN:
            deferred_cases += 1
            continue

        base = _probabilities(current)
        baseline = _baseline_error(prior)
        model_risk, _ = _fit_meta_probability(prior, current)
        memory_risk = _hierarchical_prior(prior, current)
        risk = _safe01(0.65 * model_risk + 0.35 * memory_risk)

        case_prior, support, source = _case_outcome_prior(prior, current)
        case_source_levels.append(source)
        corrected, blend, correction_action = _case_correction(
            base,
            case_prior,
            support=support,
            error_risk=risk,
            baseline_error=baseline,
        )

        # At extreme risk keep the existing controller's abstain rule. The
        # case-correction candidate never overrides an explicit abstention.
        risk_adjusted, base_action, _ = _adjust_probabilities(base, risk, baseline)
        if base_action == "ABSTAIN":
            selected = risk_adjusted
            action = "ABSTAIN"
            blend = 0.0
        elif correction_action == "CASE_CORRECTION":
            selected = corrected
            action = correction_action
        else:
            selected = risk_adjusted
            action = base_action

        y_values.append(CLASSES.index(str(current["actual_direction"])))
        base_probs.append(base)
        selected_probs.append(selected)
        actions.append(action)
        supports.append(support)
        blends.append(blend)
        risk_values.append(risk)

    if not y_values:
        return {
            "status": "DEFERRED",
            "reason": "no_pit_valid_prequential_cases",
            "n": len(ordered),
            "promotion_evidence_eligible": False,
        }

    y = np.asarray(y_values, dtype=int)
    base = np.vstack(base_probs)
    selected = np.vstack(selected_probs)
    actions_arr = np.asarray(actions)

    covered = actions_arr != "ABSTAIN"
    selected_metrics = _metrics(y[covered], selected[covered]) if np.any(covered) else {
        "accuracy": float("nan"),
        "logloss": float("nan"),
        "brier": float("nan"),
        "ece": float("nan"),
    }
    base_metrics_all = _metrics(y, base)
    selected_metrics_all = _metrics(y, selected)

    action_groups: dict[str, dict[str, Any]] = {}
    for action in ("KEEP", "CASE_CORRECTION", "SHRINK", "ABSTAIN"):
        mask = actions_arr == action
        if not np.any(mask):
            continue
        group = {
            "n": int(mask.sum()),
            "rate": float(mask.mean()),
            "base_accuracy": float(np.mean(np.argmax(base[mask], axis=1) == y[mask])),
            "selected_accuracy": float(np.mean(np.argmax(selected[mask], axis=1) == y[mask])),
            "base_logloss": _logloss(y[mask], base[mask]),
            "selected_logloss": _logloss(y[mask], selected[mask]),
            "mean_support": float(np.mean(np.asarray(supports)[mask])),
            "mean_blend": float(np.mean(np.asarray(blends)[mask])),
        }
        action_groups[action] = group

    correction_mask = actions_arr == "CASE_CORRECTION"
    return {
        "status": "OK",
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "horizon": horizon,
        "n": len(ordered),
        "prequential_test_rows": int(len(y)),
        "learning_boundary": "only_experiences_with_created_at_utc_and_settled_at_utc_strictly_before_current_prediction_time",
        "pit_violation_count": 0,
        "pit_excluded_candidate_count": int(pit_excluded_candidate_count),
        "deferred_cases": int(deferred_cases),
        "policy": {
            "case_support_floor": MIN_CASE_SUPPORT,
            "case_shrinkage": CASE_SHRINKAGE,
            "max_correction_blend": MAX_CORRECTION_BLEND,
            "case_risk_threshold": CASE_RISK_THRESHOLD,
            "explicit_abstain_threshold": 0.80,
            "risk_fusion": "0.65_prequential_meta_error + 0.35_hierarchical_case_memory",
            "candidate_actions": ("KEEP", "CASE_CORRECTION", "SHRINK", "ABSTAIN"),
        },
        "baseline": base_metrics_all,
        "selected_all_cases": selected_metrics_all,
        "selected_covered_only": {
            **selected_metrics,
            "coverage": float(np.mean(covered)),
            "abstain_rate": float(np.mean(~covered)),
        },
        "delta_selected_minus_baseline": {
            metric: float(selected_metrics_all[metric] - base_metrics_all[metric])
            for metric in ("accuracy", "logloss", "brier", "ece")
        },
        "action_groups": action_groups,
        "case_correction": {
            "rate": float(np.mean(correction_mask)),
            "n": int(correction_mask.sum()),
            "mean_support": float(np.mean(np.asarray(supports)[correction_mask])) if np.any(correction_mask) else 0.0,
            "mean_blend": float(np.mean(np.asarray(blends)[correction_mask])) if np.any(correction_mask) else 0.0,
        },
        "mean_predicted_error_risk": float(np.mean(risk_values)),
        "mean_case_support": float(np.mean(supports)),
        "case_source_levels": {
            source: int(case_source_levels.count(source))
            for source in sorted(set(case_source_levels))
        },
    }


def load_experience_rows_with_pit_scope() -> tuple[list[Any], dict[str, Any]]:
    return load_strict_primary_rows(DB)


def load_experience_rows() -> list[Any]:
    rows, _ = load_experience_rows_with_pit_scope()
    return rows


def build() -> dict[str, Any]:
    rows, pit_scope = load_experience_rows_with_pit_scope()
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_changed": False,
        "strict_pit_scope": True,
        "pit_scope": pit_scope,
        "promotion_evidence_eligible": False,
        "description": (
            "Prequential case-conditional probability correction combined with "
            "case-risk selective control; research-only candidate."
        ),
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
