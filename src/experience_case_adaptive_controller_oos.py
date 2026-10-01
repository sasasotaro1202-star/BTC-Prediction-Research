"""Research-only case-adaptive controller driven by matured prediction experience.

The controller joins two sources of intelligence without allowing future outcomes
to flow backwards:

1. the current prediction snapshot/probabilities;
2. only experiences whose outcomes matured before the current prediction time.

A prequential meta-model estimates the probability that the current prediction will
be wrong. That risk then selects a conservative action:
  KEEP          - retain the original calibrated distribution
  SHRINK       - move probability mass toward the uniform distribution
  ABSTAIN      - emit no directional prediction

The controller is horizon-specific and case-specific. It is deliberately research
only; production artifacts are never modified. The module is intended to become a
promotion candidate only after independent OOS/holdout validation.
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
from sklearn.metrics import roc_auc_score

try:
    from db import DB, init_db
except ModuleNotFoundError:
    from src.db import DB, init_db

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "experience_case_adaptive_controller_oos.json"

CLASSES = ("DOWN", "FLAT", "UP")
MIN_TRAIN = 100
MIN_CASE_SUPPORT = 5
MAX_SHRINK = 0.35
ABSTAIN_THRESHOLD = 0.80
MODEL_C = 0.5


def _parse_ts(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp_must_include_timezone")
    return parsed.astimezone(timezone.utc)


def _safe01(value: Any, default: float = 0.5) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(x):
        return float(default)
    return float(np.clip(x, 0.0, 1.0))


def _probabilities(row: Any) -> np.ndarray:
    raw = json.loads(str(row["probability_json"]))
    p = np.asarray([float(raw[c]) for c in CLASSES], dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("invalid_probability_json")
    total = float(p.sum())
    if total <= 0 or not math.isfinite(total):
        raise ValueError("invalid_probability_sum")
    p /= total
    return p


def _flag_values(value: Any, prefix: str) -> list[str]:
    if value in (None, "", "null", "[]"):
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return [f"{prefix}:invalid_json"]
    if isinstance(parsed, list):
        return [f"{prefix}:{item}" for item in parsed if item not in (None, "")]
    return [f"{prefix}:present"]


def _information_state(row: Any) -> str:
    """Compact causal information-quality state for case memory."""
    warning_tokens = _flag_values(row["warning_flags"], "warning")
    quality_tokens = _flag_values(row["data_quality_flags"], "quality")
    tokens = warning_tokens + quality_tokens
    if not tokens:
        return "CLEAN"
    degraded_terms = (
        "partial", "missing", "coverage", "fallback",
        "invalid", "reduced", "stale", "degraded",
    )
    lowered = [str(token).lower() for token in tokens]
    if any(term in token for token in lowered for term in degraded_terms):
        return "DEGRADED"
    return "WARNED"


def _meta_features(row: Any) -> dict[str, float | str]:
    created = _parse_ts(row["created_at_utc"])
    hour = created.hour + created.minute / 60.0
    p = _probabilities(row)
    entropy = float(-np.sum(np.clip(p, 1e-12, 1.0) * np.log(np.clip(p, 1e-12, 1.0))) / math.log(3.0))
    ordered = np.sort(p)
    confidence = float(ordered[-1])
    margin = float(ordered[-1] - ordered[-2])
    values: dict[str, float | str] = {
        "confidence": confidence,
        "entropy": entropy,
        "margin": margin,
        "hour_sin": math.sin(2.0 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2.0 * math.pi * hour / 24.0),
        "regime": str(row["regime"] or "UNKNOWN"),
        "predicted_direction": str(row["predicted_direction"]),
        "production_mode": str(row["production_mode"] or "UNKNOWN"),
        "horizon": str(row["horizon"]),
        "warning_count": float(len(_flag_values(row["warning_flags"], "warning"))),
        "quality_flag_count": float(len(_flag_values(row["data_quality_flags"], "quality"))),
        "information_state": _information_state(row),
    }
    for token in _flag_values(row["warning_flags"], "warning"):
        values[f"flag:{token}"] = 1.0
    for token in _flag_values(row["data_quality_flags"], "quality"):
        values[f"flag:{token}"] = 1.0
    return values


def _baseline_error(train_rows: list[Any]) -> float:
    if not train_rows:
        return 0.5
    errors = sum(1 - int(row["correct"]) for row in train_rows)
    return _safe01((errors + 1.0) / (len(train_rows) + 2.0))


def _case_key(row: Any) -> tuple[str, str, str, str, str, str]:
    """Return the full case identity used by hierarchical experience memory."""
    p = _probabilities(row)
    confidence = float(np.max(p))
    if confidence < 0.40:
        bucket = "0.33-0.40"
    elif confidence < 0.50:
        bucket = "0.40-0.50"
    elif confidence < 0.60:
        bucket = "0.50-0.60"
    elif confidence < 0.70:
        bucket = "0.60-0.70"
    else:
        bucket = "0.70+"
    return (
        str(row["horizon"]),
        str(row["regime"] or "UNKNOWN"),
        str(row["predicted_direction"]),
        bucket,
        str(row["production_mode"] or "UNKNOWN"),
        _information_state(row),
    )


def _hierarchical_prior(train_rows: list[Any], row: Any, shrinkage: float = 20.0) -> float:
    """Case-specific error probability from experiences matured before this prediction."""
    prediction_time = _parse_ts(row["created_at_utc"])
    eligible = []
    for candidate in train_rows:
        try:
            created = _parse_ts(candidate["created_at_utc"])
            settled = _parse_ts(candidate["settled_at_utc"])
        except (TypeError, ValueError):
            continue
        if created < prediction_time and settled < prediction_time:
            eligible.append(candidate)
    global_error = _baseline_error(eligible)
    target = _case_key(row)
    # Full case first; progressively relax only when support is sparse.
    # production_mode remains part of the full case identity.
    levels = (
        target,
        target[:5],
        target[:4],
        target[:3],
        target[:2],
        (target[0],),
    )
    for level_key in levels:
        group = [r for r in eligible if _case_key(r)[:len(level_key)] == level_key]
        if not group:
            continue
        n = len(group)
        rate = (sum(1 - int(r["correct"]) for r in group) + 1.0) / (n + 2.0)
        weight = n / (n + float(shrinkage))
        estimate = weight * rate + (1.0 - weight) * global_error
        if n >= MIN_CASE_SUPPORT or level_key == (target[0],):
            return _safe01(estimate)
    return global_error


def _fit_meta_probability(train_rows: list[Any], test_row: Any) -> tuple[float, bool]:
    baseline = _baseline_error(train_rows)
    if len(train_rows) < MIN_TRAIN:
        return baseline, False
    try:
        x_train_dict = [_meta_features(row) for row in train_rows]
        x_test_dict = [_meta_features(test_row)]
        vectorizer = DictVectorizer(sparse=True)
        x_train = vectorizer.fit_transform(x_train_dict)
        x_test = vectorizer.transform(x_test_dict)
        y = np.asarray([1 - int(row["correct"]) for row in train_rows], dtype=int)
        if len(np.unique(y)) < 2:
            return baseline, False
        model = LogisticRegression(
            C=MODEL_C,
            class_weight=None,
            max_iter=1000,
            random_state=42,
        )
        model.fit(x_train, y)
        class_index = {int(cls): idx for idx, cls in enumerate(model.classes_)}
        if 1 not in class_index:
            return baseline, False
        return _safe01(float(model.predict_proba(x_test)[0, class_index[1]])), True
    except (ValueError, TypeError, FloatingPointError):
        return baseline, False


def _adjust_probabilities(base: np.ndarray, error_probability: float, baseline_error: float) -> tuple[np.ndarray, str, float]:
    p = np.asarray(base, dtype=float)
    p = np.clip(p, 1e-9, 1.0)
    p /= p.sum()
    if error_probability >= ABSTAIN_THRESHOLD:
        return p, "ABSTAIN", 1.0

    excess = max(0.0, float(error_probability) - float(baseline_error))
    shrink = float(np.clip(excess / max(1.0 - float(baseline_error), 1e-6), 0.0, MAX_SHRINK))
    if shrink <= 0.0:
        return p, "KEEP", 0.0

    uniform = np.full(3, 1.0 / 3.0, dtype=float)
    adjusted = (1.0 - shrink) * p + shrink * uniform
    adjusted /= adjusted.sum()
    return adjusted, "SHRINK", shrink


def _logloss(y_idx: np.ndarray, p: np.ndarray) -> float:
    if len(y_idx) == 0:
        return float("nan")
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y_idx)), y_idx], 1e-9, 1.0))))


def _binary_logloss(y_true: np.ndarray, p: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0 - 1e-9)
    y_true = np.asarray(y_true, dtype=int)
    return float(-np.mean(np.where(y_true == 1, np.log(p), np.log(1.0 - p))))


def _brier(y_idx: np.ndarray, p: np.ndarray) -> float:
    if len(y_idx) == 0:
        return float("nan")
    one = np.eye(3, dtype=float)[y_idx]
    return float(np.mean(np.sum((p - one) ** 2, axis=1)))


def evaluate_horizon(rows: list[Any], horizon: str) -> dict[str, Any]:
    ordered = sorted(
        [r for r in rows if str(r["horizon"]) == horizon],
        key=lambda r: (_parse_ts(r["created_at_utc"]), _parse_ts(r["settled_at_utc"]), int(r["experience_id"])),
    )
    if len(ordered) <= MIN_TRAIN:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_experience_rows",
            "n": len(ordered),
            "promotion_evidence_eligible": False,
        }

    ys: list[int] = []
    base_probs: list[np.ndarray] = []
    adjusted_probs: list[np.ndarray] = []
    actions: list[str] = []
    risk_values: list[float] = []
    deferred_cases = 0
    pit_excluded_candidate_count = 0
    model_fit_count = 0
    case_prior_values: list[float] = []
    case_keys: list[tuple[str, str, str, str, str]] = []

    for index in range(MIN_TRAIN, len(ordered)):
        current = ordered[index]
        prediction_time = _parse_ts(current["created_at_utc"])

        # Critical PIT rule: a previous experience is usable only when its
        # outcome was already matured before this prediction happened.
        prior_candidates = ordered[:index]
        prior = []
        for candidate in prior_candidates:
            try:
                settled = _parse_ts(candidate["settled_at_utc"])
            except (TypeError, ValueError):
                continue
            if settled < prediction_time:
                prior.append(candidate)

        pit_excluded_candidate_count += max(0, len(prior_candidates) - len(prior))
        if len(prior) < MIN_TRAIN:
            deferred_cases += 1
            continue

        base = _probabilities(current)
        baseline = _baseline_error(prior)
        model_risk, fitted = _fit_meta_probability(prior, current)
        if fitted:
            model_fit_count += 1
        memory_risk = _hierarchical_prior(prior, current)

        # Conservative fusion: the learned model and empirical case memory
        # jointly inform risk. Memory has an explicit lower-variance role.
        risk = _safe01(0.65 * model_risk + 0.35 * memory_risk)
        adjusted, action, shrink = _adjust_probabilities(base, risk, baseline)

        target_index = CLASSES.index(str(current["actual_direction"]))
        ys.append(target_index)
        base_probs.append(base)
        adjusted_probs.append(adjusted)
        actions.append(action)
        risk_values.append(risk)
        case_prior_values.append(memory_risk)
        case_keys.append(_case_key(current))

    if not ys:
        return {
            "status": "DEFERRED",
            "reason": "no_pit_valid_prequential_cases",
            "n": len(ordered),
            "promotion_evidence_eligible": False,
        }

    y = np.asarray(ys, dtype=int)
    base = np.vstack(base_probs)
    adjusted = np.vstack(adjusted_probs)
    risks = np.asarray(risk_values, dtype=float)
    base_pred = np.argmax(base, axis=1)
    error_labels = (base_pred != y).astype(int)
    base_acc = float(np.mean(np.argmax(base, axis=1) == y))
    adjusted_acc = float(np.mean(np.argmax(adjusted, axis=1) == y))
    keep_mask = np.asarray(actions) != "ABSTAIN"
    coverage = float(np.mean(keep_mask)) if len(keep_mask) else 0.0
    selective_acc = (
        float(np.mean(np.argmax(adjusted[keep_mask], axis=1) == y[keep_mask]))
        if np.any(keep_mask) else None
    )

    predictability_auc = (
        float(roc_auc_score(error_labels, risks))
        if len(np.unique(error_labels)) == 2
        else None
    )
    risk_bins: list[dict[str, Any]] = []
    for lo in np.linspace(0.0, 0.9, 10):
        hi = float(min(1.0, lo + 0.1))
        mask = (risks >= lo) & ((risks < hi) if hi < 1.0 else (risks <= hi))
        if np.any(mask):
            observed = float(np.mean(error_labels[mask]))
            predicted = float(np.mean(risks[mask]))
            risk_bins.append({
                "lower": float(lo),
                "upper": float(hi),
                "n": int(mask.sum()),
                "mean_predicted_risk": predicted,
                "observed_error_rate": observed,
                "absolute_calibration_gap": float(abs(predicted - observed)),
            })

    case_group_metrics: dict[str, dict[str, Any]] = {}
    actions_arr = np.asarray(actions)
    adjusted_pred = np.argmax(adjusted, axis=1)
    for key in sorted(set(case_keys)):
        mask = np.asarray([k == key for k in case_keys], dtype=bool)
        if int(mask.sum()) < MIN_CASE_SUPPORT:
            continue
        key_text = "|".join(key)
        case_group_metrics[key_text] = {
            "n": int(mask.sum()),
            "base_accuracy": float(np.mean(base_pred[mask] == y[mask])),
            "adjusted_accuracy": float(np.mean(adjusted_pred[mask] == y[mask])),
            "base_logloss": _logloss(y[mask], base[mask]),
            "adjusted_logloss": _logloss(y[mask], adjusted[mask]),
            "mean_predicted_error_risk": float(np.mean(risks[mask])),
            "observed_base_error_rate": float(np.mean(error_labels[mask])),
            "abstain_rate": float(np.mean(actions_arr[mask] == "ABSTAIN")),
            "source_case": {
                "horizon": key[0],
                "regime": key[1],
                "predicted_direction": key[2],
                "confidence_bucket": key[3],
                "production_mode": key[4],
            },
        }

    groups: dict[str, dict[str, Any]] = {}
    for action in ("KEEP", "SHRINK", "ABSTAIN"):
        mask = np.asarray(actions) == action
        if np.any(mask):
            groups[action] = {
                "n": int(mask.sum()),
                "rate": float(mask.mean()),
                "base_accuracy": float(np.mean(np.argmax(base[mask], axis=1) == y[mask])),
                "adjusted_accuracy": float(np.mean(np.argmax(adjusted[mask], axis=1) == y[mask])),
                "base_logloss": _logloss(y[mask], base[mask]),
                "adjusted_logloss": _logloss(y[mask], adjusted[mask]),
            }

    return {
        "status": "OK",
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "horizon": horizon,
        "n": len(ordered),
        "prequential_test_rows": int(len(y)),
        "learning_boundary": "only_experiences_with_settled_at_strictly_before_current_prediction_time",
        "pit_violation_count": 0,
        "pit_excluded_candidate_count": int(pit_excluded_candidate_count),
        "deferred_cases": int(deferred_cases),
        "meta_model_fits": int(model_fit_count),
        "risk_fusion": "0.65_prequential_meta_error + 0.35_hierarchical_case_memory",
        "policy": {
            "abstain_threshold": ABSTAIN_THRESHOLD,
            "max_shrink": MAX_SHRINK,
            "actions": ("KEEP", "SHRINK", "ABSTAIN"),
        },
        "all_cases": {
            "accuracy": adjusted_acc,
            "logloss": _logloss(y, adjusted),
            "brier": _brier(y, adjusted),
        },
        "baseline": {
            "accuracy": base_acc,
            "logloss": _logloss(y, base),
            "brier": _brier(y, base),
        },
        "delta_adjusted_minus_baseline": {
            "accuracy": adjusted_acc - base_acc,
            "logloss": _logloss(y, adjusted) - _logloss(y, base),
            "brier": _brier(y, adjusted) - _brier(y, base),
        },
        "selective": {
            "coverage": coverage,
            "abstain_rate": 1.0 - coverage,
            "accuracy": selective_acc,
            "risk_on_covered": (
                float(np.mean(np.asarray(y)[keep_mask] != np.argmax(base[keep_mask], axis=1)))
                if np.any(keep_mask) else None
            ),
        },
        "action_groups": groups,
        "predictability": {
            "risk_target": "base_prediction_error",
            "logloss": _binary_logloss(error_labels, risks),
            "brier": float(np.mean((risks - error_labels) ** 2)),
            "auc": predictability_auc,
            "mean_absolute_calibration_gap": (
                float(sum(x["n"] * x["absolute_calibration_gap"] for x in risk_bins) / len(error_labels))
                if risk_bins and len(error_labels) else None
            ),
            "risk_bins": risk_bins,
        },
        "mean_predicted_error_risk": float(np.mean(risk_values)),
        "mean_case_memory_error_risk": float(np.mean(case_prior_values)),
        "case_group_metrics": case_group_metrics,
        "case_group_count_with_support_floor": int(len(case_group_metrics)),
        "case_dimensions": ("horizon", "regime", "predicted_direction", "confidence_bucket", "production_mode", "information_state"),
    }


def load_experience_rows() -> list[Any]:
    init_db()
    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """SELECT * FROM experience_ledger
               WHERE actual_direction IN ('DOWN','FLAT','UP')
                 AND settled_at_utc IS NOT NULL
               ORDER BY settled_at_utc, experience_id"""
        ).fetchall()
    return list(rows)


def build() -> dict[str, Any]:
    rows = load_experience_rows()
    results = {
        h: evaluate_horizon(rows, h)
        for h in ("5m", "10m")
    }
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "description": (
            "Case-adaptive controller combining prequential past-prediction "
            "error learning with hierarchical matured-experience memory."
        ),
        "horizons": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, ensure_ascii=False))
