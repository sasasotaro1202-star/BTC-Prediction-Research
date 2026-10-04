"""Deterministic GitHub-side research router for BTC-Prediction-Research.

This module does not fetch data, mutate production state, or select a model.
It reads already-published evidence and returns one bounded next-best research
workflow for the Continuous Supervisor to dispatch. Missing/invalid evidence
fails closed to the read-only readiness audit.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HORIZONS = ("5m", "10m")
ALLOWED = {
    "btc_research_readiness.yml": 3600,
    "btc_autonomous_data_frontier.yml": 900,
    "btc_adaptive_calibration_replay.yml": 28800,
    "btc_experience_policy_oos.yml": 21600,
    "btc_rich_production_challenger.yml": 86400,
    "btc_ultimate_final_v13_e2e.yml": 86400,
}

DEFAULT = {
    "workflow": "btc_research_readiness.yml",
    "threshold_seconds": 3600,
    "reason": "readiness_evidence_missing_or_invalid",
}


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def _decision(
    workflow: str,
    reason: str,
    priority: int,
    evidence_state: str,
    signals: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "workflow": workflow,
        "threshold_seconds": ALLOWED[workflow],
        "reason": reason,
        "production_impact": False,
        "priority": priority,
        "evidence_state": evidence_state,
        "signals": list(signals or []),
    }


def _calibration_waiting(root: Path) -> list[str]:
    signals: list[str] = []
    for horizon in HORIZONS:
        c = _load(root / "models" / f"{horizon}.calibration.json")
        if c is None:
            signals.append(f"{horizon}:calibration_artifact_missing")
            continue
        try:
            n = int(c.get("n_settled", 0))
        except (TypeError, ValueError):
            n = 0
        if n < 400:
            signals.append(f"{horizon}:calibration_n={n}<400")
        if c.get("fit_logloss") is None:
            signals.append(f"{horizon}:fit_logloss_missing")
        if c.get("holdout_logloss") is None:
            signals.append(f"{horizon}:holdout_logloss_missing")
    return signals


def _frontier_candidates(root: Path) -> int:
    frontier = _load(root / "data" / "historical_research" / "data_frontier.json")
    if not isinstance(frontier, dict):
        return 0
    candidates = frontier.get("candidates")
    if not isinstance(candidates, dict):
        return 0
    count = 0
    for item in candidates.values():
        if not isinstance(item, dict):
            continue
        lifecycle = item.get("lifecycle")
        if isinstance(lifecycle, dict) and lifecycle.get("research_selection_eligible") is True:
            count += 1
    return count


def _promotion_robustness_blocked(root: Path) -> tuple[bool, list[str]]:
    gate = _load(root / "data" / "historical_research" / "promotion_gate.json")
    if gate is None:
        return False, []

    reasons = str(gate.get("reason", "")).lower()
    hold = gate.get("production_safety_gate") == "HOLD" and gate.get("promotion_allowed") is False
    blockers = [
        token
        for token in ("robustness", "holdout")
        if token in reasons
    ]
    if not hold or not blockers:
        return False, []
    return True, [f"promotion_gate:{token}_blocked" for token in blockers]


def _high_confidence_overreach(root: Path) -> list[str]:
    """Find a large, actionable post-outcome confidence/accuracy gap.

    This is a research trigger only. It cannot change production confidence,
    routing, abstention, or calibration by itself.
    """
    experience = _load(root / "data" / "experience" / "experience_summary.json")
    if experience is None:
        return []
    signals: list[str] = []
    horizons = experience.get("horizons")
    if not isinstance(horizons, dict):
        return signals

    for horizon in HORIZONS:
        buckets = (
            horizons.get(horizon, {})
            if isinstance(horizons.get(horizon, {}), dict)
            else {}
        ).get("cases", {})
        if not isinstance(buckets, dict):
            continue
        buckets = buckets.get("confidence_bucket")
        if not isinstance(buckets, dict):
            continue
        high = buckets.get("0.70+")
        if not isinstance(high, dict):
            continue
        try:
            n = int(high.get("n", 0))
            accuracy = float(high["accuracy"])
            confidence = float(high["avg_confidence"])
        except (KeyError, TypeError, ValueError):
            continue
        gap = confidence - accuracy
        if n >= 50 and gap >= 0.15:
            signals.append(
                f"{horizon}:high_confidence_gap={gap:.3f};n={n}"
            )
    return signals


def choose(root: Path) -> dict[str, Any]:
    evidence = root / "data" / "historical_research"

    pit = _load(evidence / "pit_oos_audit.json")
    health = _load(evidence / "research_health.json")

    # research_readiness.json is a workflow-local/generated surface and is not
    # required to be committed to main. Durable PIT/health evidence is authoritative.
    if pit is None or health is None:
        return {
            **DEFAULT,
            "production_impact": False,
            "priority": 100,
            "evidence_state": "MISSING_OR_INVALID",
            "signals": ["durable_pit_or_health_artifact_missing"],
        }

    if health.get("ok") is not True:
        return _decision(
            "btc_research_readiness.yml",
            "research_health_not_pass",
            99,
            "DATA_HEALTH_BLOCKED",
            ["research_health_ok_not_true"],
        )

    if pit.get("ok") is not True or pit.get("pit_verified") is not True:
        return _decision(
            "btc_research_readiness.yml",
            "strict_pit_evidence_not_verified",
            98,
            "PIT_NOT_VERIFIED",
            ["pit_ok_or_verified_not_true"],
        )

    horizon_gate = pit.get("primary_horizon_gate")
    if not isinstance(horizon_gate, dict) or set(horizon_gate) != set(HORIZONS):
        return _decision(
            "btc_research_readiness.yml",
            "per_horizon_pit_gate_missing",
            97,
            "PIT_GATE_INCOMPLETE",
            ["5m_and_10m_horizon_gate_required"],
        )

    bad_horizons: list[str] = []
    for horizon in HORIZONS:
        item = horizon_gate.get(horizon)
        if not isinstance(item, dict):
            bad_horizons.append(f"{horizon}:gate_missing")
            continue
        try:
            strict = int(item.get("strict_primary_settled", 0))
            minimum = int(item.get("minimum", 0))
        except (TypeError, ValueError):
            bad_horizons.append(f"{horizon}:invalid_gate_counts")
            continue
        if item.get("ready") is not True or minimum < 300 or strict < 300:
            bad_horizons.append(
                f"{horizon}:strict={strict};minimum={minimum};ready={item.get('ready')}"
            )
    if bad_horizons:
        return _decision(
            "btc_research_readiness.yml",
            "per_horizon_strict_pit_rows_below_gate",
            96,
            "PIT_COLLECTION",
            bad_horizons,
        )

    calibration_signals = _calibration_waiting(root)
    if calibration_signals:
        return _decision(
            "btc_adaptive_calibration_replay.yml",
            "current_generation_calibration_evidence_below_effective_gate",
            95,
            "CALIBRATION_COLLECTION",
            calibration_signals,
        )

    # Promotion remains fail-closed. When the durable gate says robustness or
    # holdout evidence is the blocker, schedule the dedicated research-only
    # rich challenger. It compares against the incumbent on a frozen holdout,
    # but its own archive timing is explicitly non-strict and cannot promote.
    promotion_blocked, promotion_signals = _promotion_robustness_blocked(root)
    if promotion_blocked:
        return _decision(
            "btc_rich_production_challenger.yml",
            "promotion_gate_waits_on_robustness_or_holdout_evidence",
            90,
            "PROMOTION_HOLD_RESEARCH",
            promotion_signals,
        )

    frontier_candidates = _frontier_candidates(root)
    if frontier_candidates > 0:
        return _decision(
            "btc_autonomous_data_frontier.yml",
            "eligible_frontier_candidates_pending_research_review",
            85,
            "FRONTIER_WORK_AVAILABLE",
            [f"eligible_candidates={frontier_candidates}"],
        )

    overreach_signals = _high_confidence_overreach(root)
    if overreach_signals:
        return _decision(
            "btc_experience_policy_oos.yml",
            "high_confidence_overreach_requires_reliability_research",
            80,
            "CONFIDENCE_RELIABILITY_RISK",
            overreach_signals,
        )

    experience = _load(root / "data" / "experience" / "experience_summary.json")
    if experience is None:
        return _decision(
            "btc_experience_policy_oos.yml",
            "experience_evidence_missing",
            70,
            "EXPERIENCE_MISSING",
            ["experience_summary_missing"],
        )

    return _decision(
        "btc_ultimate_final_v13_e2e.yml",
        "routine_future_generalization_evidence_refresh",
        50,
        "HEALTHY_ROUTINE",
        [],
    )


def validate(route: dict[str, Any]) -> dict[str, Any]:
    workflow = route.get("workflow")
    if workflow not in ALLOWED:
        raise ValueError("router_selected_unapproved_workflow")
    try:
        threshold = int(route.get("threshold_seconds", 0))
        priority = int(route.get("priority", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("router_numeric_contract_invalid") from exc
    if threshold <= 0 or threshold > 7 * 24 * 3600:
        raise ValueError("router_threshold_out_of_bounds")
    if route.get("production_impact") is not False:
        raise ValueError("router_production_impact_must_be_false")
    reason = str(route.get("reason", "")).strip()
    if not reason:
        raise ValueError("router_reason_missing")
    return {
        "workflow": workflow,
        "threshold_seconds": threshold,
        "reason": reason,
        "production_impact": False,
        "priority": priority,
        "evidence_state": str(route.get("evidence_state", "UNKNOWN")),
        "signals": [str(x) for x in route.get("signals", [])] if isinstance(route.get("signals", []), list) else [],
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(validate(choose(root)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
