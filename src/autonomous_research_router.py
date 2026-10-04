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
    "btc_rich_production_challenger.yml": 28800,
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


def choose(root: Path) -> dict[str, Any]:
    evidence = root / "data" / "historical_research"

    pit = _load(evidence / "pit_oos_audit.json")
    health = _load(evidence / "research_health.json")
    frontier = _load(evidence / "data_frontier.json")

    # research_readiness.json is a workflow-local/generated surface and is not
    # required to be committed to main. Durable PIT/health evidence is authoritative.
    if pit is None or health is None:
        return {
            **DEFAULT,
            "production_impact": False,
            "priority": 100,
            "evidence_state": "MISSING_OR_INVALID",
        }

    if health.get("ok") is not True:
        return {
            "workflow": "btc_research_readiness.yml",
            "threshold_seconds": 3600,
            "reason": "research_health_not_pass",
            "production_impact": False,
            "priority": 99,
            "evidence_state": "DATA_HEALTH_BLOCKED",
        }

    if pit.get("ok") is not True or pit.get("pit_verified") is not True:
        return {
            "workflow": "btc_research_readiness.yml",
            "threshold_seconds": 3600,
            "reason": "strict_pit_evidence_not_verified",
            "production_impact": False,
            "priority": 98,
            "evidence_state": "PIT_NOT_VERIFIED",
        }

    horizon_gate = pit.get("primary_horizon_gate")
    if not isinstance(horizon_gate, dict) or set(horizon_gate) != set(HORIZONS):
        return {
            "workflow": "btc_research_readiness.yml",
            "threshold_seconds": 3600,
            "reason": "per_horizon_pit_gate_missing",
            "production_impact": False,
            "priority": 97,
            "evidence_state": "PIT_GATE_INCOMPLETE",
        }

    if any(
        not isinstance(horizon_gate.get(h), dict)
        or horizon_gate[h].get("ready") is not True
        or int(horizon_gate[h].get("strict_primary_settled", 0)) < 300
        for h in HORIZONS
    ):
        return {
            "workflow": "btc_research_readiness.yml",
            "threshold_seconds": 3600,
            "reason": "per_horizon_strict_pit_rows_below_gate",
            "production_impact": False,
            "priority": 96,
            "evidence_state": "PIT_COLLECTION",
        }

    # The frontier controller already knows whether it needs discovery,
    # historical-bound repair, or live evidence accumulation. Only route it
    # here for actions that can be advanced without changing Production.
    frontier_candidates = 0
    if isinstance(frontier, dict):
        candidates = frontier.get("candidates")
        if isinstance(candidates, dict):
            for item in candidates.values():
                if not isinstance(item, dict):
                    continue
                lifecycle = item.get("lifecycle")
                if isinstance(lifecycle, dict) and lifecycle.get("research_selection_eligible") is True:
                    frontier_candidates += 1
    if frontier_candidates > 0:
        return {
            "workflow": "btc_autonomous_data_frontier.yml",
            "threshold_seconds": 900,
            "reason": "eligible_frontier_candidates_pending_research_review",
            "production_impact": False,
            "priority": 90,
            "evidence_state": "FRONTIER_WORK_AVAILABLE",
        }

    # Current-generation production calibration is a live-evidence problem.
    # Replay research is still useful while enough strict-PIT observations
    # accumulate, but it must never activate calibration automatically.
    calibration_waiting = False
    for horizon in HORIZONS:
        c = _load(root / "models" / f"{horizon}.calibration.json")
        if c is None:
            calibration_waiting = True
            break
        try:
            n = int(c.get("n_settled", 0))
        except (TypeError, ValueError):
            n = 0
        if (
            n < 400
            or c.get("fit_logloss") is None
            or c.get("holdout_logloss") is None
        ):
            calibration_waiting = True
            break
    if calibration_waiting:
        return {
            "workflow": "btc_adaptive_calibration_replay.yml",
            "threshold_seconds": 28800,
            "reason": "current_generation_calibration_evidence_below_effective_gate",
            "production_impact": False,
            "priority": 80,
            "evidence_state": "CALIBRATION_COLLECTION",
        }

    # Once safety evidence is healthy, keep experience-policy research moving.
    # It is research-only and has its own promotion gate.
    experience = _load(root / "data" / "experience" / "experience_summary.json")
    if experience is None:
        return {
            "workflow": "btc_experience_policy_oos.yml",
            "threshold_seconds": 21600,
            "reason": "experience_evidence_missing",
            "production_impact": False,
            "priority": 70,
            "evidence_state": "EXPERIENCE_MISSING",
        }

    # Default bounded research heartbeat. V13 itself is fail-closed and has
    # protected promotion/holdout semantics; the supervisor still requires the
    # workflow to be stale before dispatching it.
    return {
        "workflow": "btc_ultimate_final_v13_e2e.yml",
        "threshold_seconds": 86400,
        "reason": "routine_future_generalization_evidence_refresh",
        "production_impact": False,
        "priority": 50,
        "evidence_state": "HEALTHY_ROUTINE",
    }


def validate(route: dict[str, Any]) -> dict[str, Any]:
    workflow = route.get("workflow")
    if workflow not in ALLOWED:
        raise ValueError("router_selected_unapproved_workflow")
    threshold = int(route.get("threshold_seconds", 0))
    if threshold <= 0 or threshold > 7 * 24 * 3600:
        raise ValueError("router_threshold_out_of_bounds")
    if route.get("production_impact") is not False:
        raise ValueError("router_production_impact_must_be_false")
    return {
        "workflow": workflow,
        "threshold_seconds": threshold,
        "reason": str(route.get("reason", "")),
        "production_impact": False,
        "priority": int(route.get("priority", 0)),
        "evidence_state": str(route.get("evidence_state", "UNKNOWN")),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(validate(choose(root)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
