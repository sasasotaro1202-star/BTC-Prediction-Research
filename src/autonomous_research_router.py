"""Deterministic GitHub-side research router for BTC-Prediction-Research.

This module does not fetch data, mutate production state, or select a model.
It reads already-published evidence and returns an ordered, bounded list of
research workflows. The Continuous Supervisor attempts candidates in order and
dispatches at most one stale research lane per heartbeat. Missing/invalid
safety evidence fails closed to the read-only readiness audit.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HORIZONS = ("5m", "10m")

# Explicitly bounded GitHub-side routing surface. Every route is research-only.
ALLOWED = {
    "btc_research_readiness.yml": 3600,
    "btc_autonomous_data_frontier.yml": 900,
    "btc_adaptive_calibration_replay.yml": 28800,
    "btc_experience_policy_oos.yml": 21600,
    "btc_selective_prediction_oos.yml": 43200,
    "btc_rich_production_challenger.yml": 28800,
    "btc_ultimate_final_v13_e2e.yml": 86400,
}

MIN_STRICT_PIT_ROWS = 300
MIN_CALIBRATION_ROWS = 400
HIGH_CONFIDENCE_BUCKET = "0.70+"
HIGH_CONFIDENCE_MIN_ROWS = 75
HIGH_CONFIDENCE_GAP_TRIGGER = 0.15


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def _route(
    workflow: str,
    reason: str,
    *,
    priority: int,
    evidence_state: str,
) -> dict[str, Any]:
    return {
        "workflow": workflow,
        "threshold_seconds": ALLOWED[workflow],
        "reason": reason,
        "production_impact": False,
        "priority": priority,
        "evidence_state": evidence_state,
    }


def _confidence_reliability_trigger(root: Path) -> tuple[bool, dict[str, Any]]:
    """Detect severe high-confidence overprediction from mature experience data.

    This is a research trigger only. It never changes production confidence,
    routing, calibration, or abstention directly.
    """
    experience = _load(root / "data" / "experience" / "experience_summary.json")
    if experience is None:
        return False, {"status": "MISSING"}

    triggered = False
    observations: dict[str, Any] = {}
    horizons = experience.get("horizons")
    if not isinstance(horizons, dict):
        return False, {"status": "INVALID"}

    for horizon in HORIZONS:
        h = horizons.get(horizon)
        if not isinstance(h, dict):
            observations[horizon] = {"status": "MISSING"}
            continue
        buckets = ((h.get("cases") or {}).get("confidence_bucket") or {})
        row = buckets.get(HIGH_CONFIDENCE_BUCKET)
        if not isinstance(row, dict):
            observations[horizon] = {"status": "MISSING"}
            continue
        try:
            n = int(row.get("n", 0) or 0)
            accuracy = float(row.get("accuracy"))
            confidence = float(row.get("avg_confidence"))
        except (TypeError, ValueError):
            observations[horizon] = {"status": "INVALID"}
            continue
        gap = confidence - accuracy
        hit = n >= HIGH_CONFIDENCE_MIN_ROWS and gap >= HIGH_CONFIDENCE_GAP_TRIGGER
        triggered = triggered or hit
        observations[horizon] = {
            "n": n,
            "accuracy": accuracy,
            "average_confidence": confidence,
            "overconfidence_gap": gap,
            "triggered": hit,
        }
    return triggered, {
        "status": "TRIGGERED" if triggered else "CLEAR",
        "threshold_gap": HIGH_CONFIDENCE_GAP_TRIGGER,
        "minimum_rows": HIGH_CONFIDENCE_MIN_ROWS,
        "buckets": observations,
    }


def choose(root: Path) -> dict[str, Any]:
    evidence = root / "data" / "historical_research"

    pit = _load(evidence / "pit_oos_audit.json")
    health = _load(evidence / "research_health.json")
    frontier = _load(evidence / "data_frontier.json")

    # Safety/readiness failures are always first and terminate the route list.
    if pit is None or health is None:
        first = _route(
            "btc_research_readiness.yml",
            "readiness_evidence_missing_or_invalid",
            priority=100,
            evidence_state="MISSING_OR_INVALID",
        )
        return {**first, "candidates": [first]}

    if health.get("ok") is not True:
        first = _route(
            "btc_research_readiness.yml",
            "research_health_not_pass",
            priority=99,
            evidence_state="DATA_HEALTH_BLOCKED",
        )
        return {**first, "candidates": [first]}

    if pit.get("ok") is not True or pit.get("pit_verified") is not True:
        first = _route(
            "btc_research_readiness.yml",
            "strict_pit_evidence_not_verified",
            priority=98,
            evidence_state="PIT_NOT_VERIFIED",
        )
        return {**first, "candidates": [first]}

    horizon_gate = pit.get("primary_horizon_gate")
    if not isinstance(horizon_gate, dict) or set(horizon_gate) != set(HORIZONS):
        first = _route(
            "btc_research_readiness.yml",
            "per_horizon_pit_gate_missing",
            priority=97,
            evidence_state="PIT_GATE_INCOMPLETE",
        )
        return {**first, "candidates": [first]}

    if any(
        not isinstance(horizon_gate.get(h), dict)
        or horizon_gate[h].get("ready") is not True
        or int(horizon_gate[h].get("strict_primary_settled", 0) or 0) < MIN_STRICT_PIT_ROWS
        or int(horizon_gate[h].get("minimum", 0) or 0) < MIN_STRICT_PIT_ROWS
        for h in HORIZONS
    ):
        first = _route(
            "btc_research_readiness.yml",
            "per_horizon_strict_pit_rows_below_gate",
            priority=96,
            evidence_state="PIT_COLLECTION",
        )
        return {**first, "candidates": [first]}

    # Keep the current model-generation calibration collection/replay signal at
    # the front. Production calibration activation remains separately gated.
    calibration_waiting = False
    for horizon in HORIZONS:
        c = _load(root / "models" / f"{horizon}.calibration.json")
        if c is None:
            calibration_waiting = True
            break
        try:
            n = int(c.get("n_settled", 0) or 0)
        except (TypeError, ValueError):
            n = 0
        if n < MIN_CALIBRATION_ROWS or c.get("fit_logloss") is None or c.get("holdout_logloss") is None:
            calibration_waiting = True
            break

    # Frontier candidates are durable discovery evidence; transient
    # data_frontier_run.json is deliberately not used as the routing authority.
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

    confidence_triggered, confidence_detail = _confidence_reliability_trigger(root)
    experience = _load(root / "data" / "experience" / "experience_summary.json")

    routes: list[dict[str, Any]] = []

    if calibration_waiting:
        routes.append(_route(
            "btc_adaptive_calibration_replay.yml",
            "current_generation_calibration_evidence_below_effective_gate",
            priority=95,
            evidence_state="CALIBRATION_COLLECTION",
        ))

    # Severe high-confidence overprediction is an evidence-backed research
    # trigger. Diagnose reliability first, then test selective prediction on a
    # later heartbeat without ever suppressing Production automatically.
    if confidence_triggered:
        routes.append(_route(
            "btc_experience_policy_oos.yml",
            "high_confidence_overprediction_needs_reliability_research",
            priority=94,
            evidence_state="CONFIDENCE_RELIABILITY_TRIGGERED",
        ))
        routes.append(_route(
            "btc_selective_prediction_oos.yml",
            "high_confidence_overprediction_warrants_selective_prediction_oos",
            priority=88,
            evidence_state="SELECTIVE_RESEARCH_PENDING",
        ))
    elif experience is None:
        routes.append(_route(
            "btc_experience_policy_oos.yml",
            "experience_evidence_missing",
            priority=84,
            evidence_state="EXPERIENCE_MISSING",
        ))

    if frontier_candidates:
        routes.append(_route(
            "btc_autonomous_data_frontier.yml",
            "eligible_frontier_candidates_pending_research_review",
            priority=90,
            evidence_state="FRONTIER_WORK_AVAILABLE",
        ))

    # Keep a bounded fallback heartbeat so the research loop does not stop when
    # the higher-priority evidence lanes become non-stale.
    routes.append(_route(
        "btc_ultimate_final_v13_e2e.yml",
        "routine_future_generalization_evidence_refresh",
        priority=50,
        evidence_state="HEALTHY_ROUTINE",
    ))

    # Deterministic de-duplication preserves priority order.
    seen: set[str] = set()
    unique_routes = []
    for item in routes:
        workflow = item["workflow"]
        if workflow in seen:
            continue
        seen.add(workflow)
        unique_routes.append(item)

    unique_routes.sort(key=lambda item: (-int(item["priority"]), item["workflow"]))
    first = unique_routes[0]
    return {
        **first,
        "candidates": unique_routes,
        "confidence_reliability": confidence_detail,
        "frontier_candidates": frontier_candidates,
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

    raw_candidates = route.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("router_candidates_missing")

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            raise ValueError("router_candidate_not_object")
        candidate_workflow = candidate.get("workflow")
        if candidate_workflow not in ALLOWED:
            raise ValueError("router_candidate_workflow_outside_allowlist")
        candidate_threshold = int(candidate.get("threshold_seconds", 0))
        if candidate_threshold != ALLOWED[candidate_workflow]:
            raise ValueError("router_candidate_threshold_mismatch")
        if candidate.get("production_impact") is not False:
            raise ValueError("router_candidate_production_impact_must_be_false")
        if candidate_workflow in seen:
            raise ValueError("router_candidate_duplicate")
        seen.add(candidate_workflow)
        candidates.append({
            "workflow": candidate_workflow,
            "threshold_seconds": candidate_threshold,
            "reason": str(candidate.get("reason", "")),
            "production_impact": False,
            "priority": int(candidate.get("priority", 0)),
            "evidence_state": str(candidate.get("evidence_state", "UNKNOWN")),
        })

    # The published first route must match the top candidate exactly.
    if workflow != candidates[0]["workflow"]:
        raise ValueError("router_first_route_mismatch")
    if int(route["threshold_seconds"]) != candidates[0]["threshold_seconds"]:
        raise ValueError("router_first_threshold_mismatch")

    return {
        "workflow": workflow,
        "threshold_seconds": threshold,
        "reason": str(route.get("reason", "")),
        "production_impact": False,
        "priority": int(route.get("priority", 0)),
        "evidence_state": str(route.get("evidence_state", "UNKNOWN")),
        "candidates": candidates,
        "confidence_reliability": route.get("confidence_reliability", {}),
        "frontier_candidates": int(route.get("frontier_candidates", 0) or 0),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(validate(choose(root)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
