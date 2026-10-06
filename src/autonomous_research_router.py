"""Deterministic GitHub-side research router for BTC-Prediction-Research.

The router only reads durable published evidence. It never fetches data, selects
a Production model, mutates the Production registry, or calls an external LLM.

It returns an ordered list of bounded research lanes. The Continuous Supervisor
may dispatch at most one stale candidate per heartbeat and falls through when a
higher-priority lane is already active or fresh.
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
    "btc_return_distribution_tail_oos.yml": 21600,
    "btc_experience_policy_oos.yml": 21600,
    "btc_selective_prediction_oos.yml": 43200,
    "btc_uncertainty_layer_oos.yml": 21600,
    "btc_rich_production_challenger.yml": 86400,
    "btc_ultimate_final_v13_e2e.yml": 86400,
}

MIN_STRICT_PIT_ROWS = 300
MIN_CALIBRATION_ROWS = 400
HIGH_CONFIDENCE_MIN_ROWS = 50
HIGH_CONFIDENCE_GAP_TRIGGER = 0.15
FAILURE_RISK_MIN_META_SAMPLES = 20
FAILURE_RISK_TRIGGER = 0.70
DRIFT_SCORE_TRIGGER = 0.10
MODEL_DISAGREEMENT_DRIFT_TRIGGER = 0.10


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
        "priority": int(priority),
        "evidence_state": evidence_state,
        "signals": list(signals or []),
    }


def _ordered(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    if not decisions:
        raise ValueError("router_no_research_candidate")

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in decisions:
        workflow = item.get("workflow")
        if workflow in seen:
            continue
        seen.add(str(workflow))
        unique.append(item)

    unique.sort(key=lambda item: (-int(item["priority"]), str(item["workflow"])))
    signals: list[str] = []
    for item in unique:
        for signal in item.get("signals", []):
            if signal not in signals:
                signals.append(signal)

    first = unique[0]
    return {
        **first,
        "candidates": unique,
        "signals": signals,
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
        if n < MIN_CALIBRATION_ROWS:
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
    blockers = [token for token in ("robustness", "holdout") if token in reasons]
    if not hold or not blockers:
        return False, []
    return True, [f"promotion_gate:{token}_blocked" for token in blockers]


def _return_tail_research_status(root: Path) -> list[str]:
    """Return research-only triggers for the conditional return/tail lane."""
    path = root / "data" / "historical_research" / "return_distribution_tail_oos.json"
    if not path.is_file():
        return ["return_distribution_tail_evidence_missing"]

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return ["return_distribution_tail_evidence_invalid"]

    if obj.get("research_only") is not True or obj.get("production_changed") is not False:
        return ["return_distribution_tail_boundary_invalid"]

    generated = obj.get("generated_at_utc")
    parsed = None
    if generated:
        try:
            from datetime import datetime, timezone
            parsed = datetime.fromisoformat(str(generated).replace("Z", "+00:00")).astimezone(timezone.utc)
        except (TypeError, ValueError):
            return ["return_distribution_tail_generated_at_invalid"]

    if parsed is None:
        return ["return_distribution_tail_generated_at_missing"]

    from datetime import datetime, timezone
    age = (datetime.now(timezone.utc) - parsed).total_seconds()
    if age < 0:
        return ["return_distribution_tail_generated_in_future"]

    # A DEFERRED result is expected while live-primary mature rows accumulate;
    # once the artifact is stale, rerun rather than treating DEFERRED as success.
    threshold = 21600
    if age >= threshold:
        status = str(obj.get("horizons", {}).get("5m", {}).get("status", "UNKNOWN"))
        status10 = str(obj.get("horizons", {}).get("10m", {}).get("status", "UNKNOWN"))
        return [f"return_distribution_tail_stale:{int(age)}s;5m={status};10m={status10}"]
    return []


def _uncertainty_drift_signals(root: Path) -> list[str]:
    """Return research-only uncertainty triggers from durable drift evidence."""
    signals: list[str] = []
    for horizon in HORIZONS:
        obj = _load(root / "data" / "historical_research" / f"innovative_control_{horizon}_drift_detector.json")
        if obj is None or obj.get("research_only") is not True:
            continue
        latest = obj.get("latest_drift")
        if not isinstance(latest, dict):
            continue
        try:
            drift_score = float(latest["drift_score"])
            disagreement = float(latest["model_disagreement_drift"])
        except (KeyError, TypeError, ValueError):
            continue
        if drift_score >= DRIFT_SCORE_TRIGGER or disagreement >= MODEL_DISAGREEMENT_DRIFT_TRIGGER:
            signals.append(
                f"{horizon}:drift_score={drift_score:.3f};model_disagreement_drift={disagreement:.3f}"
            )
    return signals


def _future_failure_risk_signals(root: Path) -> list[str]:
    """Return research-only triggers when learned future-failure risk is materially high."""
    signals: list[str] = []
    for horizon in HORIZONS:
        obj = _load(
            root / "data" / "historical_research"
            / f"innovative_control_{horizon}_future_failure_model.json"
        )
        if obj is None or obj.get("research_only") is not True:
            continue
        latest_risk = obj.get("latest_risk")
        meta_samples = obj.get("meta_samples")
        if not isinstance(latest_risk, dict) or not isinstance(meta_samples, dict):
            continue
        for model_name, risk_value in sorted(latest_risk.items()):
            try:
                risk = float(risk_value)
                samples = int(meta_samples.get(model_name, 0))
            except (TypeError, ValueError):
                continue
            if not (0.0 <= risk <= 1.0):
                continue
            if samples >= FAILURE_RISK_MIN_META_SAMPLES and risk >= FAILURE_RISK_TRIGGER:
                signals.append(
                    f"{horizon}:{model_name}:future_failure_risk={risk:.3f};n={samples}"
                )
    return signals


def _high_confidence_overreach(root: Path) -> list[str]:
    """Research-only trigger for mature high-confidence overprediction."""
    experience = _load(root / "data" / "experience" / "experience_summary.json")
    if experience is None:
        return []

    signals: list[str] = []
    horizons = experience.get("horizons")
    if not isinstance(horizons, dict):
        return signals

    for horizon in HORIZONS:
        h = horizons.get(horizon)
        if not isinstance(h, dict):
            continue
        cases = h.get("cases")
        if not isinstance(cases, dict):
            continue
        buckets = cases.get("confidence_bucket")
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
        if n >= HIGH_CONFIDENCE_MIN_ROWS and gap >= HIGH_CONFIDENCE_GAP_TRIGGER:
            signals.append(f"{horizon}:high_confidence_gap={gap:.3f};n={n}")
    return signals


def choose(root: Path) -> dict[str, Any]:
    evidence = root / "data" / "historical_research"

    pit = _load(evidence / "pit_oos_audit.json")
    health = _load(evidence / "research_health.json")

    # Durable safety evidence is a hard gate. Do not continue to lower-priority
    # research lanes when readiness itself is uncertain.
    if pit is None or health is None:
        return _ordered([
            _decision(
                "btc_research_readiness.yml",
                "readiness_evidence_missing_or_invalid",
                100,
                "MISSING_OR_INVALID",
                ["durable_pit_or_health_artifact_missing"],
            )
        ])

    if health.get("ok") is not True:
        return _ordered([
            _decision(
                "btc_research_readiness.yml",
                "research_health_not_pass",
                99,
                "DATA_HEALTH_BLOCKED",
                ["research_health_ok_not_true"],
            )
        ])

    if pit.get("ok") is not True or pit.get("pit_verified") is not True:
        return _ordered([
            _decision(
                "btc_research_readiness.yml",
                "strict_pit_evidence_not_verified",
                98,
                "PIT_NOT_VERIFIED",
                ["pit_ok_or_verified_not_true"],
            )
        ])

    horizon_gate = pit.get("primary_horizon_gate")
    if not isinstance(horizon_gate, dict) or set(horizon_gate) != set(HORIZONS):
        return _ordered([
            _decision(
                "btc_research_readiness.yml",
                "per_horizon_pit_gate_missing",
                97,
                "PIT_GATE_INCOMPLETE",
                ["5m_and_10m_horizon_gate_required"],
            )
        ])

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
        if item.get("ready") is not True or minimum < MIN_STRICT_PIT_ROWS or strict < MIN_STRICT_PIT_ROWS:
            bad_horizons.append(
                f"{horizon}:strict={strict};minimum={minimum};ready={item.get('ready')}"
            )
    if bad_horizons:
        return _ordered([
            _decision(
                "btc_research_readiness.yml",
                "per_horizon_strict_pit_rows_below_gate",
                96,
                "PIT_COLLECTION",
                bad_horizons,
            )
        ])

    calibration_signals = _calibration_waiting(root)
    promotion_blocked, promotion_signals = _promotion_robustness_blocked(root)
    frontier_candidates = _frontier_candidates(root)
    overreach_signals = _high_confidence_overreach(root)
    uncertainty_signals = _uncertainty_drift_signals(root)
    future_failure_signals = _future_failure_risk_signals(root)
    experience = _load(root / "data" / "experience" / "experience_summary.json")

    routes: list[dict[str, Any]] = []

    if calibration_signals:
        routes.append(_decision(
            "btc_adaptive_calibration_replay.yml",
            "current_generation_calibration_evidence_below_effective_gate",
            95,
            "CALIBRATION_COLLECTION",
            calibration_signals,
        ))

    if promotion_blocked:
        routes.append(_decision(
            "btc_rich_production_challenger.yml",
            "promotion_gate_waits_on_robustness_or_holdout_evidence",
            90,
            "PROMOTION_HOLD_RESEARCH",
            promotion_signals,
        ))

    return_tail_signals = _return_tail_research_status(root)
    if return_tail_signals:
        routes.append(_decision(
            "btc_return_distribution_tail_oos.yml",
            "conditional_return_distribution_and_tail_evidence_is_missing_or_stale",
            88,
            "RETURN_DISTRIBUTION_TAIL_RESEARCH",
            return_tail_signals,
        ))

    combined_uncertainty_signals = uncertainty_signals + future_failure_signals
    if combined_uncertainty_signals:
        failure_priority = 86 if future_failure_signals else 84
        failure_reason = (
            "future_failure_risk_or_material_drift_requires_uncertainty_research"
            if future_failure_signals
            else "material_drift_or_model_disagreement_requires_uncertainty_research"
        )
        failure_state = (
            "FUTURE_FAILURE_RISK"
            if future_failure_signals
            else "DRIFT_UNCERTAINTY_RISK"
        )
        routes.append(_decision(
            "btc_uncertainty_layer_oos.yml",
            failure_reason,
            failure_priority,
            failure_state,
            combined_uncertainty_signals,
        ))

    if overreach_signals:
        routes.append(_decision(
            "btc_experience_policy_oos.yml",
            "high_confidence_overreach_requires_reliability_research",
            85,
            "CONFIDENCE_RELIABILITY_RISK",
            overreach_signals,
        ))
        routes.append(_decision(
            "btc_selective_prediction_oos.yml",
            "high_confidence_overreach_warrants_selective_prediction_oos",
            82,
            "SELECTIVE_RESEARCH_PENDING",
            overreach_signals,
        ))
    elif experience is None:
        routes.append(_decision(
            "btc_experience_policy_oos.yml",
            "experience_evidence_missing",
            70,
            "EXPERIENCE_MISSING",
            ["experience_summary_missing"],
        ))

    if frontier_candidates > 0:
        routes.append(_decision(
            "btc_autonomous_data_frontier.yml",
            "eligible_frontier_candidates_pending_research_review",
            80,
            "FRONTIER_WORK_AVAILABLE",
            [f"eligible_candidates={frontier_candidates}"],
        ))

    routes.append(_decision(
        "btc_ultimate_final_v13_e2e.yml",
        "routine_future_generalization_evidence_refresh",
        50,
        "HEALTHY_ROUTINE",
        [],
    ))

    return _ordered(routes)


def validate(route: dict[str, Any]) -> dict[str, Any]:
    workflow = route.get("workflow")
    if workflow not in ALLOWED:
        raise ValueError("router_selected_unapproved_workflow")

    try:
        threshold = int(route.get("threshold_seconds", 0))
        priority = int(route.get("priority", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("router_numeric_contract_invalid") from exc

    if threshold != ALLOWED[workflow]:
        raise ValueError("router_threshold_mismatch")
    if route.get("production_impact") is not False:
        raise ValueError("router_production_impact_must_be_false")

    reason = str(route.get("reason", "")).strip()
    evidence_state = str(route.get("evidence_state", "")).strip()
    if not reason:
        raise ValueError("router_reason_missing")
    if not evidence_state:
        raise ValueError("router_evidence_state_missing")

    signals = route.get("signals", [])
    if not isinstance(signals, list) or not all(isinstance(x, str) and x for x in signals):
        raise ValueError("router_signals_invalid")

    raw_candidates = route.get("candidates")
    if not isinstance(raw_candidates, list) or not (1 <= len(raw_candidates) <= len(ALLOWED)):
        raise ValueError("router_candidates_missing_or_out_of_bounds")

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    previous_priority: int | None = None
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            raise ValueError("router_candidate_not_object")
        candidate_workflow = candidate.get("workflow")
        if candidate_workflow not in ALLOWED:
            raise ValueError("router_candidate_workflow_outside_allowlist")
        if candidate_workflow in seen:
            raise ValueError("router_candidate_duplicate")
        seen.add(candidate_workflow)

        candidate_threshold = int(candidate.get("threshold_seconds", 0))
        if candidate_threshold != ALLOWED[candidate_workflow]:
            raise ValueError("router_candidate_threshold_mismatch")
        if candidate.get("production_impact") is not False:
            raise ValueError("router_candidate_production_impact_must_be_false")

        candidate_priority = int(candidate.get("priority", 0))
        if previous_priority is not None and candidate_priority > previous_priority:
            raise ValueError("router_candidates_not_priority_ordered")
        previous_priority = candidate_priority

        candidate_reason = str(candidate.get("reason", "")).strip()
        candidate_state = str(candidate.get("evidence_state", "")).strip()
        candidate_signals = candidate.get("signals", [])
        if not candidate_reason:
            raise ValueError("router_candidate_reason_missing")
        if not candidate_state:
            raise ValueError("router_candidate_evidence_state_missing")
        if not isinstance(candidate_signals, list) or not all(
            isinstance(x, str) and x for x in candidate_signals
        ):
            raise ValueError("router_candidate_signals_invalid")

        candidates.append({
            "workflow": candidate_workflow,
            "threshold_seconds": candidate_threshold,
            "reason": candidate_reason,
            "production_impact": False,
            "priority": candidate_priority,
            "evidence_state": candidate_state,
            "signals": list(candidate_signals),
        })

    if candidates[0]["workflow"] != workflow:
        raise ValueError("router_first_candidate_mismatch")

    return {
        "workflow": workflow,
        "threshold_seconds": threshold,
        "reason": reason,
        "production_impact": False,
        "priority": priority,
        "evidence_state": evidence_state,
        "signals": list(signals),
        "candidates": candidates,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(validate(choose(root)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
