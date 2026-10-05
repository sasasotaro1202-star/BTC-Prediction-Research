"""Deterministic GitHub-side research router for BTC-Prediction-Research.

The router only reads durable published evidence. It never fetches data, selects
a Production model, mutates the Production registry, or calls an external LLM.

It returns an ordered list of bounded research lanes. The Continuous Supervisor
may dispatch at most one stale candidate per heartbeat and falls through when a
higher-priority lane is already active or fresh.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HORIZONS = ("5m", "10m")

ALLOWED = {
    "btc_research_readiness.yml": 3600,
    "btc_recency_challenger.yml": 86400,
    "btc_autonomous_data_frontier.yml": 900,
    "btc_adaptive_calibration_replay.yml": 28800,
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
DRIFT_SCORE_TRIGGER = 0.10
MODEL_DISAGREEMENT_DRIFT_TRIGGER = 0.10
MODEL_STALE_DAYS = 7.0


def _parse_utc_timestamp(raw: object) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


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


def _model_staleness_signals(root: Path) -> list[str]:
    """Detect stale Production Champion generations for research routing only.

    The reference clock is the durable experience snapshot rather than wall time,
    keeping replay/test behavior deterministic. Missing, malformed, candidate,
    or future-dated model metadata is fail-closed and does not trigger routing.
    """
    experience = _load(root / "data" / "experience" / "experience_summary.json")
    if experience is None:
        return []

    if "status" in experience and experience.get("status") != "OK":
        return []

    reference_raw = experience.get("generated_at_utc")
    if not reference_raw:
        return []
    reference = _parse_utc_timestamp(reference_raw)
    if reference is None:
        return []

    signals: list[str] = []
    for horizon in HORIZONS:
        meta = _load(root / "models" / f"{horizon}.json")
        if not isinstance(meta, dict):
            continue
        if meta.get("candidate") is not False:
            continue
        model_version = str(meta.get("model_version", "")).strip()
        artifact = str(meta.get("artifact", "")).strip()
        if not model_version or not artifact:
            continue

        trained_raw = meta.get("trained_at_utc")
        if not trained_raw:
            continue
        trained = _parse_utc_timestamp(trained_raw)
        if trained is None:
            continue

        age_days = (reference - trained).total_seconds() / 86400.0
        if age_days < 0.0:
            continue
        if age_days >= MODEL_STALE_DAYS:
            signals.append(
                f"{horizon}:model={model_version};age_days={age_days:.1f}>{MODEL_STALE_DAYS:g}"
            )
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
    model_staleness_signals = _model_staleness_signals(root)
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

    if model_staleness_signals:
        routes.append(_decision(
            "btc_recency_challenger.yml",
            "production_champion_generation_is_stale_and_requires_recent_data_reassessment",
            92,
            "MODEL_STALENESS_RISK",
            model_staleness_signals,
        ))

    if promotion_blocked:
        routes.append(_decision(
            "btc_rich_production_challenger.yml",
            "promotion_gate_waits_on_robustness_or_holdout_evidence",
            90,
            "PROMOTION_HOLD_RESEARCH",
            promotion_signals,
        ))

    if uncertainty_signals:
        routes.append(_decision(
            "btc_uncertainty_layer_oos.yml",
            "material_drift_or_model_disagreement_requires_uncertainty_research",
            84,
            "DRIFT_UNCERTAINTY_RISK",
            uncertainty_signals,
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
