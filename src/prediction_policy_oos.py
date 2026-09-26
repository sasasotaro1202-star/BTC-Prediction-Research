"""Research-only v13 prediction policy/controller layer for BTC.

This layer consumes the already-computed, PIT-safe v6 research state and chooses
a prediction strategy, output form, compute tier, information-acquisition
priority, and revision action. It never mutates production artifacts.

Measured incremental information value is intentionally kept separate from
heuristic utility proxies. A candidate cannot be treated as having measured
OOS value unless an independent ablation supplies that evidence.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "historical_research" / "maximum_future_generalization_v6_registry.json"
OUT = ROOT / "data" / "historical_research" / "prediction_policy_oos.json"

HORIZON_MINUTES = {"5m": 5, "10m": 10}
VALID_STRATEGIES = {
    "STANDARD_MODEL",
    "ENSEMBLE",
    "RETRIEVAL_FUSION",
    "SPECIALIST",
    "DEEP_COMPUTE",
    "SCENARIO",
    "ABSTAIN",
}
VALID_OUTPUTS = {
    "SINGLE_PREDICTION",
    "PROBABILITY_DISTRIBUTION",
    "PROBABILITY_RANGE",
    "PREDICTION_SET",
    "SCENARIO",
    "ABSTAIN",
}
VALID_ACTIONS = {
    "MAINTAIN",
    "MICRO_REVISION",
    "MAJOR_REVISION",
    "RECOMPUTE",
    "DEEP_RECOMPUTE",
    "ADD_INFORMATION",
    "CHANGE_MODEL",
    "ADD_RETRIEVAL",
    "TTA",
    "SCENARIO_EXPANSION",
    "ABSTAIN",
    "FALLBACK",
}


def _clip01(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return float(max(0.0, min(1.0, value)))


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_failure(result: dict[str, Any]) -> dict[str, float]:
    latest = result.get("failure_monitoring", {}).get("latest_failure_risk", {})
    if not isinstance(latest, dict):
        latest = {}
    return {
        "mean": _clip01(latest.get("mean"), 0.5),
        "max": _clip01(latest.get("max"), 0.5),
        "time_to_failure_blocks": max(
            0.0,
            float(latest.get("expected_time_to_failure_blocks", 0.0))
            if _finite(latest.get("expected_time_to_failure_blocks"))
            else 0.0,
        ),
    }


def _state(result: dict[str, Any]) -> dict[str, Any]:
    pred = result.get("predictability", {})
    uncertainty = result.get("uncertainty", {})
    transition = result.get("regime_transition", {})
    shock = result.get("information_shock", {})
    momentum = result.get("prediction_momentum", {})
    counterfactual = result.get("counterfactual_stability", {})
    failure = _safe_failure(result)
    return {
        "predictability": _clip01(pred.get("global"), 0.5),
        "predictability_velocity": float(pred.get("velocity", 0.0) or 0.0),
        "predictability_acceleration": float(pred.get("acceleration", 0.0) or 0.0),
        "uncertainty": _clip01(uncertainty.get("total"), 0.5),
        "drift": _clip01(
            result.get("hidden_state", {}).get("stress", 0.0),
            0.0,
        ),
        "information_shock": _clip01(shock.get("shock_score"), 0.0),
        "regime": str(transition.get("current", "UNKNOWN")),
        "next_regime": str(transition.get("next_regime", "UNKNOWN")),
        "next_regime_probability": _clip01(
            transition.get("next_probability", {}).get(
                str(transition.get("next_regime", "UNKNOWN")), 0.0
            ),
            0.0,
        ),
        "regime_stability": _clip01(transition.get("stay_probability"), 0.5),
        "prediction_velocity": max(0.0, float(momentum.get("velocity", 0.0) or 0.0)),
        "prediction_reversal": _clip01(momentum.get("reversal"), 0.0),
        "counterfactual_instability": _clip01(
            counterfactual.get("instability"), 0.0
        ),
        "failure_mean": failure["mean"],
        "failure_max": failure["max"],
        "time_to_failure_blocks": failure["time_to_failure_blocks"],
    }


def select_compute_tier(state: dict[str, Any]) -> str:
    """Choose a bounded research compute tier; thresholds are policy, not fitted parameters."""
    risk = max(
        state["uncertainty"],
        state["failure_mean"],
        1.0 - state["predictability"],
    )
    stress = max(
        state["information_shock"],
        state["drift"],
        state["counterfactual_instability"],
    )
    if risk >= 0.80 or stress >= 0.85:
        return "DEEP"
    if risk >= 0.55 or stress >= 0.55:
        return "ENSEMBLE"
    return "STANDARD"


def select_strategy(state: dict[str, Any], *, compute_tier: str | None = None) -> dict[str, Any]:
    tier = compute_tier or select_compute_tier(state)
    risk = max(state["uncertainty"], state["failure_max"], 1.0 - state["predictability"])
    transition = state["next_regime_probability"] >= 0.60 and state["regime_stability"] < 0.70
    shocked = state["information_shock"] >= 0.60
    unstable = state["counterfactual_instability"] >= 0.60 or state["prediction_reversal"] >= 0.60

    if risk >= 0.92:
        strategy = "ABSTAIN"
        reason = "extreme predicted risk / low predictability"
    elif tier == "DEEP" and (shocked or unstable):
        strategy = "DEEP_COMPUTE"
        reason = "high stress requires deeper counterfactual/research computation"
    elif transition or state["regime"] != state["next_regime"]:
        strategy = "RETRIEVAL_FUSION"
        reason = "probable regime transition benefits from historical-state evidence"
    elif tier == "ENSEMBLE":
        strategy = "ENSEMBLE"
        reason = "moderate uncertainty/risk requires diversified model evidence"
    elif state["uncertainty"] >= 0.55:
        strategy = "SCENARIO"
        reason = "uncertainty is high enough to avoid a single-form output"
    elif state["predictability"] >= 0.70 and state["failure_mean"] < 0.35:
        strategy = "STANDARD_MODEL"
        reason = "state is relatively predictable with lower model-failure risk"
    else:
        strategy = "SPECIALIST"
        reason = "intermediate state warrants specialist research path"

    return {"strategy": strategy, "reason": reason, "compute_tier": tier}


def select_output_format(state: dict[str, Any], strategy: str) -> dict[str, Any]:
    if strategy == "ABSTAIN" or state["predictability"] < 0.20:
        return {"format": "ABSTAIN", "reason": "prediction value is too low relative to uncertainty"}
    if state["predictability"] >= 0.72 and state["uncertainty"] < 0.35:
        return {"format": "SINGLE_PREDICTION", "reason": "high predictability / low uncertainty"}
    if state["predictability"] >= 0.45 and state["uncertainty"] < 0.60:
        return {"format": "PROBABILITY_DISTRIBUTION", "reason": "probability output preserves uncertainty"}
    if strategy in {"SCENARIO", "RETRIEVAL_FUSION", "DEEP_COMPUTE"}:
        return {"format": "SCENARIO", "reason": "state transition or stress is material"}
    if state["uncertainty"] >= 0.75:
        return {"format": "PREDICTION_SET", "reason": "very high uncertainty; allow multiple outcomes"}
    return {"format": "PROBABILITY_RANGE", "reason": "moderate uncertainty warrants bounded output"}


def select_action(
    state: dict[str, Any],
    *,
    previous_strategy: str | None = None,
    previous_predictability: float | None = None,
) -> dict[str, Any]:
    strategy = select_strategy(state)
    if strategy["strategy"] == "ABSTAIN":
        action = "ABSTAIN"
        reason = "policy-selected abstention"
    elif state["failure_max"] >= 0.80:
        action = "CHANGE_MODEL"
        reason = "high model-failure risk"
    elif state["information_shock"] >= 0.65:
        action = "ADD_INFORMATION"
        reason = "information shock warrants refreshed inputs"
    elif state["counterfactual_instability"] >= 0.65:
        action = "DEEP_RECOMPUTE"
        reason = "local instability warrants deeper recomputation"
    elif state["next_regime_probability"] >= 0.65:
        action = "ADD_RETRIEVAL"
        reason = "probable regime transition warrants historical retrieval"
    else:
        current = strategy["strategy"]
        prior = previous_strategy
        pd = None if previous_predictability is None else state["predictability"] - float(previous_predictability)
        if prior == current and (pd is None or abs(pd) < 0.08) and state["information_shock"] < 0.40:
            action = "MAINTAIN"
            reason = "hysteresis prevents unnecessary revision"
        elif prior and prior != current:
            action = "MICRO_REVISION"
            reason = "controlled strategy transition"
        else:
            action = "RECOMPUTE"
            reason = "initial dynamic strategy selection"
    return {"action": action, "reason": reason}


def rank_information_sources(
    state: dict[str, Any],
    candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Rank candidate information by a utility proxy.

    measured_incremental_oos_value must remain null until independent ablation
    evidence is available. This prevents confusing feature importance with
    information-acquisition value.
    """
    candidates = candidates or [
        {"name": "fresh_market_microstructure", "cost": 0.15, "failure_risk": 0.10},
        {"name": "cross_exchange_state", "cost": 0.20, "failure_risk": 0.12},
        {"name": "exogenous_information", "cost": 0.35, "failure_risk": 0.25},
        {"name": "historical_state_retrieval", "cost": 0.10, "failure_risk": 0.08},
    ]
    urgency = max(
        state["information_shock"],
        state["uncertainty"],
        1.0 - state["predictability"],
    )
    ranked = []
    for item in candidates:
        cost = _clip01(item.get("cost"), 0.5)
        failure_risk = _clip01(item.get("failure_risk"), 0.5)
        proxy = _clip01(
            urgency * (1.0 - cost) * (1.0 - failure_risk),
            0.0,
        )
        ranked.append({
            "name": str(item.get("name", "unknown")),
            "estimated_utility_proxy": proxy,
            "measured_incremental_oos_value": None,
            "measurement_status": "UNVERIFIED_UNTIL_ABLATION",
            "cost": cost,
            "failure_risk": failure_risk,
        })
    ranked.sort(key=lambda x: x["estimated_utility_proxy"], reverse=True)
    return ranked


def project_prediction_trajectory(state: dict[str, Any], steps: int = 3) -> dict[str, Any]:
    """Project controller confidence/predictability as a labeled proxy, never as OOS evidence."""
    steps = max(1, min(int(steps), 12))
    predictability = state["predictability"]
    failure = state["failure_mean"]
    rows = []
    for step in range(0, steps + 1):
        p = _clip01(
            predictability
            + state["predictability_velocity"] * step
            + 0.5 * state["predictability_acceleration"] * step * step,
            predictability,
        )
        f = _clip01(
            failure
            + 0.04 * step * max(0.0, state["information_shock"] + state["drift"] - 0.5),
            failure,
        )
        rows.append({
            "step": step,
            "predictability_proxy": p,
            "failure_risk_proxy": f,
        })
    return {
        "status": "PROXY_NOT_OOS_VERIFIED",
        "rows": rows,
        "method": "state_velocity_acceleration_projection",
    }


def forecast_contract(
    horizon: str,
    result: dict[str, Any],
    strategy: str,
    output_format: str,
    action: str,
) -> dict[str, Any]:
    generated = str(result.get("generated_at_utc") or datetime.now(timezone.utc).isoformat())
    try:
        generated_dt = datetime.fromisoformat(generated.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        generated_dt = datetime.now(timezone.utc)
    failure = _safe_failure(result)
    blocks = failure["time_to_failure_blocks"]
    horizon_min = HORIZON_MINUTES.get(horizon, 5)
    lifetime_min = max(float(horizon_min), min(60.0, horizon_min * max(1.0, blocks)))
    valid_until = generated_dt + timedelta(minutes=lifetime_min)
    return {
        "prediction_time": generated_dt.isoformat(),
        "valid_until": valid_until.isoformat(),
        "data_snapshot": "maximum_future_generalization_v6_registry",
        "model_version": str(result.get("champion_benchmark", {}).get("model_version", "research_panel")),
        "strategy": strategy,
        "output_format": output_format,
        "action": action,
        "confidence_proxy": _clip01(
            result.get("predictability", {}).get("global"), 0.5
        ),
        "predictability": _clip01(
            result.get("predictability", {}).get("global"), 0.5
        ),
        "failure_risk": failure["mean"],
        "time_to_failure_blocks": blocks,
        "pit_status": "INHERITED_FROM_V6_EVIDENCE",
        "research_only": True,
        "production_changed": False,
    }


def build_policy_record(horizon: str, result: dict[str, Any]) -> dict[str, Any]:
    state = _state(result)
    strategy = select_strategy(state)
    output = select_output_format(state, strategy["strategy"])
    action = select_action(state)
    info = rank_information_sources(state)
    trajectory = project_prediction_trajectory(state, steps=3)
    contract = forecast_contract(
        horizon,
        result,
        strategy["strategy"],
        output["format"],
        action["action"],
    )
    return {
        "horizon": horizon,
        "status": "IMPLEMENTED_NOT_OOS_VERIFIED",
        "research_only": True,
        "production_changed": False,
        "state": state,
        "strategy_selection": strategy,
        "output_selection": output,
        "action_selection": action,
        "information_acquisition": {
            "status": "HEURISTIC_PRIORITY_ONLY",
            "candidates": info,
            "measured_incremental_oos_value_required": True,
        },
        "adaptive_compute": {
            "tier": strategy["compute_tier"],
            "policy_version": 1,
        },
        "prediction_trajectory": trajectory,
        "forecast_contract": contract,
        "prediction_ledger": {
            "schema_version": 1,
            "prediction_time": contract["prediction_time"],
            "strategy": strategy["strategy"],
            "action": action["action"],
            "output_format": output["format"],
            "reason": action["reason"],
            "immutable_input": "maximum_future_generalization_v6_registry",
        },
    }


def validate_record(record: dict[str, Any]) -> None:
    if record.get("research_only") is not True or record.get("production_changed") is not False:
        raise ValueError("policy layer must remain research-only")
    if record.get("strategy_selection", {}).get("strategy") not in VALID_STRATEGIES:
        raise ValueError("invalid strategy selection")
    if record.get("output_selection", {}).get("format") not in VALID_OUTPUTS:
        raise ValueError("invalid output format")
    if record.get("action_selection", {}).get("action") not in VALID_ACTIONS:
        raise ValueError("invalid action")
    for key in ("predictability", "uncertainty", "failure_mean", "failure_max"):
        if not _finite(record.get("state", {}).get(key)):
            raise ValueError(f"non-finite state field: {key}")


def main() -> None:
    if not REGISTRY.is_file() or REGISTRY.stat().st_size <= 0:
        raise SystemExit("v6 registry missing; refusing to synthesize v13 policy evidence")
    obj = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if obj.get("research_only") is not True or obj.get("production_changed") is not False:
        raise SystemExit("v6 registry is not an accepted research-only input")
    horizons = obj.get("horizons")
    if not isinstance(horizons, dict) or not horizons:
        raise SystemExit("v6 registry has no horizon results")

    records = {}
    for horizon, result in horizons.items():
        if horizon not in HORIZON_MINUTES or not isinstance(result, dict):
            continue
        if result.get("status") != "OK":
            records[horizon] = {
                "horizon": horizon,
                "status": "BLOCKED",
                "reason": result.get("reason", "v6 horizon not OK"),
                "research_only": True,
                "production_changed": False,
            }
            continue
        record = build_policy_record(horizon, result)
        validate_record(record)
        records[horizon] = record

    if not records:
        raise SystemExit("no v6 horizon is available for v13 policy evaluation")

    payload = {
        "schema_version": 1,
        "experiment": "prediction_policy_controller_v13",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_changed": False,
        "status": "IMPLEMENTED_EXECUTED_RESEARCH_ONLY",
        "oos_policy_value_status": "UNVERIFIED_UNTIL_POLICY_LEVEL_ABLATION",
        "horizons": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
