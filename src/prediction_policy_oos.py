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


STRATEGY_TO_V6_VARIANT = {
    "STANDARD_MODEL": "soft_ensemble",
    "ENSEMBLE": "adaptive_ensemble",
    "RETRIEVAL_FUSION": "three_layers_retrieval",
    "SPECIALIST": "three_layers_regime",
    "DEEP_COMPUTE": "full_architecture",
    "SCENARIO": "full_architecture",
}

_CLASS_INDEX = {"DOWN": 0, "FLAT": 1, "UP": 2}


def _policy_state_from_v6_block(block: dict[str, Any]) -> dict[str, Any]:
    raw = dict(block.get("state") or {})
    raw["failure_monitoring"] = {"latest_failure_risk": dict(block.get("failure_state") or {})}
    return _state(raw)


def policy_metrics(y: list[str], probs: Any) -> dict[str, float | int]:
    import numpy as np

    if probs is None:
        return {"n": 0, "accuracy": None, "logloss": None, "brier": None, "calibration_error": None}
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(y) != len(p) or len(y) == 0:
        raise ValueError("invalid_policy_metric_shape")
    yi = np.asarray([_CLASS_INDEX[str(v)] for v in y], dtype=int)
    p = np.clip(p, 1e-7, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    pred = np.argmax(p, axis=1)
    one = np.eye(3)[yi]
    accuracy = float(np.mean(pred == yi))
    logloss = float(-np.mean(np.log(p[np.arange(len(yi)), yi])))
    brier = float(np.mean(np.sum((p - one) ** 2, axis=1)))
    ece = 0.0
    conf = np.max(p, axis=1)
    correct = (pred == yi).astype(float)
    edges = np.linspace(0.0, 1.0, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi if hi < 1.0 else conf <= hi)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(conf[mask])) - float(np.mean(correct[mask])))
    return {
        "n": int(len(y)),
        "accuracy": accuracy,
        "logloss": logloss,
        "brier": brier,
        "calibration_error": float(ece),
    }


def evaluate_policy_case(
    block: dict[str, Any],
    *,
    previous_strategy: str | None = None,
    previous_predictability: float | None = None,
) -> dict[str, Any]:
    state = _policy_state_from_v6_block(block)
    strategy = select_strategy(state)
    output = select_output_format(state, strategy["strategy"])
    action = select_action(
        state,
        previous_strategy=previous_strategy,
        previous_predictability=previous_predictability,
    )
    selected = STRATEGY_TO_V6_VARIANT.get(strategy["strategy"])
    if strategy["strategy"] == "ABSTAIN":
        selected = None
    elif selected is None:
        raise ValueError(f"unmapped_policy_strategy:{strategy['strategy']}")
    probs = None
    if selected is not None:
        variants = block.get("variants") or {}
        if selected not in variants:
            raise ValueError(f"missing_policy_variant:{selected}")
        probs = variants[selected].get("probs")
    return {
        "state": state,
        "strategy_selection": strategy,
        "output_selection": output,
        "action_selection": action,
        "variant": selected,
        "covered": strategy["strategy"] != "ABSTAIN",
        "probs": probs,
    }


def summarize_policy_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    all_y: list[str] = []
    all_baseline: list[Any] = []
    covered_y: list[str] = []
    covered_policy: list[Any] = []
    covered_baseline: list[Any] = []
    strategy_counts: dict[str, int] = {}
    output_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    changed_strategy = 0
    prev_strategy = None
    prev_predictability = None
    cases = []

    for block in blocks:
        y = [str(v) for v in block.get("y", [])]
        baseline = (block.get("variants") or {}).get("soft_ensemble", {}).get("probs")
        if baseline is None or len(y) != len(baseline):
            raise ValueError("policy_baseline_missing")
        case = evaluate_policy_case(
            block,
            previous_strategy=prev_strategy,
            previous_predictability=prev_predictability,
        )
        strategy = case["strategy_selection"]["strategy"]
        output = case["output_selection"]["format"]
        action = case["action_selection"]["action"]
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        output_counts[output] = output_counts.get(output, 0) + 1
        action_counts[action] = action_counts.get(action, 0) + 1
        if prev_strategy is not None and strategy != prev_strategy:
            changed_strategy += 1
        all_y.extend(y)
        all_baseline.append(np.asarray(baseline, dtype=float))
        if case["covered"]:
            policy_p = np.asarray(case["probs"], dtype=float)
            covered_y.extend(y)
            covered_policy.append(policy_p)
            covered_baseline.append(np.asarray(baseline, dtype=float))
        cases.append({
            "index": int(block.get("index", len(cases))),
            "strategy": strategy,
            "variant": case["variant"],
            "output_format": output,
            "action": action,
            "covered": bool(case["covered"]),
            "predictability": float(case["state"]["predictability"]),
            "uncertainty": float(case["state"]["uncertainty"]),
            "failure_risk": float(case["state"]["failure_mean"]),
        })
        prev_strategy = strategy
        prev_predictability = float(case["state"]["predictability"])

    if not all_y:
        return {
            "status": "DEFERRED",
            "reason": "no_policy_cases",
            "n_blocks": 0,
            "n_samples": 0,
        }

    baseline_probs = np.vstack(all_baseline)
    baseline = policy_metrics(all_y, baseline_probs)
    covered = len(covered_y)
    policy = (
        policy_metrics(covered_y, np.vstack(covered_policy))
        if covered else policy_metrics([], None)
    )
    matched = (
        policy_metrics(covered_y, np.vstack(covered_baseline))
        if covered else policy_metrics([], None)
    )

    def delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for key in ("accuracy", "logloss", "brier", "calibration_error"):
            av, bv = a.get(key), b.get(key)
            out[key] = None if av is None or bv is None else float(av - bv)
        return out

    return {
        "status": "MEASURED_DEV_OOS",
        "n_blocks": int(len(blocks)),
        "n_samples": int(len(all_y)),
        "coverage": float(covered / len(all_y)),
        "abstain_rate": float(1.0 - covered / len(all_y)),
        "baseline": baseline,
        "policy": policy,
        "matched_baseline": matched,
        "delta_vs_full_baseline": delta(policy, baseline),
        "delta_vs_matched_baseline": delta(policy, matched),
        "strategy_counts": strategy_counts,
        "output_counts": output_counts,
        "action_counts": action_counts,
        "strategy_switches": int(changed_strategy),
        "cases": cases,
        "policy_value_is_selection_free": True,
        "note": "Development OOS only; frozen holdout remains a separate protected evaluation.",
    }


def evaluate_policy_holdout_case(
    block: dict[str, Any],
    *,
    previous_strategy: str | None = None,
    previous_predictability: float | None = None,
) -> dict[str, Any]:
    case = evaluate_policy_case(
        block,
        previous_strategy=previous_strategy,
        previous_predictability=previous_predictability,
    )
    y = [str(v) for v in block.get("y", [])]
    baseline = (block.get("variants") or {}).get("soft_ensemble", {}).get("probs")
    return {
        "decision": {
            "strategy_selection": case["strategy_selection"],
            "output_selection": case["output_selection"],
            "action_selection": case["action_selection"],
            "variant": case["variant"],
            "covered": case["covered"],
        },
        "policy_metrics": policy_metrics(y, case["probs"]),
        "baseline_metrics": policy_metrics(y, baseline),
        "covered_baseline_metrics": policy_metrics(y, baseline) if case["covered"] else policy_metrics([], None),
        "comparison_scope": "FROZEN_HOLDOUT",
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
        "oos_evidence": result.get("prediction_policy_oos", {}),
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
        "oos_policy_value_status": (
            "MEASURED_DEV_OOS_AND_FROZEN_HOLDOUT"
            if all(
                isinstance(v, dict)
                and v.get("prediction_policy_oos", {}).get("development", {}).get("status") == "MEASURED_DEV_OOS"
                and v.get("prediction_policy_oos", {}).get("frozen_holdout", {}).get("status") == "FROZEN_HOLDOUT_EVALUATED"
                for v in records.values()
            )
            else "UNVERIFIED_UNTIL_POLICY_LEVEL_ABLATION"
        ),
        "horizons": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
