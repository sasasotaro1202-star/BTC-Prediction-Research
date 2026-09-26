"""V13 meta-control extension for BTC prediction research.

This module extends the existing Ultimate Final V13 contract without creating a
second prediction engine. It consumes the already verified V6/V13 evidence and
adds the missing policy layer: strategy selection, output-format selection,
active-information gating, update policy, adaptive compute, trajectory/state
monitoring, strategy failure memory, and prediction-ledger contracts.

No production artifact is selected or mutated. Missing point-in-time live
probability or source-level incremental-OOS evidence is treated as DEFERRED,
never fabricated.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data" / "historical_research"
V6 = EVIDENCE / "maximum_future_generalization_v6_registry.json"
V13 = EVIDENCE / "ultimate_final_v13_report.json"
OUT = EVIDENCE / "ultimate_final_v13_meta_control.json"
STRATEGY_OUT = EVIDENCE / "ultimate_final_v13_prediction_strategy.json"
TRAJECTORY_OUT = EVIDENCE / "ultimate_final_v13_prediction_trajectory.json"
INFO_OUT = EVIDENCE / "ultimate_final_v13_active_information.json"
MEMORY_OUT = EVIDENCE / "ultimate_final_v13_failure_memory.json"

CLASSES = ("DOWN", "FLAT", "UP")


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else None


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _clamp(value: Any, lo: float = 0.0, hi: float = 1.0, default: float = 0.0) -> float:
    return float(np.clip(_finite(value, default), lo, hi))


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def probability_contract(probability: Any) -> dict[str, Any]:
    try:
        p = np.asarray(probability, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return {"status": "FAIL", "reason": "probability_not_numeric"}
    if p.size != 3 or not np.all(np.isfinite(p)):
        return {"status": "FAIL", "reason": "probability_shape_or_nan"}
    if np.any(p < 0) or float(p.sum()) <= 0:
        return {"status": "FAIL", "reason": "probability_negative_or_zero_sum"}
    p = p / float(p.sum())
    return {
        "status": "PASS",
        "sum": float(p.sum()),
        "min": float(p.min()),
        "max": float(p.max()),
    }


def _state(v6: Mapping[str, Any], v13: Mapping[str, Any]) -> dict[str, float]:
    pred = v6.get("predictability", {}) or {}
    fail = v6.get("failure_monitoring", {}) or {}
    fail_now = fail.get("latest_failure_risk", {}) or {}
    unc = v6.get("uncertainty", {}) or {}
    fr = v6.get("feature_reliability", {}) or {}
    info = v6.get("information_shock", {}) or {}
    mom = v6.get("prediction_momentum", {}) or {}
    report = (v13.get("core_three_layers", {}) or {}).get("model_disagreement")
    if not isinstance(report, dict):
        report = {}
    disagreement = _clamp(
        report.get("pairwise_class_disagreement"),
        default=_clamp((v6.get("error_correlation", {}) or {}).get("mean_abs_error_correlation"), default=0.25),
    )
    drift_obj = v6.get("drift", {}) or {}
    drift = _clamp(
        drift_obj.get("drift_score"),
        default=_clamp(info.get("shock_score"), default=0.0),
    )
    source = v6.get("source_reliability", {})
    if isinstance(source, dict):
        source = source.get("reliability", 0.85)
    source = _clamp(source, default=0.85)
    feature_unreliability = 1.0 - _clamp(fr.get("global"), default=0.50)
    uncertainty = _clamp(unc.get("total"), default=0.50)
    ood = _clamp(
        0.45 * feature_unreliability
        + 0.35 * drift
        + 0.20 * uncertainty
    )
    return {
        "predictability": _clamp(pred.get("global"), default=0.50),
        "failure_risk_mean": _clamp(fail_now.get("mean"), default=0.50),
        "failure_risk_max": _clamp(fail_now.get("max"), default=0.50),
        "drift": drift,
        "uncertainty": uncertainty,
        "source_reliability": source,
        "feature_unreliability": feature_unreliability,
        "ood": ood,
        "disagreement": disagreement,
        "prediction_velocity": _finite(mom.get("velocity")),
        "prediction_acceleration": _finite(mom.get("acceleration")),
    }


def select_strategy(state: Mapping[str, float], evidence: Mapping[str, Any]) -> dict[str, Any]:
    if state["source_reliability"] < 0.40 or state["ood"] >= 0.85:
        return {"strategy": "FALLBACK", "reason": "unsafe_source_or_ood"}
    if state["predictability"] <= 0.20 or state["uncertainty"] >= 0.85 or state["failure_risk_max"] >= 0.85:
        return {"strategy": "ABSTAIN", "reason": "extreme_risk"}
    if state["uncertainty"] >= 0.80 or state["failure_risk_mean"] >= 0.70:
        candidate = "FULL_ARCHITECTURE"
    elif state["predictability"] < 0.45 or state["drift"] >= 0.65:
        candidate = "ADAPTIVE_ENSEMBLE"
    elif state["disagreement"] >= 0.40:
        candidate = "THREE_LAYERS"
    else:
        candidate = "STANDARD_ENSEMBLE"

    summary = evidence.get("oos_summary", {}) or {}
    key = {
        "STANDARD_ENSEMBLE": "soft_ensemble",
        "ADAPTIVE_ENSEMBLE": "adaptive_ensemble",
        "THREE_LAYERS": "three_layers",
        "FULL_ARCHITECTURE": "full_architecture",
    }.get(candidate)
    blocks = _finite((summary.get(key, {}) or {}).get("blocks"), 0.0) if key else 0.0
    if blocks < 4 and candidate not in {"ABSTAIN", "FALLBACK"}:
        candidate = "STANDARD_ENSEMBLE"
    return {"strategy": candidate, "reason": "state_conditioned_policy", "oos_blocks": blocks}


def select_output(state: Mapping[str, float]) -> dict[str, Any]:
    if state["source_reliability"] < 0.40 or state["ood"] >= 0.90:
        return {"format": "ABSTAIN", "reason": "unsafe_state"}
    if state["predictability"] >= 0.72 and state["uncertainty"] < 0.30:
        return {"format": "SINGLE", "reason": "high_predictability"}
    if state["predictability"] >= 0.50 and state["uncertainty"] < 0.55:
        return {"format": "PROBABILITY", "reason": "moderate_predictability"}
    if state["predictability"] >= 0.30 and state["uncertainty"] < 0.75:
        return {"format": "RANGE", "reason": "low_predictability_or_high_uncertainty"}
    if state["predictability"] > 0.0:
        return {"format": "SET_OR_SCENARIO", "reason": "very_low_predictability"}
    return {"format": "ABSTAIN", "reason": "no_predictability_evidence"}


def update_policy(state: Mapping[str, float]) -> dict[str, Any]:
    need = _clamp(
        0.30 * (1.0 - state["predictability"])
        + 0.25 * state["drift"]
        + 0.20 * state["uncertainty"]
        + 0.15 * state["failure_risk_mean"]
        + 0.10 * state["disagreement"]
    )
    if state["source_reliability"] < 0.40 or state["ood"] >= 0.90:
        action = "FALLBACK"
    elif state["predictability"] <= 0.20 or state["uncertainty"] >= 0.90:
        action = "ABSTAIN"
    elif need < 0.20:
        action = "MAINTAIN"
    elif need < 0.45:
        action = "RECOMPUTE"
    else:
        action = "DEEP_RECOMPUTE"
    return {
        "action": action,
        "update_need": need,
        "current_probability_available": False,
        "note": "Aggregate V6/V13 evidence does not expose a live point-probability vector; no probability delta or flip is fabricated",
    }


def adaptive_compute(state: Mapping[str, float]) -> dict[str, Any]:
    difficulty = _clamp(
        0.35 * (1.0 - state["predictability"])
        + 0.25 * state["uncertainty"]
        + 0.20 * state["failure_risk_mean"]
        + 0.20 * state["drift"]
    )
    if difficulty < 0.25:
        tier, budget = "STANDARD", 1
    elif difficulty < 0.50:
        tier, budget = "ENSEMBLE", 2
    elif difficulty < 0.75:
        tier, budget = "DEEP", 3
    else:
        tier, budget = "SAFE_STOP", 0
    return {"difficulty": difficulty, "tier": tier, "compute_budget_units": budget}


def trajectory_proxy(state: Mapping[str, float]) -> dict[str, Any]:
    future_predictability = []
    future_failure = []
    p = state["predictability"]
    f = state["failure_risk_mean"]
    velocity = _finite(state["prediction_velocity"])
    for step in range(0, 7):
        decay = 0.70 ** step
        future_predictability.append({
            "step": step,
            "predictability": _clamp(p + velocity * decay),
        })
        future_failure.append({
            "step": step,
            "failure_risk": _clamp(f + 0.50 * max(0.0, -velocity) * decay),
        })
    return {
        "status": "PARTIAL",
        "predictability_trajectory": future_predictability,
        "failure_risk_trajectory": future_failure,
        "probability_trajectory": {
            "status": "DEFERRED",
            "reason": "no_point_in_time_probability_vector_in_aggregate_evidence",
        },
        "revision_points": [
            x for x in future_predictability if x["predictability"] < 0.35
        ],
        "validity_score": _clamp(
            0.60 * state["predictability"] + 0.40 * (1.0 - state["uncertainty"])
        ),
    }


def active_information(evidence: Mapping[str, Any]) -> dict[str, Any]:
    configured = evidence.get("active_information_sources", {})
    if not isinstance(configured, dict) or not configured:
        configured = {
            "market": {"pit_status": "PASS"},
            "depth": {"pit_status": "DEFERRED"},
            "taker": {"pit_status": "DEFERRED"},
            "premium": {"pit_status": "DEFERRED"},
            "news": {"pit_status": "DEFERRED"},
        }
    candidates = []
    for source, item in configured.items():
        item = item if isinstance(item, dict) else {}
        pit = str(item.get("pit_status", "UNKNOWN")).upper()
        gain = item.get("incremental_oos_gain")
        if pit != "PASS":
            candidates.append({
                "source": source,
                "status": "DEFERRED",
                "reason": "PIT_NOT_VERIFIED",
            })
            continue
        if not isinstance(gain, (int, float)) or not math.isfinite(float(gain)):
            candidates.append({
                "source": source,
                "status": "DEFERRED",
                "reason": "INCREMENTAL_OOS_VALUE_UNMEASURED",
            })
            continue
        cost = max(0.0, _finite(item.get("cost")))
        failure = _clamp(item.get("retrieval_failure_risk"), default=0.0)
        value = float(gain) - cost - 0.25 * failure
        candidates.append({
            "source": source,
            "status": "CANDIDATE",
            "incremental_oos_gain": float(gain),
            "cost": cost,
            "retrieval_failure_risk": failure,
            "expected_value": value,
        })
    viable = [x for x in candidates if x["status"] == "CANDIDATE"]
    viable.sort(key=lambda x: x["expected_value"], reverse=True)
    return {
        "status": "OK" if viable else "DEFERRED",
        "selected": viable[0] if viable else None,
        "candidates": candidates,
        "policy": "incremental_OOS - retrieval_cost - retrieval_failure_risk; PIT failure is fail-closed",
    }


def strategy_memory(state: Mapping[str, float], evidence: Mapping[str, Any], strategy: str) -> dict[str, Any]:
    summary = evidence.get("oos_summary", {}) or {}
    keys = {
        "STANDARD_ENSEMBLE": "soft_ensemble",
        "ADAPTIVE_ENSEMBLE": "adaptive_ensemble",
        "THREE_LAYERS": "three_layers",
        "FULL_ARCHITECTURE": "full_architecture",
    }
    row = summary.get(keys.get(strategy, ""), {}) or {}
    blocks = _finite(row.get("blocks"), 0.0)
    if blocks <= 0:
        return {
            "status": "DEFERRED",
            "strategy": strategy,
            "reason": "NO_STRATEGY_OOS_EVIDENCE",
        }
    return {
        "status": "OK",
        "strategy": strategy,
        "oos_blocks": blocks,
        "accuracy": _finite(row.get("accuracy")),
        "logloss": _finite(row.get("logloss"), 1.5),
        "brier": _finite(row.get("brier"), 0.67),
        "ece": _finite(row.get("calibration_error")),
        "failure_risk_proxy": _clamp(
            0.50 * state["failure_risk_mean"]
            + 0.25 * (1.0 - _clamp(row.get("accuracy"), default=0.33))
            + 0.25 * _clamp(row.get("calibration_error"), default=0.0)
        ),
    }


def prediction_ledger_contract(
    horizon: str,
    state: Mapping[str, float],
    strategy: str,
    output_format: str,
    action: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "status": "SCHEMA_READY_RESEARCH_ONLY",
        "forecast_contract": {
            "prediction_time": None,
            "valid_until": None,
            "data_snapshot": None,
            "model_version": None,
            "strategy": strategy,
            "output_format": output_format,
            "confidence": None,
            "predictability": state["predictability"],
            "uncertainty": state["uncertainty"],
            "future_failure_risk": state["failure_risk_mean"],
            "current_regime": None,
            "future_regime": None,
            "prediction_lifetime": None,
            "revision_needed": action not in {"MAINTAIN"},
            "pit_status": "UNAVAILABLE_AT_AGGREGATE",
            "research_only": True,
            "production_changed": False,
        },
        "ledger": {
            "horizon": horizon,
            "strategy": strategy,
            "action": action,
            "failure_risk": state["failure_risk_mean"],
            "ood_proxy": state["ood"],
        },
        "integrity_note": "No missing live fields are guessed from aggregate research evidence",
    }


def component_status(v13: Mapping[str, Any]) -> dict[str, str]:
    verification = v13.get("verification", {}) or {}
    return {
        "strategy_selection": "IMPLEMENTED",
        "prediction_output_selection": "IMPLEMENTED",
        "active_information": "IMPLEMENTED",
        "prediction_update_policy": "IMPLEMENTED",
        "prediction_trajectory": "PARTIAL",
        "adaptive_compute": "IMPLEMENTED",
        "strategy_failure_memory": "IMPLEMENTED",
        "prediction_ledger": "IMPLEMENTED",
        "prediction_history_retrieval": "BLOCKED",
        "live_point_prediction": "BLOCKED",
        "pit": "VERIFIED" if verification.get("pit") is True else "HOLD",
        "leakage": "VERIFIED" if verification.get("leakage") is True else "HOLD",
        "frozen_holdout": "VERIFIED" if verification.get("frozen_holdout") is True else "HOLD",
        "robustness": "VERIFIED" if verification.get("robustness") is True else "HOLD",
        "shadow": "VERIFIED" if verification.get("shadow") is True else "HOLD",
    }


def build(v6: Mapping[str, Any], v13: Mapping[str, Any]) -> dict[str, Any]:
    horizons = v6.get("horizons", {}) or {}
    outputs = {}
    for horizon in ("5m", "10m"):
        evidence = horizons.get(horizon, {})
        if not evidence and horizon == "5m":
            evidence = v6.get("horizons", {}).get("5m", {})
        state = _state(evidence, v13)
        strategy = select_strategy(state, evidence)
        output = select_output(state)
        update = update_policy(state)
        compute = adaptive_compute(state)
        trajectory = trajectory_proxy(state)
        info = active_information(evidence)
        memory = strategy_memory(state, evidence, strategy["strategy"])
        ledger = prediction_ledger_contract(
            horizon,
            state,
            strategy["strategy"],
            output["format"],
            update["action"],
        )
        outputs[horizon] = {
            "status": "OK" if evidence.get("status") == "OK" else str(evidence.get("status", "UNKNOWN")),
            "state": state,
            "prediction_strategy": strategy,
            "prediction_output": output,
            "prediction_update": update,
            "adaptive_compute": compute,
            "prediction_trajectory": trajectory,
            "active_information": info,
            "strategy_failure_memory": memory,
            "prediction_ledger": ledger,
            "probability_safety": {
                "status": "DEFERRED",
                "reason": "no live point probability vector available in aggregate",
            },
            "component_status": component_status(v13),
        }
    return {
        "schema_version": 1,
        "experiment": "btc_ultimate_final_v13_meta_control",
        "generated_at_utc": utc_now(),
        "research_only": True,
        "production_changed": False,
        "research_mode": "EXTENSION",
        "upstream_v13_report_present": bool(v13),
        "upstream_v6_registry_present": bool(v6),
        "horizons": outputs,
        "performance_success": "NOT_ESTABLISHED_BY_META_CONTROL_ALONE",
        "policy": {
            "future_generalization_priority": True,
            "pit_fail_closed": True,
            "no_live_probability_fabrication": True,
            "no_automatic_production_promotion": True,
        },
        "limitations": [
            "Source-level incremental OOS information value is deferred unless explicit evidence is present.",
            "Probability trajectory and individual prediction revision accuracy require point-in-time live prediction rows.",
            "Historical prediction-history retrieval is schema-ready but not activated from aggregate evidence.",
        ],
    }


def write_outputs(result: Mapping[str, Any]) -> None:
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    h5 = (result.get("horizons", {}) or {}).get("5m", {})
    strategy = h5.get("prediction_strategy", {})
    trajectory = h5.get("prediction_trajectory", {})
    info = h5.get("active_information", {})
    memory = h5.get("strategy_failure_memory", {})
    STRATEGY_OUT.write_text(json.dumps(strategy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    TRAJECTORY_OUT.write_text(json.dumps(trajectory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    INFO_OUT.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MEMORY_OUT.write_text(json.dumps(memory, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    v6 = _read(V6)
    v13 = _read(V13)
    if v6 is None:
        raise SystemExit("maximum_future_generalization_v6_registry_missing")
    if v13 is None:
        raise SystemExit("ultimate_final_v13_report_missing")
    result = build(v6, v13)
    write_outputs(result)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
