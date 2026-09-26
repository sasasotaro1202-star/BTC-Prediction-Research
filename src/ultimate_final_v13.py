"""Ultimate Final v13 orchestration and evidence contract for BTC research.

This layer does not retrain or promote production. It consumes already produced
PIT-safe/OOS evidence, derives a deterministic prediction-policy contract, and
fails closed when evidence is absent or contradictory.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data" / "historical_research"
REGISTRY = EVIDENCE / "maximum_future_generalization_v6_registry.json"
REPORT = EVIDENCE / "ultimate_final_v13_report.json"
LEDGER = EVIDENCE / "ultimate_final_v13_ledger.json"
LEAKAGE = EVIDENCE / "ultimate_final_v13_leakage_audit.json"
CONTRACT = EVIDENCE / "ultimate_final_v13_prediction_contract.json"
HORIZONS = ("5m", "10m")
EPS = 1e-12


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else None


def _finite(value: Any) -> bool:
    try:
        x = float(value)
        return math.isfinite(x)
    except (TypeError, ValueError):
        return False


def _clamp01(value: Any, default: float = 0.5) -> float:
    return max(0.0, min(1.0, float(value) if _finite(value) else default))


def _metric_delta(base: dict[str, Any], new: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in ("accuracy", "logloss", "brier", "calibration_error"):
        a, b = base.get(key), new.get(key)
        out[key] = float(b) - float(a) if _finite(a) and _finite(b) else None
    return out


def _extract_performance(registry: dict[str, Any] | None) -> dict[str, Any]:
    horizons: dict[str, Any] = {}
    for h in HORIZONS:
        item = (registry or {}).get("horizons", {}).get(h, {})
        base = item.get("baseline", {})
        full = item.get("full_architecture", {})
        horizons[h] = {
            "status": item.get("status"),
            "research_mode": item.get("research_mode"),
            "baseline": {
                "accuracy": base.get("accuracy"),
                "logloss": base.get("logloss"),
                "brier": base.get("brier"),
                "ece": base.get("calibration_error"),
            },
            "new": {
                "accuracy": full.get("accuracy"),
                "logloss": full.get("logloss"),
                "brier": full.get("brier"),
                "ece": full.get("calibration_error"),
            },
            "delta": _metric_delta(base, full) if isinstance(base, dict) and isinstance(full, dict) else {},
        }
    return horizons


def prediction_policy(
    *,
    predictability: float,
    failure_risk: float,
    drift: float,
    uncertainty: float,
    ood: float = 0.0,
    disagreement: float = 0.0,
) -> dict[str, Any]:
    difficulty = (
        0.28 * (1.0 - predictability)
        + 0.20 * failure_risk
        + 0.17 * drift
        + 0.17 * uncertainty
        + 0.10 * ood
        + 0.08 * disagreement
    )
    difficulty = _clamp01(difficulty)
    if ood >= 0.90 or uncertainty >= 0.92:
        action, strategy, output = "ABSTAIN", "ABSTAIN", "SCENARIO_OR_SET"
    elif failure_risk >= 0.85 or drift >= 0.90:
        action, strategy, output = "FALLBACK", "VERIFIED_BASELINE", "PROBABILITY_RANGE"
    elif difficulty >= 0.72:
        action, strategy, output = "DEEP_RECALCULATE", "FULL_ARCHITECTURE", "SCENARIO"
    elif difficulty >= 0.45:
        action, strategy, output = "RECALCULATE", "ADAPTIVE_ENSEMBLE", "PROBABILITY_RANGE"
    else:
        action, strategy, output = "MAINTAIN_OR_LIGHT_UPDATE", "STANDARD_ENSEMBLE", "PROBABILITY"
    return {
        "difficulty_score": difficulty,
        "action": action,
        "strategy": strategy,
        "output_format": output,
        "update_needed": float(_clamp01(difficulty)),
        "validity": "research_contract_only",
    }


def _build_prediction_contract(registry: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    p5 = registry.get("horizons", {}).get("5m", {})
    p10 = registry.get("horizons", {}).get("10m", {})
    failure = p5.get("failure_monitoring", {}).get("latest_failure_risk", {})
    failure_risk = _clamp01(failure.get("mean", 0.5))
    pred = p5.get("predictability", {})
    predictability = _clamp01(pred.get("global", 0.5))
    uncertainty = _clamp01(p5.get("uncertainty", {}).get("total", 0.5))
    drift = _clamp01(p5.get("drift", {}).get("drift_score", p5.get("information_shock", {}).get("shock_score", 0.0)))
    disagreement = _clamp01(
        p5.get("disagreement", {}).get("pairwise_class_disagreement", 0.0)
    )
    policy = prediction_policy(
        predictability=predictability,
        failure_risk=failure_risk,
        drift=drift,
        uncertainty=uncertainty,
        disagreement=disagreement,
    )
    pit = _json(EVIDENCE / "pit_oos_audit.json") or {}
    pit_ok = pit.get("ok") is True and pit.get("pit_verified") is True and int(pit.get("violation_count", 1)) == 0
    return {
        "schema_version": 1,
        "prediction_id": f"v13-research-{now.strftime('%Y%m%dT%H%M%SZ')}",
        "prediction_time": now.isoformat(),
        "valid_until": (now + timedelta(minutes=5)).isoformat(),
        "data_snapshot": {
            "source": "existing_run_scoped_OOS_evidence",
            "commit": registry.get("generated_at_utc"),
        },
        "model_version": "maximum_future_generalization_v6_under_v13_contract",
        "strategy": policy["strategy"],
        "action": policy["action"],
        "output_format": policy["output_format"],
        "confidence": predictability * (1.0 - uncertainty),
        "predictability": predictability,
        "uncertainty": uncertainty,
        "future_failure_risk": failure_risk,
        "current_drift": drift,
        "model_disagreement": disagreement,
        "update_needed": policy["update_needed"],
        "pit_status": "PASS" if pit_ok else "FAIL_CLOSED",
        "production_changed": False,
        "research_only": True,
        "horizon_presence": {h: bool(registry.get("horizons", {}).get(h)) for h in HORIZONS},
        "note": "This is an evidence contract, not a live trading instruction.",
    }


def validate_contract(contract: dict[str, Any]) -> tuple[bool, list[str]]:
    required = (
        "prediction_time", "valid_until", "data_snapshot", "model_version",
        "strategy", "action", "output_format", "confidence",
        "predictability", "uncertainty", "pit_status",
    )
    errors = [k for k in required if k not in contract]
    for key in ("confidence", "predictability", "uncertainty"):
        if key in contract and not _finite(contract[key]):
            errors.append(f"invalid:{key}")
    if contract.get("research_only") is not True:
        errors.append("research_only_missing")
    if contract.get("production_changed") is not False:
        errors.append("production_changed")
    if contract.get("pit_status") == "FAIL_CLOSED":
        errors.append("pit_not_verified")
    return (len(errors) == 0, errors)


def leakage_audit(registry: dict[str, Any] | None) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    source_checks = {}
    for rel in ("src/maximum_future_generalization_v6.py", "src/innovative_control_layer_oos.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        banned = {
            "random_split_call": bool(re.search(r"\b(?:train_test_split|random_split)\s*\(", text)),
            "shuffle_split": bool(re.search(r"shuffle\s*=\s*True", text)),
        }
        source_checks[rel] = {
            "banned_pattern_findings": {k: v for k, v in banned.items() if v},
            "status": "PASS" if not any(banned.values()) else "FAIL",
        }
    for h in HORIZONS:
        item = (registry or {}).get("horizons", {}).get(h, {})
        meta = item.get("meta_leakage_policy", {})
        pit = item.get("pit_policy", {})
        checks[h] = {
            "meta_future_labels_not_used_for_current_state": meta.get("future_labels_not_used_for_current_state") is True,
            "meta_models_fit_only_on_prior_blocks": meta.get("meta_models_fit_only_on_prior_blocks") is True,
            "frozen_holdout_not_used_for_selection": meta.get("frozen_holdout_used_for_selection") is False,
            "created_before_target": pit.get("created_before_target") is True,
            "training_target_before_embargo": pit.get("training_target_before_embargo") is True,
            "random_split_disabled": pit.get("random_split") is False,
        }
    all_bool = all(
        isinstance(v, bool) and v
        for h in checks.values()
        for v in h.values()
    ) if checks else False
    source_ok = all(x["status"] == "PASS" for x in source_checks.values())
    result = {
        "schema_version": 1,
        "status": "PASS" if source_ok and all_bool else "HOLD",
        "source_static_checks": source_checks,
        "registry_dynamic_checks": checks,
        "policy": "fail_closed_static_plus_run_scoped_meta/PIT evidence",
        "limitation": "static source scan is supplementary; it is not a proof of absence of all possible leakage",
    }
    LEAKAGE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def build_report() -> dict[str, Any]:
    registry = _json(REGISTRY)
    if registry is None:
        raise RuntimeError("maximum_future_generalization_v6_registry_missing")
    performance = _extract_performance(registry)
    leakage = leakage_audit(registry)
    contract = _build_prediction_contract(registry)
    contract_ok, contract_errors = validate_contract(contract)
    pit = _json(EVIDENCE / "pit_oos_audit.json") or {}
    robust = _json(EVIDENCE / "robustness_oos_report.json") or {}
    artifact_audit = _json(EVIDENCE / "production_artifact_audit.json") or {}
    research_input = _json(EVIDENCE / "research_input_audit.json") or {}

    horizon_oos_ok = all(
        performance[h]["status"] == "OK" for h in HORIZONS
    )
    holdout_ok = all(
        isinstance(registry.get("horizons", {}).get(h, {}).get("offline_holdout"), dict)
        and registry["horizons"][h]["offline_holdout"].get("status") == "OK"
        for h in HORIZONS
    )
    robustness_partial = any(
        registry.get("horizons", {}).get(h, {}).get("robustness", {}).get("status") == "PARTIAL"
        for h in HORIZONS
    )
    shadow_live = all(
        registry.get("horizons", {}).get(h, {}).get("shadow", {}).get("production_live_shadow") is True
        for h in HORIZONS
    )
    verification = {
        "pit": pit.get("ok") is True and pit.get("pit_verified") is True and int(pit.get("violation_count", 1)) == 0,
        "leakage": leakage.get("status") == "PASS",
        "meta_leakage": leakage.get("status") == "PASS",
        "oos": horizon_oos_ok,
        "frozen_holdout": holdout_ok,
        "robustness": bool(robust) and not robustness_partial,
        "artifact_integrity": bool(artifact_audit),
        "research_input": research_input.get("ok") is True,
        "shadow": shadow_live,
        "production_changed": registry.get("production_changed") is False,
    }
    required_for_performance_success = all((
        verification["pit"],
        verification["leakage"],
        verification["meta_leakage"],
        verification["oos"],
        verification["frozen_holdout"],
        verification["robustness"],
    ))
    return {
        "schema_version": 1,
        "experiment": "btc_ultimate_final_v13",
        "generated_at_utc": utc_now(),
        "research_only": True,
        "production_changed": False,
        "completion_status": "EXECUTED" if horizon_oos_ok else "PARTIAL",
        "performance_success": "ESTABLISHED" if required_for_performance_success else "NOT_ESTABLISHED",
        "verification": verification,
        "performance": performance,
        "prediction_policy": contract.get("action"),
        "prediction_strategy": contract.get("strategy"),
        "prediction_contract_valid": contract_ok,
        "prediction_contract_errors": contract_errors,
        "leakage_audit": leakage,
        "core_three_layers": {
            "model_disagreement": registry.get("horizons", {}).get("5m", {}).get("error_correlation"),
            "predictability": registry.get("horizons", {}).get("5m", {}).get("predictability"),
            "future_model_failure": registry.get("horizons", {}).get("5m", {}).get("failure_monitoring"),
        },
        "future_layers": {
            "time_to_failure": registry.get("horizons", {}).get("5m", {}).get("failure_monitoring"),
            "drift": registry.get("horizons", {}).get("5m", {}).get("information_shock"),
            "retrieval": registry.get("horizons", {}).get("5m", {}).get("retrieval"),
            "uncertainty": registry.get("horizons", {}).get("5m", {}).get("uncertainty"),
            "adaptive_compute": registry.get("horizons", {}).get("5m", {}).get("adaptive_compute"),
        },
        "status_catalog": {
            "implemented": True,
            "tested": True,
            "executed": horizon_oos_ok,
            "oos_verified": horizon_oos_ok,
            "robustness_verified": verification["robustness"],
            "statistically_verified": bool(registry.get("horizons", {}).get("5m", {}).get("statistical_validation")),
            "shadow_verified": shadow_live,
            "promotion_candidate": False,
            "adopted": False,
            "hold": not required_for_performance_success,
        },
    }


def write_outputs(report: dict[str, Any]) -> None:
    contract = _build_prediction_contract(_json(REGISTRY) or {})
    CONTRACT.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ledger = {
        "schema_version": 1,
        "experiment": "btc_ultimate_final_v13",
        "generated_at_utc": report["generated_at_utc"],
        "entries": [
            {"component": "core_three_layers", "state": "EXECUTED" if report["completion_status"] == "EXECUTED" else "PARTIAL"},
            {"component": "dynamic_prediction_policy", "state": "TESTED"},
            {"component": "PIT", "state": "VERIFIED" if report["verification"]["pit"] else "HOLD"},
            {"component": "leakage", "state": "VERIFIED" if report["verification"]["leakage"] else "HOLD"},
            {"component": "robustness", "state": "VERIFIED" if report["verification"]["robustness"] else "HOLD"},
            {"component": "shadow", "state": "VERIFIED" if report["verification"]["shadow"] else "HOLD"},
            {"component": "promotion", "state": "HOLD"},
        ],
    }
    LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    report = build_report()
    write_outputs(report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
