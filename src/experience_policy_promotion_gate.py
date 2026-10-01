"""Fail-closed promotion gate for experience-derived prediction policies.

The gate never mutates production. It converts research OOS evidence into an
explicit HOLD / PROMOTION_CANDIDATE / REJECTED decision while requiring fresh
strict PIT evidence and an independent evaluation boundary before promotion.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = {
    "schema_version": 1,
    "min_relative_logloss_gain": 0.03,
    "min_relative_brier_gain": 0.01,
    "min_prequential_test_rows": 300,
    "require_pit_verified": True,
    "max_pit_audit_age_sec": 3600,
    "require_independent_holdout": True,
    "candidate": "hierarchical_experience_memory",
}


def _load(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def _pit_freshness(path: Path, max_age: int) -> dict[str, Any]:
    obj = _load(path)
    generated = obj.get("generated_at_utc")
    try:
        parsed = datetime.fromisoformat(str(generated).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("pit_audit_timestamp_naive")
        age = max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return {"fresh": False, "status": "INVALID", "age_sec": None}
    fresh = age <= max_age
    return {"fresh": fresh, "status": "FRESH" if fresh else "STALE", "age_sec": round(age, 3)}


def _relative_gain(baseline: float, candidate: float) -> float:
    denom = max(abs(float(baseline)), 1e-12)
    return (float(baseline) - float(candidate)) / denom


def evaluate(
    research: dict[str, Any],
    pit_audit: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)

    decision = "PROMOTION_CANDIDATE"
    reasons: list[str] = []
    horizons = research.get("horizons")
    if not isinstance(horizons, dict) or set(horizons) != {"5m", "10m"}:
        return {
            "schema_version": 1,
            "status": "HOLD",
            "decision": "HOLD",
            "research_only": True,
            "production_changed": False,
            "promotion_evidence_eligible": False,
            "reasons": ["experience_policy_horizons_incomplete"],
        }

    pit_verified = bool(pit_audit.get("pit_verified"))
    if bool(cfg["require_pit_verified"]) and not pit_verified:
        decision = "HOLD"
        reasons.append("strict_pit_not_verified")

    pit_generated = pit_audit.get("generated_at_utc")
    freshness = _pit_freshness_from_obj(pit_audit, int(cfg["max_pit_audit_age_sec"]))
    if not freshness["fresh"]:
        decision = "HOLD"
        reasons.append("pit_audit_not_fresh")

    per_horizon: dict[str, Any] = {}
    for horizon in ("5m", "10m"):
        item = horizons[horizon]
        if item.get("status") != "OK":
            decision = "HOLD"
            reasons.append(f"{horizon}:experience_policy_status_not_ok")
            per_horizon[horizon] = {"status": "BLOCKED"}
            continue
        n = int(item.get("prequential_test_rows", 0) or 0)
        baseline = item.get("baseline_error_probability") or {}
        candidate = item.get("hierarchical_experience_memory") or {}
        ll_gain = _relative_gain(baseline.get("logloss"), candidate.get("logloss"))
        br_gain = _relative_gain(baseline.get("brier"), candidate.get("brier"))
        adequate_n = n >= int(cfg["min_prequential_test_rows"])
        metric_pass = ll_gain >= float(cfg["min_relative_logloss_gain"]) and br_gain >= float(cfg["min_relative_brier_gain"])
        per_horizon[horizon] = {
            "status": "OK",
            "prequential_test_rows": n,
            "relative_logloss_gain": ll_gain,
            "relative_brier_gain": br_gain,
            "metric_pass": metric_pass,
            "sample_pass": adequate_n,
        }
        if not adequate_n:
            decision = "HOLD"
            reasons.append(f"{horizon}:insufficient_prequential_rows")
        if not metric_pass:
            decision = "REJECTED"
            reasons.append(f"{horizon}:candidate_does_not_meet_metric_gate")

    if bool(cfg["require_independent_holdout"]):
        decision = "HOLD" if decision != "REJECTED" else decision
        reasons.append("independent_holdout_evidence_required")

    return {
        "schema_version": 1,
        "status": "HOLD" if decision == "HOLD" else ("REJECTED" if decision == "REJECTED" else "PROMOTION_CANDIDATE"),
        "decision": decision,
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "candidate": str(cfg["candidate"]),
        "config": cfg,
        "pit": {
            "pit_verified": pit_verified,
            "generated_at_utc": pit_generated,
            "fresh": freshness["fresh"],
            "age_sec": freshness["age_sec"],
            "status": freshness["status"],
        },
        "horizons": per_horizon,
        "reasons": reasons,
    }


def _pit_freshness_from_obj(obj: dict[str, Any], max_age: int) -> dict[str, Any]:
    generated = obj.get("generated_at_utc")
    try:
        parsed = datetime.fromisoformat(str(generated).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("pit_audit_timestamp_naive")
        age = max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return {"fresh": False, "status": "INVALID", "age_sec": None}
    fresh = age <= max_age
    return {"fresh": fresh, "status": "FRESH" if fresh else "STALE", "age_sec": round(age, 3)}


def build(
    research_path: Path = ROOT / "data/historical_research/experience_policy_oos.json",
    pit_path: Path = ROOT / "data/historical_research/pit_oos_audit.json",
    config_path: Path = ROOT / "config/EXPERIENCE_POLICY_PROMOTION_GATE.json",
    output_path: Path = ROOT / "data/historical_research/experience_policy_promotion_gate.json",
) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    loaded_cfg = _load(config_path)
    cfg.update(loaded_cfg)
    research = _load(research_path)
    pit = _load(pit_path)
    if not research:
        report = {
            "schema_version": 1,
            "status": "HOLD",
            "decision": "HOLD",
            "research_only": True,
            "production_changed": False,
            "promotion_evidence_eligible": False,
            "reasons": ["missing_experience_policy_oos"],
        }
    elif not pit:
        report = {
            "schema_version": 1,
            "status": "HOLD",
            "decision": "HOLD",
            "research_only": True,
            "production_changed": False,
            "promotion_evidence_eligible": False,
            "reasons": ["missing_pit_audit"],
        }
    else:
        report = evaluate(research, pit, cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False))
