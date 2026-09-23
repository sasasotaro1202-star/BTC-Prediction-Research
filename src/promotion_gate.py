"""Fail-closed promotion safety evaluation for BTC research evidence.

This module is intentionally independent of production model training. It reads
already-produced evidence, writes a descriptive gate artifact, and never mutates
production model files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HORIZONS = ("5m", "10m")
POLICY = "no_production_change_without_explicit_all_horizon_candidate_acceptance_and_safety_evidence"
MIN_CALIBRATION_ROWS = 300


def evaluate_promotion(prod: dict[str, Any], robust: dict[str, Any], blends: dict[str, dict[str, Any]], pit: dict[str, Any] | None = None, calibrations: dict[str, dict[str, Any]] | None = None, research_input: dict[str, Any] | None = None) -> dict[str, Any]:
    robust_ok = (
        robust.get("research_only") is True
        and robust.get("policy") == "diagnostic_only_no_model_input_no_promotion_effect"
        and set(robust.get("horizons", {})) == set(HORIZONS)
        and all(robust["horizons"][h].get("status") == "ok" for h in HORIZONS)
        and all(robust["horizons"][h].get("final_holdout_protected") is True for h in HORIZONS)
    )
    production_ok = prod.get("status") == "PASS"
    def _candidate_holdout_ok(h: str) -> bool:
        item = blends.get(h, {}) if isinstance(blends, dict) else {}
        if item.get("status") != "accepted":
            return False
        if item.get("holdout_protected") is not True or item.get("holdout_used_for_selection") is not False:
            return False
        try:
            holdout_n = int(item.get("holdout_n", 0))
            baseline_ll = float(item["baseline_logloss"])
            candidate_ll = float(item["candidate_logloss"])
            baseline_br = float(item["baseline_brier"])
            candidate_br = float(item["candidate_brier"])
        except (KeyError, TypeError, ValueError):
            return False
        return (
            holdout_n > 0
            and all(map(lambda v: v == v and abs(v) != float("inf"), [baseline_ll, candidate_ll, baseline_br, candidate_br]))
            and candidate_ll <= baseline_ll
            and candidate_br <= baseline_br
        )

    candidate_ready = all(_candidate_holdout_ok(h) for h in HORIZONS)
    pit_ok = (
        isinstance(pit, dict)
        and pit.get("ok") is True
        and pit.get("pit_verified") is True
        and int(pit.get("violation_count", 1)) == 0
        and int(pit.get("checked_predictions", 0)) > 0
        and int(pit.get("verified_primary_predictions", 0)) >= int(pit.get("min_strict_pit_rows", 300))
    )
    calibration_ok = True
    for h in HORIZONS:
        item = calibrations.get(h, {}) if isinstance(calibrations, dict) else {}
        try:
            calibration_ok = calibration_ok and (
                item.get("horizon") == h
                and 0.5 <= float(item.get("temperature", 1.0)) <= 3.0
                and bool(item.get("model_version"))
                and int(item.get("n_settled", 0)) >= MIN_CALIBRATION_ROWS
                and item.get("fit_logloss") is not None
                and item.get("holdout_logloss") is not None
            )
        except (TypeError, ValueError):
            calibration_ok = False
    research_input_ok = isinstance(research_input, dict) and research_input.get("ok") is True
    safety_ok = bool(production_ok and robust_ok and pit_ok and calibration_ok and research_input_ok)
    promotion_allowed = bool(safety_ok and candidate_ready)

    if promotion_allowed:
        status = "ELIGIBLE_PENDING_EXPLICIT_PROMOTION"
        reason = "all safety evidence passed and both horizon candidates are explicitly accepted"
    else:
        status = "HOLD"
        reasons = []
        if not production_ok:
            reasons.append("production_integrity_not_pass")
        if not robust_ok:
            reasons.append("robustness_evidence_invalid_or_incomplete")
        if not candidate_ready:
            reasons.append("candidate_or_frozen_holdout_non_regression_not_verified")
        if not pit_ok:
            reasons.append("pit_oos_audit_not_fully_verified")
        if not calibration_ok:
            reasons.append("calibration_evidence_invalid_or_missing")
        if not research_input_ok:
            reasons.append("research_input_audit_not_pass")
        reason = ";".join(reasons)

    return {
        "schema_version": 1,
        "research_only": True,
        "final_holdout_protected": True,
        "production_safety_gate": "PASS" if safety_ok else "HOLD",
        "promotion_allowed": promotion_allowed,
        "promotion_status": status,
        "reason": reason,
        "candidate_status": {h: blends.get(h, {}).get("status", "missing") for h in HORIZONS},
        "pit_oos_verified": pit_ok,
        "calibration_verified": calibration_ok,
        "research_input_audit_verified": research_input_ok,
        "policy": POLICY,
    }


def _production_integrity_from_evidence(evidence: Path) -> dict[str, Any]:
    """Construct a fail-closed production safety verdict from artifacts produced in this run.
    
    Older runners may not materialize production_integrity.json. In that case,
    derive the verdict only from independently validated artifact and PIT reports.
    Never infer PASS from file existence alone.
    """
    explicit = evidence / "production_integrity.json"
    if explicit.exists():
        obj = json.loads(explicit.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise SystemExit("production_integrity.json must contain an object")
        return obj

    artifact_path = evidence / "production_artifact_audit.json"
    pit_path = evidence / "pit_oos_audit.json"
    if not artifact_path.exists() or not pit_path.exists():
        raise SystemExit("production safety evidence missing: artifact audit and PIT/OOS audit are required")

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    pit = json.loads(pit_path.read_text(encoding="utf-8"))
    artifacts = artifact.get("artifacts")
    artifact_ok = (
        isinstance(artifacts, list)
        and len(artifacts) == len(HORIZONS)
        and all(
            item.get("horizon") in HORIZONS
            and item.get("runtime_model_reload_ok") is True
            and len(str(item.get("model_sha256", ""))) == 64
            and len(str(item.get("metadata_sha256", ""))) == 64
            and item.get("runtime_classes") == ["DOWN", "FLAT", "UP"]
            and int(item.get("runtime_feature_count", -1)) > 0
            for item in artifacts
        )
    )
    pit_ok = (
        pit.get("ok") is True
        and pit.get("pit_verified") is True
        and int(pit.get("violation_count", 1)) == 0
        and int(pit.get("verified_primary_predictions", 0)) >= int(pit.get("min_strict_pit_rows", 300))
    )
    return {
        "schema_version": 1,
        "status": "PASS" if artifact_ok and pit_ok else "HOLD",
        "derived": True,
        "artifact_audit_ok": artifact_ok,
        "pit_oos_ok": pit_ok,
        "artifact_audit_source": str(artifact_path.name),
        "pit_oos_source": str(pit_path.name),
    }


def run(root: Path) -> dict[str, Any]:
    evidence = root / "data" / "historical_research"
    prod = _production_integrity_from_evidence(evidence)
    robust = json.loads((evidence / "robustness_oos_report.json").read_text(encoding="utf-8"))
    pit_path = evidence / "pit_oos_audit.json"
    pit = json.loads(pit_path.read_text(encoding="utf-8")) if pit_path.exists() else None
    research_input_path = evidence / "research_input_audit.json"
    research_input = json.loads(research_input_path.read_text(encoding="utf-8")) if research_input_path.exists() else None

    calibrations = {}
    for horizon in HORIZONS:
        cpath = root / "models" / f"{horizon}.calibration.json"
        calibrations[horizon] = json.loads(cpath.read_text(encoding="utf-8")) if cpath.exists() else {}
    blends = {}
    for horizon in HORIZONS:
        path = root / "models" / f"{horizon}.blend.json"
        blends[horizon] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "missing"}

    result = evaluate_promotion(prod, robust, blends, pit, calibrations, research_input)
    (evidence / "promotion_gate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = run(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    # A valid HOLD is a safe, expected gate outcome: research may continue while
    # promotion remains blocked. Missing/malformed evidence raises inside run().
    raise SystemExit(0)
