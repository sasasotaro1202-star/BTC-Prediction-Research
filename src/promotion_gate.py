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


def evaluate_promotion(prod: dict[str, Any], robust: dict[str, Any], blends: dict[str, dict[str, Any]]) -> dict[str, Any]:
    robust_ok = (
        robust.get("research_only") is True
        and robust.get("policy") == "diagnostic_only_no_model_input_no_promotion_effect"
        and set(robust.get("horizons", {})) == set(HORIZONS)
        and all(robust["horizons"][h].get("status") == "ok" for h in HORIZONS)
        and all(robust["horizons"][h].get("final_holdout_protected") is True for h in HORIZONS)
    )
    production_ok = prod.get("status") == "PASS"
    candidate_ready = all(blends.get(h, {}).get("status") == "accepted" for h in HORIZONS)
    promotion_allowed = bool(production_ok and robust_ok and candidate_ready)

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
            reasons.append("candidate_not_accepted_for_both_horizons")
        reason = ";".join(reasons)

    return {
        "schema_version": 1,
        "research_only": True,
        "final_holdout_protected": True,
        "production_safety_gate": "PASS" if production_ok and robust_ok else "HOLD",
        "promotion_allowed": promotion_allowed,
        "promotion_status": status,
        "reason": reason,
        "candidate_status": {h: blends.get(h, {}).get("status", "missing") for h in HORIZONS},
        "policy": POLICY,
    }


def run(root: Path) -> dict[str, Any]:
    evidence = root / "data" / "historical_research"
    prod = json.loads((evidence / "production_integrity.json").read_text(encoding="utf-8"))
    robust = json.loads((evidence / "robustness_oos_report.json").read_text(encoding="utf-8"))
    blends = {}
    for horizon in HORIZONS:
        path = root / "models" / f"{horizon}.blend.json"
        blends[horizon] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "missing"}

    result = evaluate_promotion(prod, robust, blends)
    (evidence / "promotion_gate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = run(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    # Evidence failure is a hard stop; candidate rejection alone is a safe HOLD.
    raise SystemExit(0 if result["production_safety_gate"] == "PASS" else 1)
