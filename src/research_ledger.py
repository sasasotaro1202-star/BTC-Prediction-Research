"""Build a compact, durable ledger from one completed 9H BTC research run.

The ledger is descriptive only: it never selects, promotes, or mutates a model.
It intentionally stores compact summaries rather than large OOS matrices/artifacts.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data" / "historical_research"
OUT = EVIDENCE / "9h_latest_summary.json"

LANES = {
    "context_router": "context_router_oos.json",
    "interaction": "interaction_oos_report.json",
    "robustness": "robustness_oos_report.json",
    "adaptive_ensemble": "adaptive_ensemble_oos.json",
    "risk_aware_dynamic": "risk_aware_dynamic_oos.json",
    "rolling_challenger": "rolling_challenger_oos.json",
    "extended_features": "extended_features_oos.json",
    "model_zoo_regime": "model_zoo_regime_oos.json",
}

def _load(name: str) -> dict[str, Any]:
    path = EVIDENCE / name
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing research evidence: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"invalid research evidence {path}: {type(exc).__name__}") from exc
    if not isinstance(obj, dict):
        raise SystemExit(f"research evidence root must be object: {path}")
    return obj

def _compact_lane(obj: dict[str, Any]) -> dict[str, Any]:
    summary = obj.get("summary")
    holdout = obj.get("final_holdout")
    out: dict[str, Any] = {
        "status": obj.get("status"),
        "research_only": obj.get("research_only"),
        "production_changed": obj.get("production_changed"),
        "final_holdout_protected": obj.get("final_holdout_protected"),
        "promotion_evidence_eligible": obj.get("promotion_evidence_eligible"),
        "policy": obj.get("policy"),
    }
    if isinstance(summary, dict):
        out["summary"] = summary
    if isinstance(holdout, dict):
        out["final_holdout"] = holdout
    return out

def build() -> dict[str, Any]:
    report = _load("report.json")
    pit = _load("pit_oos_audit.json")
    research_input = _load("research_input_audit.json")
    artifact = _load("production_artifact_audit.json")
    gate = _load("promotion_gate.json")

    lanes: dict[str, Any] = {}
    for lane, filename in LANES.items():
        lanes[lane] = _compact_lane(_load(filename))

    payload = {
        "schema_version": 1,
        "ledger_type": "btc_9h_research_summary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "github_sha": os.environ.get("GITHUB_SHA", ""),
        "workflow_run_id": int(os.environ.get("GITHUB_RUN_ID", "0")),
        "workflow_run_attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "0")),
        "research": {
            "protocol_version": report.get("protocol_version"),
            "rows": report.get("rows"),
            "days": report.get("days"),
            "features_count": len(report.get("features", [])) if isinstance(report.get("features"), list) else None,
            "horizons": {
                h: {
                    "samples": report.get("horizons", {}).get(h, {}).get("samples"),
                    "class_counts": report.get("horizons", {}).get(h, {}).get("class_counts"),
                }
                for h in ("5m", "10m")
            },
        },
        "pit_oos": {
            "ok": pit.get("ok"),
            "pit_verified": pit.get("pit_verified"),
            "violation_count": pit.get("violation_count"),
            "checked_predictions": pit.get("checked_predictions"),
            "verified_primary_predictions": pit.get("verified_primary_predictions"),
            "min_strict_pit_rows": pit.get("min_strict_pit_rows"),
        },
        "research_input_audit": {
            "ok": research_input.get("ok"),
            "schema_version": research_input.get("schema_version"),
            "data_source": research_input.get("data_source"),
        },
        "production_artifact_audit": {
            "artifacts": artifact.get("artifacts"),
            "policy": artifact.get("policy"),
        },
        "promotion_gate": {
            "production_safety_gate": gate.get("production_safety_gate"),
            "promotion_allowed": gate.get("promotion_allowed"),
            "promotion_status": gate.get("promotion_status"),
            "reason": gate.get("reason"),
            "candidate_status": gate.get("candidate_status"),
            "pit_oos_verified": gate.get("pit_oos_verified"),
            "calibration_verified": gate.get("calibration_verified"),
            "research_input_audit_verified": gate.get("research_input_audit_verified"),
        },
        "lanes": lanes,
    }
    if not payload["github_sha"] or payload["workflow_run_id"] <= 0:
        raise SystemExit("research run identity is incomplete")
    return payload

def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = build()
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "path": str(OUT),
        "bytes": OUT.stat().st_size,
        "promotion_status": payload["promotion_gate"]["promotion_status"],
    }, sort_keys=True))

if __name__ == "__main__":
    main()
