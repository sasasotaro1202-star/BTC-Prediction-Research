"""Independent, fail-closed readiness audit for BTC research scope.

This is a research/readiness surface only. It never changes production models,
prediction state, or activation policy. A source being catalogued is not
evidence of PIT/OOS/production readiness.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from btc_source_frontier_catalog import SOURCES, direct_high_priority_sources, free_sources

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data" / "historical_research"
MIN_STRICT_PIT_ROWS = 300
HORIZONS = ("5m", "10m")


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing readiness evidence: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"malformed readiness evidence: {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise SystemExit(f"readiness evidence must be an object: {path}")
    return obj


def classify_state(
    *,
    production_integrity: str,
    research_health_ok: bool,
    pit_verified: bool,
    verified_primary_predictions: int,
    min_strict_pit_rows: int,
    promotion_status: str,
) -> tuple[str, list[str]]:
    if production_integrity != "PASS":
        return "BLOCKED_INTEGRITY", ["production_integrity_not_pass"]
    if not research_health_ok:
        return "BLOCKED_DATA_HEALTH", ["research_health_not_pass"]
    if not pit_verified or verified_primary_predictions < min_strict_pit_rows:
        return (
            "PIT_COLLECTION",
            [f"strict_primary_pit<{min_strict_pit_rows} ({verified_primary_predictions}/{min_strict_pit_rows})"],
        )
    if promotion_status == "ELIGIBLE_PENDING_EXPLICIT_PROMOTION":
        return "PROMOTION_REVIEW", []
    return "RESEARCH_VALIDATION", ["safety_or_candidate_evidence_not_ready"]


def source_frontier() -> dict[str, Any]:
    candidates = []
    for source in SOURCES:
        candidates.append(
            {
                "source_id": source.source_id,
                "name": source.name,
                "family": source.family,
                "access": source.access,
                "pit": source.pit,
                "realtime": source.realtime,
                "historical": source.historical,
                "key_required": source.key_required,
                "payloads": list(source.payloads),
                "priority": source.priority,
                "stage": "REGISTERED",
                "activation_allowed": False,
            }
        )
    return {
        "total_cataloged": len(candidates),
        "free_research_sources": len(free_sources()),
        "direct_high_priority_realtime_pit_sources": len(direct_high_priority_sources()),
        "production_eligible_from_catalog": 0,
        "no_implicit_activation": True,
        "candidates": candidates,
    }


def build_readiness(
    production_integrity: dict[str, Any],
    research_health: dict[str, Any],
    pit: dict[str, Any],
    promotion: dict[str, Any],
) -> dict[str, Any]:
    minimum = max(MIN_STRICT_PIT_ROWS, int(pit.get("min_strict_pit_rows", MIN_STRICT_PIT_ROWS)))
    primary = int(pit.get("verified_primary_predictions", 0))
    state, reasons = classify_state(
        production_integrity=str(production_integrity.get("status", "UNKNOWN")),
        research_health_ok=research_health.get("ok") is True,
        pit_verified=pit.get("pit_verified") is True,
        verified_primary_predictions=primary,
        min_strict_pit_rows=minimum,
        promotion_status=str(promotion.get("promotion_status", "HOLD")),
    )
    horizon_health = {}
    for horizon in HORIZONS:
        item = (research_health.get("checks") or {}).get(horizon, {})
        horizon_health[horizon] = {
            "rows": int(item.get("rows", 0)),
            "invalid_probability_rows": int(item.get("invalid_probability_rows", -1)),
            "ok": item.get("ok") is True,
        }
    return {
        "schema_version": 1,
        "status": "EVALUATED",
        "readiness_state": state,
        "reasons": reasons,
        "research_only": True,
        "production_changed": False,
        "no_implicit_activation": True,
        "strict_pit": {
            "verified": pit.get("pit_verified") is True,
            "verified_primary_predictions": primary,
            "minimum": minimum,
            "remaining": max(0, minimum - primary),
            "violation_count": int(pit.get("violation_count", -1)),
        },
        "production": {
            "integrity_status": production_integrity.get("status", "UNKNOWN"),
            "promotion_status": promotion.get("promotion_status", "UNKNOWN"),
            "promotion_allowed": promotion.get("promotion_allowed") is True,
        },
        "horizons": horizon_health,
        "source_frontier": source_frontier(),
        "policy": {
            "registered_is_not_ready": True,
            "production_requires_independent_pit_oos_calibration_evidence": True,
            "activation_requires_explicit_promotion": True,
        },
    }


def run(root: Path = ROOT) -> dict[str, Any]:
    evidence = root / "data" / "historical_research"
    result = build_readiness(
        _load(evidence / "production_integrity.json"),
        _load(evidence / "research_health.json"),
        _load(evidence / "pit_oos_audit.json"),
        _load(evidence / "promotion_gate.json"),
    )
    out = evidence / "research_readiness.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
