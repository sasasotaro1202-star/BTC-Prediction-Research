"""Independent, fail-closed readiness audit for BTC research scope.

This is a research/readiness surface only. It never changes production models,
prediction state, or activation policy. A source being catalogued is not
evidence of PIT/OOS/production readiness.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from btc_source_frontier_catalog import SOURCES, direct_high_priority_sources, free_sources

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data" / "historical_research"
MIN_STRICT_PIT_ROWS = 300
MIN_EFFECTIVE_CALIBRATION_ROWS = 400
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
    primary_horizon_gate: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    if production_integrity != "PASS":
        return "BLOCKED_INTEGRITY", ["production_integrity_not_pass"]
    if not research_health_ok:
        return "BLOCKED_DATA_HEALTH", ["research_health_not_pass"]
    if not pit_verified:
        return (
            "PIT_COLLECTION",
            [f"strict_primary_pit<{min_strict_pit_rows} ({verified_primary_predictions}/{min_strict_pit_rows})"],
        )
    if not isinstance(primary_horizon_gate, dict) or set(primary_horizon_gate) != set(HORIZONS):
        return "PIT_COLLECTION", ["per_horizon_strict_pit_gate_missing_or_invalid"]
    horizon_failures = []
    for horizon in HORIZONS:
        item = primary_horizon_gate.get(horizon)
        if not isinstance(item, dict):
            horizon_failures.append(f"{horizon}:missing")
            continue
        try:
            settled = int(item.get("strict_primary_settled", -1))
            item_minimum = int(item.get("minimum", min_strict_pit_rows))
        except (TypeError, ValueError):
            horizon_failures.append(f"{horizon}:invalid_counts")
            continue
        if settled < min_strict_pit_rows or item_minimum < min_strict_pit_rows or item.get("ready") is not True:
            horizon_failures.append(f"{horizon}:{settled}/{min_strict_pit_rows}")
    if horizon_failures:
        return "PIT_COLLECTION", ["per_horizon_strict_pit_gate_failed:" + ",".join(horizon_failures)]
    if verified_primary_predictions < min_strict_pit_rows:
        return (
            "PIT_COLLECTION",
            [f"strict_primary_pit<{min_strict_pit_rows} ({verified_primary_predictions}/{min_strict_pit_rows})"],
        )
    if promotion_status == "ELIGIBLE_PENDING_EXPLICIT_PROMOTION":
        return "PROMOTION_REVIEW", []
    return "RESEARCH_VALIDATION", ["safety_or_candidate_evidence_not_ready"]


def _calibration_state(root: Path) -> dict[str, dict[str, Any]]:
    """Summarize current-generation calibration readiness without changing calibration."""
    result: dict[str, dict[str, Any]] = {}
    for horizon in HORIZONS:
        path = root / "models" / f"{horizon}.calibration.json"
        if not path.is_file():
            result[horizon] = {
                "status": "MISSING",
                "n_settled": 0,
                "minimum_effective_rows": MIN_EFFECTIVE_CALIBRATION_ROWS,
                "remaining_rows": MIN_EFFECTIVE_CALIBRATION_ROWS,
                "fit_logloss_available": False,
                "holdout_logloss_available": False,
            }
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                raise ValueError("calibration_artifact_not_object")
            n = int(obj.get("n_settled", 0))
            fit_ok = obj.get("fit_logloss") is not None
            holdout_ok = obj.get("holdout_logloss") is not None
            version = str(obj.get("model_version", "")).strip()

            model_meta_path = root / "models" / f"{horizon}.json"
            model_artifact_path = root / "models" / f"{horizon}.joblib"
            expected_version = ""
            expected_hash = None
            if model_meta_path.is_file() and model_artifact_path.is_file():
                model_meta = json.loads(model_meta_path.read_text(encoding="utf-8"))
                expected_version = str(model_meta.get("model_version", "")).strip()
                digest = hashlib.sha256()
                with model_artifact_path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                expected_hash = digest.hexdigest()

            binding_ok = (
                bool(version)
                and bool(expected_version)
                and version == expected_version
                and obj.get("model_sha256") == expected_hash
            )
            status = (
                "READY"
                if (
                    n >= MIN_EFFECTIVE_CALIBRATION_ROWS
                    and fit_ok
                    and holdout_ok
                    and binding_ok
                )
                else "WAITING"
            )
            result[horizon] = {
                "status": status,
                "n_settled": n,
                "minimum_effective_rows": MIN_EFFECTIVE_CALIBRATION_ROWS,
                "remaining_rows": max(0, MIN_EFFECTIVE_CALIBRATION_ROWS - n),
                "fit_logloss_available": fit_ok,
                "holdout_logloss_available": holdout_ok,
                "model_version": version,
                "expected_model_version": expected_version,
                "model_sha256_match": bool(expected_hash) and obj.get("model_sha256") == expected_hash,
                "binding_ok": binding_ok,
                "artifact": path.name,
            }
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            result[horizon] = {
                "status": "INVALID",
                "n_settled": 0,
                "minimum_effective_rows": MIN_EFFECTIVE_CALIBRATION_ROWS,
                "remaining_rows": MIN_EFFECTIVE_CALIBRATION_ROWS,
                "fit_logloss_available": False,
                "holdout_logloss_available": False,
                "error": type(exc).__name__,
                "artifact": path.name,
            }
    return result


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
    calibration: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    minimum = max(MIN_STRICT_PIT_ROWS, int(pit.get("min_strict_pit_rows", MIN_STRICT_PIT_ROWS)))
    primary = int(pit.get("verified_primary_predictions", 0))
    calibration = calibration if isinstance(calibration, dict) else {}
    calibration_ready = bool(calibration) and all(
        isinstance(calibration.get(h), dict) and calibration[h].get("status") == "READY"
        for h in HORIZONS
    )
    state, reasons = classify_state(
        production_integrity=str(production_integrity.get("status", "UNKNOWN")),
        research_health_ok=research_health.get("ok") is True,
        pit_verified=pit.get("pit_verified") is True,
        verified_primary_predictions=primary,
        min_strict_pit_rows=minimum,
        promotion_status=str(promotion.get("promotion_status", "HOLD")),
        primary_horizon_gate=pit.get("primary_horizon_gate"),
    )
    if state == "RESEARCH_VALIDATION" and not calibration_ready:
        state = "CALIBRATION_COLLECTION"
        reasons = ["calibration_evidence_not_ready"]

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
            "per_horizon_gate": pit.get("primary_horizon_gate"),
        },
        "calibration": {
            "all_horizons_ready": calibration_ready,
            "effective_minimum_rows": MIN_EFFECTIVE_CALIBRATION_ROWS,
            "horizons": calibration,
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
        _calibration_state(root),
    )
    out = evidence / "research_readiness.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
