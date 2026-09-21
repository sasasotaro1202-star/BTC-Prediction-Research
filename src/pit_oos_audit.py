"""Audit prediction records for point-in-time and out-of-sample integrity."""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from db import DB, init_db

OUT = Path(DB).parent / "historical_research" / "pit_oos_audit.json"
MAX_FUTURE_SKEW_SECONDS = 60
MIN_PREDICTIONS = 1


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def validate_provenance_envelope(record: dict, prefix: str) -> list[str]:
    """Validate available-at provenance without inventing unknown source timestamps."""
    violations: list[str] = []
    if not isinstance(record, dict):
        return [f"{prefix}:provenance_not_object"]
    required = ("available_at", "retrieved_at", "prediction_cutoff")
    parsed = {}
    for key in required:
        if key not in record or record.get(key) in (None, ""):
            violations.append(f"{prefix}:missing_{key}")
            continue
        try:
            parsed[key] = parse_utc(str(record[key]))
        except Exception:
            violations.append(f"{prefix}:invalid_{key}")
    if violations:
        return violations
    available = parsed["available_at"]
    retrieved = parsed["retrieved_at"]
    cutoff = parsed["prediction_cutoff"]
    if available > retrieved:
        violations.append(f"{prefix}:available_at_after_retrieved_at")
    if retrieved > cutoff:
        violations.append(f"{prefix}:retrieved_at_after_prediction_cutoff")
    if available > cutoff:
        violations.append(f"{prefix}:available_at_after_prediction_cutoff")
    for key in ("event_time", "publication_time", "revision_time"):
        value = record.get(key)
        if value in (None, ""):
            continue
        try:
            parsed[key] = parse_utc(str(value))
        except Exception:
            violations.append(f"{prefix}:invalid_{key}")
    if "event_time" in parsed and parsed["event_time"] > available:
        violations.append(f"{prefix}:event_time_after_available_at")
    if "publication_time" in parsed and parsed["publication_time"] > available:
        violations.append(f"{prefix}:publication_time_after_available_at")
    if "revision_time" in parsed and "publication_time" in parsed and parsed["revision_time"] < parsed["publication_time"]:
        violations.append(f"{prefix}:revision_time_before_publication_time")
    return violations

def audit() -> dict:
    # Fail closed: an empty/missing state must never be reported as a clean audit.
    db_path = Path(DB)
    if not db_path.exists() or db_path.stat().st_size == 0:
        result = {
            "ok": False,
            "checked_predictions": 0,
            "violations": ["prediction_database_missing_or_empty"],
            "violation_count": 1,
            "policy": "decision_time_before_targets; market_inputs_not_after_decision_time; degraded_mode_cannot_claim_direction",
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    init_db()
    checked = 0
    violations: list[str] = []
    legacy_unverified_count = 0
    verified_count = 0
    with sqlite3.connect(DB) as con:
        rows = con.execute(
            "SELECT prediction_id, created_at_utc, target_5m, target_10m, model_version, scenario_json "
            "FROM predictions ORDER BY created_at_utc, prediction_id"
        ).fetchall()

    if len(rows) < MIN_PREDICTIONS:
        violations.append("prediction_database_has_no_predictions")

    now = datetime.now(timezone.utc)
    for prediction_id, created_raw, target5_raw, target10_raw, model_version, scenario_raw in rows:
        checked += 1
        try:
            created = parse_utc(created_raw)
            target5 = parse_utc(target5_raw)
            target10 = parse_utc(target10_raw)
        except Exception:
            violations.append(f"{prediction_id}:invalid_timestamp")
            continue
        if created > now:
            skew = (created - now).total_seconds()
            if skew > MAX_FUTURE_SKEW_SECONDS:
                violations.append(f"{prediction_id}:prediction_time_in_future")
        if target10 <= target5:
            violations.append(f"{prediction_id}:10m_target_not_after_5m_target")
        try:
            scenario = json.loads(scenario_raw or "{}")
        except Exception:
            violations.append(f"{prediction_id}:invalid_scenario_json")
            continue
        decision_raw = scenario.get("decision_time_utc")
        provisional_provenance = scenario.get("provenance")
        if not decision_raw and isinstance(provisional_provenance, dict):
            decision_raw = provisional_provenance.get("prediction_cutoff")
        decision = created
        if decision_raw:
            try:
                decision = parse_utc(str(decision_raw))
                if abs((decision - created).total_seconds()) > MAX_FUTURE_SKEW_SECONDS:
                    violations.append(f"{prediction_id}:decision_time_mismatch")
                if target5 <= decision:
                    violations.append(f"{prediction_id}:5m_target_not_after_decision")
                if target10 <= decision:
                    violations.append(f"{prediction_id}:10m_target_not_after_decision")
                cutoff_raw = scenario.get("market_data_cutoff_utc")
                if cutoff_raw:
                    cutoff = parse_utc(str(cutoff_raw))
                    if cutoff > decision:
                        violations.append(f"{prediction_id}:market_cutoff_after_decision")
            except Exception:
                violations.append(f"{prediction_id}:invalid_pit_metadata")
                decision = created

        # Enforce the stronger PIT contract when provenance is present:
        # every explicitly available input must have become available no later
        # than the prediction/decision timestamp. Missing source-native
        # publication timestamps remain acceptable only when the source adapter
        # explicitly records a conservative acquisition-time available_at.
        try:
            provenance = scenario.get("provenance", {})
            if provenance and not isinstance(provenance, dict):
                raise ValueError("provenance_not_object")
            top_available = provenance.get("available_at") if isinstance(provenance, dict) else None
            if top_available:
                available = parse_utc(str(top_available))
                if available > decision:
                    violations.append(f"{prediction_id}:available_at_after_decision")
            sources = provenance.get("sources", {}) if isinstance(provenance, dict) else {}
            if sources and not isinstance(sources, dict):
                raise ValueError("sources_not_object")
            for source_name, source_info in sources.items():
                if not isinstance(source_info, dict):
                    violations.append(f"{prediction_id}:invalid_source_provenance:{source_name}")
                    continue
                source_status = str(source_info.get("status", ""))
                source_available = source_info.get("available_at")
                if source_status in {"ok", "ok_current_only"} and not source_available:
                    violations.append(f"{prediction_id}:missing_available_at:{source_name}")
                    continue
                if source_available:
                    source_dt = parse_utc(str(source_available))
                    if source_dt > decision:
                        violations.append(f"{prediction_id}:source_available_at_after_decision:{source_name}")
        except Exception:
            violations.append(f"{prediction_id}:invalid_pit_provenance")

        # Degraded predictions must always satisfy the explicit safety policy,
        # including legacy rows. Check this before legacy provenance handling so
        # an old degraded record cannot bypass the policy contract.
        if model_version == "DEGRADED_NO_FRESH_DATA" and scenario.get("policy") != "safe_degraded_no_directional_claim":
            violations.append(f"{prediction_id}:degraded_policy_mismatch")
        provenance = scenario.get("provenance")
        if provenance is None:
            # Old prediction rows predate the provenance contract. They cannot be
            # promoted to PIT-verified merely by inference, but they are not the
            # same thing as a current structural PIT violation. Keep them visible
            # as UNVERIFIED_LEGACY and require a separate zero-legacy condition
            # before promotion.
            if any(k in scenario for k in ("decision_time_utc", "market_data_cutoff_utc")):
                violations.append(f"{prediction_id}:missing_top_level_provenance")
            else:
                legacy_unverified_count += 1
                continue
        else:
            violations.extend(validate_provenance_envelope(provenance, f"{prediction_id}:provenance"))
            sources = provenance.get("sources") if isinstance(provenance, dict) else None
            if not isinstance(sources, dict) or not sources:
                violations.append(f"{prediction_id}:missing_source_provenance")
            else:
                for source_name, source_record in sources.items():
                    violations.extend(validate_provenance_envelope(source_record, f"{prediction_id}:source:{source_name}"))
        if provenance is not None and not any(v.startswith(f"{prediction_id}:") for v in violations):
            verified_count += 1

    result = {
        "ok": not violations,
        "status": "PASS" if not violations and legacy_unverified_count == 0 else (
            "PASS_WITH_LEGACY_UNVERIFIED" if not violations else "FAILED"
        ),
        "pit_verified": bool(not violations and legacy_unverified_count == 0),
        "checked_predictions": checked,
        "verified_predictions": verified_count,
        "legacy_unverified_count": legacy_unverified_count,
        "violations": violations[:100],
        "violation_count": len(violations),
        "policy": "decision_time_before_targets; market_inputs_not_after_decision_time; legacy_rows_without_provenance_remain_unverified_and_cannot_enable_promotion",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    result = audit()
    print(json.dumps(result, ensure_ascii=False))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
