"""Audit prediction records for point-in-time and out-of-sample integrity.

Legacy prediction rows that predate the provenance contract remain visible and
unverified, but they do not permanently block future promotion. Only rows with
the explicit PIT contract participate in the strict promotion evidence scope.
"""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from db import DB, init_db

OUT = Path(DB).parent / "historical_research" / "pit_oos_audit.json"
MAX_FUTURE_SKEW_SECONDS = 60
MIN_PREDICTIONS = 1
MIN_STRICT_PIT_ROWS = 300
AVAILABLE_STATUSES = {"ok", "ok_current_only"}


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
    db_path = Path(DB)
    if not db_path.exists() or db_path.stat().st_size == 0:
        result = {
            "ok": False,
            "status": "FAILED",
            "pit_verified": False,
            "checked_predictions": 0,
            "verified_predictions": 0,
            "legacy_unverified_count": 0,
            "violations": ["prediction_database_missing_or_empty"],
            "violation_count": 1,
            "policy": "strict_pit_scope_with_legacy_unverified_quarantine",
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    init_db()
    checked = 0
    violations: list[str] = []
    legacy_violations: list[str] = []
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
        row_prefix = f"{prediction_id}:"
        try:
            created = parse_utc(created_raw)
            target5 = parse_utc(target5_raw)
            target10 = parse_utc(target10_raw)
        except Exception:
            violations.append(f"{prediction_id}:invalid_timestamp")
            continue

        if created > now and (created - now).total_seconds() > MAX_FUTURE_SKEW_SECONDS:
            violations.append(f"{prediction_id}:prediction_time_in_future")

        try:
            scenario = json.loads(scenario_raw or "{}")
        except Exception:
            violations.append(f"{prediction_id}:invalid_scenario_json")
            continue

        provenance = scenario.get("provenance")
        provenance_present = isinstance(provenance, dict)
        is_legacy = (
            not provenance_present
            and "decision_time_utc" not in scenario
            and "market_data_cutoff_utc" not in scenario
        )
        scoped = legacy_violations if is_legacy else violations

        decision_raw = scenario.get("decision_time_utc")
        if not decision_raw and provenance_present:
            decision_raw = provenance.get("prediction_cutoff")

        decision = created
        if decision_raw:
            try:
                decision = parse_utc(str(decision_raw))
                if abs((decision - created).total_seconds()) > MAX_FUTURE_SKEW_SECONDS:
                    scoped.append(f"{prediction_id}:decision_time_mismatch")
                cutoff_raw = scenario.get("market_data_cutoff_utc")
                if cutoff_raw:
                    cutoff = parse_utc(str(cutoff_raw))
                    if cutoff > decision:
                        scoped.append(f"{prediction_id}:market_cutoff_after_decision")
            except Exception:
                scoped.append(f"{prediction_id}:invalid_pit_metadata")
                decision = created

        # Target ordering/chronology is a hard structural invariant even for
        # legacy rows. Legacy provenance may be unverified, but an invalid future
        # target makes the prediction itself unusable and must remain a failure.
        if target10 <= target5:
            violations.append(f"{prediction_id}:10m_target_not_after_5m_target")
        if target5 <= decision:
            violations.append(f"{prediction_id}:5m_target_not_after_decision")
        if target10 <= decision:
            violations.append(f"{prediction_id}:10m_target_not_after_decision")

        if provenance_present:
            scoped.extend(validate_provenance_envelope(provenance, f"{prediction_id}:provenance"))
            sources = provenance.get("sources")
            if not isinstance(sources, dict) or not sources:
                scoped.append(f"{prediction_id}:missing_source_provenance")
            else:
                for source_name, source_record in sources.items():
                    if not isinstance(source_record, dict):
                        scoped.append(f"{prediction_id}:source:{source_name}:provenance_not_object")
                        continue
                    source_status = str(source_record.get("status", ""))
                    if source_status in AVAILABLE_STATUSES:
                        scoped.extend(
                            validate_provenance_envelope(
                                source_record,
                                f"{prediction_id}:source:{source_name}",
                            )
                        )
        elif not is_legacy:
            scoped.append(f"{prediction_id}:missing_top_level_provenance")

        if model_version == "DEGRADED_NO_FRESH_DATA":
            if scenario.get("policy") != "safe_degraded_no_directional_claim":
                violations.append(f"{prediction_id}:degraded_policy_mismatch")

        if is_legacy:
            legacy_unverified_count += 1
        elif not any(v.startswith(row_prefix) for v in violations):
            verified_count += 1

    # Promotion evidence is tied to the production benchmark venue. Fallback
    # observations remain useful research data but cannot satisfy the primary
    # PIT requirement.
    pit_ready = bool(
        not violations
        and verified_primary_count >= MIN_STRICT_PIT_ROWS
    )
    result = {
        "ok": not violations,
        "status": (
            "PASS"
            if pit_ready
            else ("PASS_WITH_LEGACY_UNVERIFIED" if not violations else "FAILED")
        ),
        "pit_verified": pit_ready,
        "pit_verified_reason": (
            "strict_binance_primary_scope_verified"
            if pit_ready
            else f"insufficient_binance_primary_pit_rows:{verified_primary_count}/{MIN_STRICT_PIT_ROWS}"
        ),
        "checked_predictions": checked,
        "verified_predictions": verified_count,
        "verified_primary_predictions": verified_primary_count,
        "verified_fallback_predictions": verified_fallback_count,
        "min_strict_pit_rows": MIN_STRICT_PIT_ROWS,
        "legacy_unverified_count": legacy_unverified_count,
        "legacy_violation_count": len(legacy_violations),
        "legacy_violations": legacy_violations[:50],
        "violations": violations[:100],
        "violation_count": len(violations),
        "policy": "strict_pit_scope_with_legacy_unverified_quarantine",
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
