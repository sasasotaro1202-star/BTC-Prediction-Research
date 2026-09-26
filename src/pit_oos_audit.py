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
from pit_history import record_pit_history

OUT = Path(DB).parent / "historical_research" / "pit_oos_audit.json"
MAX_FUTURE_SKEW_SECONDS = 60
MIN_PREDICTIONS = 1
MIN_STRICT_PIT_ROWS = 300
SITUATION_META_MIN_ROWS = 3000
ONLINE_EXPERT_MIN_ROWS = 140
AVAILABLE_STATUSES = {"ok", "ok_current_only"}
# a4d43aa added per-source prediction_cutoff to live Coinbase provenance.
# Rows created before that contract existed are quarantined, not upgraded retroactively.
LEGACY_COINBASE_CUTOFF_UTC = datetime.fromisoformat("2026-09-22T05:04:26+00:00")
LEGACY_COINBASE_MODEL_VERSION = "5m:coinbase_fallback.rf.v1|10m:coinbase_fallback.rf.v1"


def table_columns(con, table: str) -> set[str]:
    return {row[1] for row in con.execute(f"PRAGMA table_info({table})").fetchall()}


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
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            result["history"] = record_pit_history(result, OUT.parent / "pit_history")
        except Exception as exc:
            result["history"] = {"status": "FAILED", "error": f"{type(exc).__name__}:{exc}"}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    init_db()
    checked = 0
    violations: list[str] = []
    legacy_violations: list[str] = []
    legacy_unverified_count = 0
    verified_count = 0
    verified_primary_count = 0
    verified_fallback_count = 0

    with sqlite3.connect(DB) as con:
        columns = table_columns(con, "predictions")
        actual5 = "actual_direction_5m" if "actual_direction_5m" in columns else "NULL AS actual_direction_5m"
        actual10 = "actual_direction_10m" if "actual_direction_10m" in columns else "NULL AS actual_direction_10m"
        rows = con.execute(
            "SELECT prediction_id, created_at_utc, target_5m, target_10m, model_version, "
            f"{actual5}, {actual10}, scenario_json "
            "FROM predictions ORDER BY created_at_utc, prediction_id"
        ).fetchall()

    if len(rows) < MIN_PREDICTIONS:
        violations.append("prediction_database_has_no_predictions")

    now = datetime.now(timezone.utc)
    coverage = {
        "5m": {
            "settled_predictions": 0,
            "strict_primary_settled": 0,
            "situation_meta_ready": 0,
            "online_expert_ready": 0,
        },
        "10m": {
            "settled_predictions": 0,
            "strict_primary_settled": 0,
            "situation_meta_ready": 0,
            "online_expert_ready": 0,
        },
    }

    for (
        prediction_id,
        created_raw,
        target5_raw,
        target10_raw,
        model_version,
        actual5,
        actual10,
        scenario_raw,
    ) in rows:
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
        coinbase_source = (
            provenance.get("sources", {}).get("coinbase_futures", {})
            if provenance_present and isinstance(provenance.get("sources"), dict)
            else {}
        )
        legacy_coinbase_contract = (
            model_version == LEGACY_COINBASE_MODEL_VERSION
            and created < LEGACY_COINBASE_CUTOFF_UTC
            and isinstance(coinbase_source, dict)
            and not coinbase_source.get("prediction_cutoff")
        )
        scoped = legacy_violations if (is_legacy or legacy_coinbase_contract) else violations

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

        # Horizon ordering is always a hard structural invariant.
        if target10 <= target5:
            violations.append(f"{prediction_id}:10m_target_not_after_5m_target")
        # Pre-contract Coinbase fallback rows used the post-fetch top-level cutoff
        # as a legacy decision boundary, which can legitimately sit after target_5m.
        # Quarantine only that exact known pre-contract cohort; all post-contract
        # rows remain fail-closed.
        target5_decision_violation = target5 <= decision
        target10_decision_violation = target10 <= decision
        target_scope = legacy_violations if legacy_coinbase_contract else violations
        if target5_decision_violation:
            target_scope.append(f"{prediction_id}:5m_target_not_after_decision")
        if target10_decision_violation:
            target_scope.append(f"{prediction_id}:10m_target_not_after_decision")

        primary_source_valid = False
        fallback_source_valid = False
        if provenance_present:
            scoped.extend(validate_provenance_envelope(provenance, f"{prediction_id}:provenance"))
            sources = provenance.get("sources")
            if not isinstance(sources, dict) or not sources:
                scoped.append(f"{prediction_id}:missing_source_provenance")
            else:
                valid_sources = set()
                for source_name, source_record in sources.items():
                    if not isinstance(source_record, dict):
                        scoped.append(f"{prediction_id}:source:{source_name}:provenance_not_object")
                        continue
                    source_status = str(source_record.get("status", ""))
                    if source_status in AVAILABLE_STATUSES:
                        source_errors = validate_provenance_envelope(
                            source_record,
                            f"{prediction_id}:source:{source_name}",
                        )
                        scoped.extend(source_errors)
                        if not source_errors:
                            valid_sources.add(source_name)
                primary_source_valid = "binance_futures" in valid_sources
                fallback_source_valid = (
                    not primary_source_valid
                    and bool(valid_sources.intersection({"bybit_futures", "coinbase_futures", "kraken_futures"}))
                )
        elif not is_legacy:
            scoped.append(f"{prediction_id}:missing_top_level_provenance")

        situation_obj = scenario.get("situation") if isinstance(scenario, dict) else None
        micro_obj = scenario.get("microstructure") if isinstance(scenario, dict) else None
        components_obj = scenario.get("components") if isinstance(scenario, dict) else None

        if actual5 not in (None, ""):
            coverage["5m"]["settled_predictions"] += 1
        if actual10 not in (None, ""):
            coverage["10m"]["settled_predictions"] += 1

        if model_version == "DEGRADED_NO_FRESH_DATA":
            if scenario.get("policy") != "safe_degraded_no_directional_claim":
                violations.append(f"{prediction_id}:degraded_policy_mismatch")

        row_has_active_violation = any(v.startswith(row_prefix) for v in violations)
        if is_legacy:
            legacy_unverified_count += 1
        elif legacy_coinbase_contract:
            legacy_unverified_count += 1
        elif not row_has_active_violation:
            verified_count += 1
            if primary_source_valid:
                verified_primary_count += 1
            elif fallback_source_valid:
                verified_fallback_count += 1

        if not row_has_active_violation and primary_source_valid:
            if actual5 not in (None, ""):
                coverage["5m"]["strict_primary_settled"] += 1
                if isinstance(situation_obj, dict) and isinstance(micro_obj, dict) and isinstance(components_obj, dict):
                    coverage["5m"]["situation_meta_ready"] += 1
                    coverage["5m"]["online_expert_ready"] += 1
            if actual10 not in (None, ""):
                coverage["10m"]["strict_primary_settled"] += 1
                if isinstance(situation_obj, dict) and isinstance(micro_obj, dict) and isinstance(components_obj, dict):
                    coverage["10m"]["situation_meta_ready"] += 1
                    coverage["10m"]["online_expert_ready"] += 1

    # Promotion evidence is tied to the production benchmark venue. Fallback
    # observations remain useful research data but cannot satisfy the primary
    # PIT requirement.
    pit_ready = bool(
        not violations
        and verified_primary_count >= MIN_STRICT_PIT_ROWS
    )
    for horizon in ("5m", "10m"):
        item = coverage[horizon]
        item["situation_meta_rows_needed"] = max(0, SITUATION_META_MIN_ROWS - item["situation_meta_ready"])
        item["online_expert_rows_needed"] = max(0, ONLINE_EXPERT_MIN_ROWS - item["online_expert_ready"])
        item["estimated_5m_cycles_lower_bound"] = int(item["situation_meta_rows_needed"])
        item["estimated_days_lower_bound"] = float(item["estimated_5m_cycles_lower_bound"] * 5 / (60 * 24))

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
        "coverage": coverage,
        "coverage_policy": (
            "strict_primary_settled requires actual outcome + valid primary PIT; "
            "situation_meta_ready additionally requires persisted situation, microstructure, and components; "
            "estimated_days is a lower bound assuming one qualifying prediction every 5 minutes"
        ),
        "legacy_unverified_count": legacy_unverified_count,
        "legacy_violation_count": len(legacy_violations),
        "legacy_violations": legacy_violations[:50],
        "violations": violations[:100],
        "violation_count": len(violations),
        "policy": "strict_pit_scope_with_legacy_unverified_quarantine",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        result["history"] = record_pit_history(result, OUT.parent / "pit_history")
    except Exception as exc:
        result["history"] = {"status": "FAILED", "error": f"{type(exc).__name__}:{exc}"}
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
