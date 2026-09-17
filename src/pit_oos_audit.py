"""Audit prediction records for point-in-time and out-of-sample integrity."""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from db import DB, init_db

OUT = Path(DB).parent / "historical_research" / "pit_oos_audit.json"
MAX_FUTURE_SKEW_SECONDS = 60

def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)

def audit() -> dict:
    init_db()
    checked = 0
    violations: list[str] = []
    with sqlite3.connect(DB) as con:
        rows = con.execute(
            "SELECT prediction_id, created_at_utc, target_5m, target_10m, model_version, scenario_json "
            "FROM predictions ORDER BY created_at_utc, prediction_id"
        ).fetchall()
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
        if target5 <= created:
            violations.append(f"{prediction_id}:5m_target_not_after_decision")
        if target10 <= target5:
            violations.append(f"{prediction_id}:10m_target_not_after_5m_target")
        try:
            scenario = json.loads(scenario_raw or "{}")
        except Exception:
            violations.append(f"{prediction_id}:invalid_scenario_json")
            continue
        decision_raw = scenario.get("decision_time_utc")
        if decision_raw:
            try:
                decision = parse_utc(str(decision_raw))
                if abs((decision - created).total_seconds()) > MAX_FUTURE_SKEW_SECONDS:
                    violations.append(f"{prediction_id}:decision_time_mismatch")
                cutoff_raw = scenario.get("market_data_cutoff_utc")
                if cutoff_raw:
                    cutoff = parse_utc(str(cutoff_raw))
                    if cutoff > decision:
                        violations.append(f"{prediction_id}:market_cutoff_after_decision")
            except Exception:
                violations.append(f"{prediction_id}:invalid_pit_metadata")
        if model_version == "DEGRADED_NO_FRESH_DATA" and scenario.get("policy") != "safe_degraded_no_directional_claim":
            violations.append(f"{prediction_id}:degraded_policy_mismatch")
    result = {
        "ok": not violations,
        "checked_predictions": checked,
        "violations": violations[:100],
        "violation_count": len(violations),
        "policy": "decision_time_before_targets; market_inputs_not_after_decision_time; degraded_mode_cannot_claim_direction",
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
