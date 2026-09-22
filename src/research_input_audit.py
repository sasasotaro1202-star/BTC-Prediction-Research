"""Strict research-input chronology, PIT, and immutable identity audit for BTC OOS evidence."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_compare import strict_pit_provenance_reason
from prediction_identity import model_prediction_event_key
from feature_schema import FEATURES

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "predictions.db"
OUT = ROOT / "data" / "historical_research" / "research_input_audit.json"
HORIZONS = {"5m": "actual_direction_5m", "10m": "actual_direction_10m"}
TARGETS = {"5m": "target_5m", "10m": "target_10m"}
CLASSES = ("DOWN", "FLAT", "UP")

def _dt(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None

def _safe_json(value):
    try:
        obj = json.loads(value or "{}")
        return obj if isinstance(obj, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}

def audit_horizon(con, horizon: str) -> dict:
    target_col = TARGETS[horizon]
    actual_col = HORIZONS[horizon]
    rows = con.execute(
        f"""SELECT prediction_id, created_at_utc, {target_col}, model_version, feature_json, {actual_col},
                   p_up_{horizon}, p_down_{horizon}, p_flat_{horizon}, scenario_json
            FROM predictions
            ORDER BY created_at_utc, prediction_id"""
    ).fetchall()

    invalid_ts = 0
    chronology_violations = []
    duplicate_groups = Counter()
    settled_duplicate_groups = Counter()
    class_counts = Counter()
    total_rows = len(rows)
    settled_rows = 0
    valid_rows = 0
    malformed_identity_rows = 0
    quarantined_identity_rows = 0
    quarantine_reasons = Counter()
    strict_pit_rows = 0
    strict_pit_settled_rows = 0
    strict_pit_failure_reasons = Counter()

    for (
        prediction_id, created_at, target_at, model_version, feature_json, actual,
        p_up, p_down, p_flat, scenario_json,
    ) in rows:
        created_dt = _dt(created_at)
        target_dt = _dt(target_at)
        chronological = created_dt is not None and target_dt is not None and created_dt < target_dt
        if created_dt is None or target_dt is None:
            invalid_ts += 1
        elif not chronological:
            chronology_violations.append({
                "prediction_id": int(prediction_id),
                "created_at_utc": str(created_at),
                "target_at_utc": str(target_at),
            })

        is_degraded = model_version == "DEGRADED_NO_FRESH_DATA"
        legacy_coinbase_v1 = (
            model_version == "5m:v1.0|10m:v1.0"
            and _safe_json(scenario_json).get("price_source") == "coinbase_btc_usd"
        )
        if legacy_coinbase_v1:
            # First-generation Coinbase fallback rows predate the canonical
            # 15-feature snapshot contract. Keep them visible as historical
            # state, but exclude them from identity/OOS evidence explicitly.
            quarantined_identity_rows += 1
            quarantine_reasons["legacy_coinbase_v1_incomplete_feature_snapshot"] += 1
        if legacy_coinbase_v1:
            pass
        elif is_degraded:
            # Degraded rows intentionally contain no directional feature snapshot.
            # They are preserved in the canonical DB but are not valid candidate/OOS
            # events and therefore cannot participate in immutable model-event
            # identity. Keep them visible as explicit quarantine, not as a silent
            # success and not as a malformed research observation.
            quarantined_identity_rows += 1
            quarantine_reasons["degraded_prediction_no_feature_snapshot"] += 1
        else:
            try:
                obj = _safe_json(feature_json)
                x = [float(obj[k]) for k in FEATURES]
                production = [float(p_down), float(p_flat), float(p_up)]
                event_key = model_prediction_event_key(
                    created=created_at, target=target_at,
                    model_version=model_version, x=x, production=production,
                )
                duplicate_groups[event_key] += 1
                if actual is not None:
                    settled_duplicate_groups[event_key] += 1
            except (KeyError, TypeError, ValueError, OverflowError):
                malformed_identity_rows += 1

        if actual is not None:
            settled_rows += 1
            if actual in CLASSES:
                class_counts[str(actual)] += 1

        scenario = _safe_json(scenario_json)
        if legacy_coinbase_v1:
            strict_pit_failure_reasons["legacy_coinbase_v1_quarantined"] += 1
        elif is_degraded:
            strict_pit_failure_reasons["degraded_prediction_quarantined"] += 1
        else:
            pit_reason = strict_pit_provenance_reason(scenario, created_at)
            if pit_reason is None and chronological:
                strict_pit_rows += 1
                if actual is not None:
                    strict_pit_settled_rows += 1
            else:
                strict_pit_failure_reasons[pit_reason or "prediction_time_not_before_target"] += 1

        if chronological:
            valid_rows += 1

    duplicate_excess = int(sum(max(0, n - 1) for n in duplicate_groups.values()))
    settled_duplicate_excess = int(
        sum(max(0, n - 1) for n in settled_duplicate_groups.values())
    )
    majority = max(class_counts.values()) / settled_rows if settled_rows and class_counts else None

    return {
        "total_rows": total_rows,
        "settled_rows": settled_rows,
        "valid_rows": valid_rows,
        "strict_pit_rows": strict_pit_rows,
        "strict_pit_settled_rows": strict_pit_settled_rows,
        "strict_pit_excluded_rows": total_rows - strict_pit_rows,
        "strict_pit_settled_excluded_rows": settled_rows - strict_pit_settled_rows,
        "strict_pit_failure_reasons": dict(strict_pit_failure_reasons),
        "malformed_identity_rows": malformed_identity_rows,
        "quarantined_identity_rows": quarantined_identity_rows,
        "quarantine_reasons": dict(quarantine_reasons),
        "excluded_rows": total_rows - valid_rows,
        "invalid_timestamp_rows": invalid_ts,
        "chronology_violation_count": len(chronology_violations),
        "chronology_violations_sample": chronology_violations[:20],
        "duplicate_exact_key_row_excess": duplicate_excess,
        "settled_duplicate_exact_key_row_excess": settled_duplicate_excess,
        "class_counts": dict(class_counts),
        "majority_accuracy": majority,
    }

def main() -> int:
    if not DB.exists() or DB.stat().st_size <= 0:
        raise SystemExit("research input audit: prediction database missing or empty")
    with sqlite3.connect(DB) as con:
        horizons = {h: audit_horizon(con, h) for h in HORIZONS}

    failures = []
    for h, item in horizons.items():
        if item["invalid_timestamp_rows"]:
            failures.append(f"{h}:invalid_timestamp_rows={item['invalid_timestamp_rows']}")
        if item["chronology_violation_count"]:
            failures.append(f"{h}:chronology_violations={item['chronology_violation_count']}")
        if item["malformed_identity_rows"]:
            failures.append(f"{h}:malformed_identity_rows={item['malformed_identity_rows']}")
        if item["duplicate_exact_key_row_excess"]:
            failures.append(f"{h}:duplicate_exact_key_row_excess={item['duplicate_exact_key_row_excess']}")

    payload = {
        "schema_version": 2,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_changed": False,
        "ok": not failures,
        "status": "PASS" if not failures else "HOLD",
        "policy": "all_prediction_event_chronology_identity_and_strict_pit_audit",
        "failure_reasons": failures,
        "horizons": horizons,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
