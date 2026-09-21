
"""Strict research-input chronology and duplicate audit for BTC OOS evidence."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_compare import _strict_pit_provenance_ok
from datetime import datetime, timezone
from pathlib import Path

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
        f"""SELECT prediction_id, created_at_utc, {target_col}, feature_json, {actual_col}, scenario_json
            FROM predictions
            WHERE {actual_col} IS NOT NULL
            ORDER BY created_at_utc, prediction_id"""
    ).fetchall()

    invalid_ts = 0
    chronology_violations = []
    duplicate_groups = Counter()
    class_counts = Counter()
    valid_rows = 0
    strict_pit_rows = 0

    for prediction_id, created_at, target_at, feature_json, actual, scenario_json in rows:
        created_dt = _dt(created_at)
        target_dt = _dt(target_at)
        if created_dt is None or target_dt is None:
            invalid_ts += 1
        elif created_dt >= target_dt:
            chronology_violations.append({
                "prediction_id": int(prediction_id),
                "created_at_utc": str(created_at),
                "target_at_utc": str(target_at),
            })

        obj = _safe_json(feature_json)
        fp = hashlib.sha256(
            json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        duplicate_groups[(str(created_at), str(target_at), fp)] += 1

        if actual in CLASSES:
            class_counts[str(actual)] += 1
        if created_dt is not None and target_dt is not None and created_dt < target_dt:
            valid_rows += 1
            try:
                scenario = json.loads(scenario_json or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                scenario = {}
            if _strict_pit_provenance_ok(scenario, created_at):
                strict_pit_rows += 1

    duplicate_excess = int(sum(max(0, n - 1) for n in duplicate_groups.values()))
    total = len(rows)
    majority = (max(class_counts.values()) / total) if total and class_counts else None

    return {
        "settled_rows": total,
        "valid_rows": valid_rows,
        "strict_pit_rows": strict_pit_rows,
        "strict_pit_excluded_rows": total - strict_pit_rows,
        "excluded_rows": total - valid_rows,
        "invalid_timestamp_rows": invalid_ts,
        "chronology_violation_count": len(chronology_violations),
        "chronology_violations_sample": chronology_violations[:20],
        "duplicate_exact_key_row_excess": duplicate_excess,
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
        if item["duplicate_exact_key_row_excess"]:
            failures.append(f"{h}:duplicate_exact_key_row_excess={item['duplicate_exact_key_row_excess']}")

    payload = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_changed": False,
        "ok": not failures,
        "status": "PASS" if not failures else "HOLD",
        "policy": "strict_prediction_before_target_filter_plus_exact_duplicate_audit",
        "failure_reasons": failures,
        "horizons": horizons,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
