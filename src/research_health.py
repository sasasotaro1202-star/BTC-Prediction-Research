"""Non-fatal integrity/health audit for the BTC research loop.

The audit never changes a production model. It checks that the database schema,
model metadata, calibration files, and prediction probabilities remain mutually
consistent. Operational failures are reported as data, not raised as workflow
failures, so a transient public-data outage cannot break the research cycle.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from db import DB, init_db
from feature_schema import FEATURES as CANONICAL_FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUT = ROOT / "data" / "historical_research" / "research_health.json"
HORIZONS = ("5m", "10m")
CLASSES = ("DOWN", "FLAT", "UP")
EXPECTED_FEATURES = len(CANONICAL_FEATURES)


def table_columns(con, table: str) -> set[str]:
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}


def count_duplicate_prediction_events(con) -> int:
    """Count excess rows using the same canonical event key as research audits."""
    groups = {}
    for created, target_5m, target_10m, feature_json in con.execute(
        "SELECT created_at_utc,target_5m,target_10m,feature_json FROM predictions"
    ):
        try:
            obj = json.loads(feature_json or "{}")
            canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError, json.JSONDecodeError):
            canonical = str(feature_json)
        fp = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        key = (str(created), str(target_5m), str(target_10m), fp)
        groups[key] = groups.get(key, 0) + 1
    return int(sum(max(0, n - 1) for n in groups.values()))


def finite_probability_row(row) -> bool:
    p = [float(x) for x in row]
    return all(math.isfinite(x) and x >= 0.0 for x in p) and abs(sum(p) - 1.0) <= 1e-5


def audit() -> dict:
    init_db()
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "ok": True, "checks": {}}
    required = {
        "prediction_id", "created_at_utc", "target_5m", "target_10m", "base_price",
        "p_up_5m", "p_down_5m", "p_flat_5m", "p_up_10m", "p_down_10m", "p_flat_10m",
        "model_version", "feature_json", "scenario_json", "actual_direction_5m", "actual_direction_10m",
    }
    try:
        with sqlite3.connect(DB) as con:
            cols = table_columns(con, "predictions")
            missing = sorted(required - cols)
            result["checks"]["schema"] = {"missing": missing, "ok": not missing}
            if missing:
                result["ok"] = False
                return result
            n = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            bad = con.execute("""
                SELECT COUNT(*) FROM predictions
                WHERE p_up_5m IS NULL OR p_down_5m IS NULL OR p_flat_5m IS NULL
                   OR p_up_10m IS NULL OR p_down_10m IS NULL OR p_flat_10m IS NULL
            """).fetchone()[0]
            duplicate_excess = count_duplicate_prediction_events(con)
            result["checks"]["database"] = {
                "predictions": int(n),
                "null_probability_rows": int(bad),
                "duplicate_exact_key_row_excess": duplicate_excess,
                "ok": bad == 0 and duplicate_excess == 0 and n > 0,
            }
            if bad or duplicate_excess or n <= 0:
                result["ok"] = False
            for h in HORIZONS:
                rows = con.execute(f"SELECT p_down_{h},p_flat_{h},p_up_{h} FROM predictions").fetchall()
                invalid = sum(not finite_probability_row(r) for r in rows)
                result["checks"][h] = {"rows": len(rows), "invalid_probability_rows": invalid, "ok": invalid == 0}
                if invalid:
                    result["ok"] = False
    except Exception as exc:
        result["ok"] = False
        result["checks"]["database_error"] = f"{type(exc).__name__}: {exc}"
        return result

    for h in HORIZONS:
        check = {"ok": True}
        mp = MODEL_DIR / f"{h}.json"
        mf = MODEL_DIR / f"{h}.joblib"
        cp = MODEL_DIR / f"{h}.calibration.json"
        bp = MODEL_DIR / f"{h}.blend.json"
        if mp.exists():
            try:
                obj = json.loads(mp.read_text(encoding="utf-8"))
                check["model_metadata"] = {
                    "artifact_ok": obj.get("artifact") == f"{h}.joblib",
                    "classes_ok": obj.get("classes") in (None, list(CLASSES)),
                    "feature_count": len(obj.get("features", [])) if obj.get("features") else None,
                    "feature_schema_ok": obj.get("features") in (None, list(CANONICAL_FEATURES)),
                }
                if not check["model_metadata"]["artifact_ok"] or not check["model_metadata"]["classes_ok"]:
                    check["ok"] = False
                if obj.get("features") and (len(obj["features"]) != EXPECTED_FEATURES or obj["features"] != list(CANONICAL_FEATURES)):
                    check["ok"] = False
            except Exception as exc:
                check["ok"] = False
                check["model_metadata_error"] = f"{type(exc).__name__}: {exc}"
        if mp.exists() != mf.exists():
            check["ok"] = False
            check["artifact_pair_ok"] = False
        else:
            check["artifact_pair_ok"] = True
        for path, name in ((cp, "calibration"), (bp, "blend")):
            if path.exists():
                try:
                    obj = json.loads(path.read_text(encoding="utf-8"))
                    check[name] = {"json_ok": True}
                    if name == "calibration":
                        t = float(obj.get("temperature", 1.0)); check[name]["bounds_ok"] = 0.5 <= t <= 3.0
                    else:
                        w = float(obj.get("base_weight", 0.20)); check[name]["bounds_ok"] = 0.0 <= w <= 0.45
                    if not check[name]["bounds_ok"]:
                        check["ok"] = False
                except Exception as exc:
                    check["ok"] = False
                    check[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
        result["checks"][f"model_{h}"] = check
        result["ok"] = result["ok"] and check["ok"]

    return result


def main() -> int:
    result = audit()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
