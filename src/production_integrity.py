"""Strict production-integrity gate for BTC prediction runs.

This gate is deliberately independent from model scoring. A green workflow is
not sufficient: the run must also have valid artifacts, probabilities,
point-in-time metadata, and a usable champion model.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib

from db import DB, init_db

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
HEALTH = ROOT / "data" / "historical_research" / "research_health.json"
CLASSES = ["DOWN", "FLAT", "UP"]
FEATURES = [
    "ret_1m", "ret_3m", "ret_5m", "ret_10m", "acceleration",
    "volatility_5m", "volatility_10m", "range_position_10m", "body_1m",
    "upper_wick_1m", "lower_wick_1m", "volume_ratio", "volume_trend",
    "ema_gap_5m", "ema_gap_10m",
]
DEFAULT_MAX_AGE_SECONDS = 900


def max_prediction_age_seconds() -> float:
    raw = os.getenv("BTC_INTEGRITY_MAX_PREDICTION_AGE_SECONDS", str(DEFAULT_MAX_AGE_SECONDS))
    try:
        value = float(raw)
    except ValueError:
        fail(f"invalid BTC_INTEGRITY_MAX_PREDICTION_AGE_SECONDS: {raw!r}")
    if not math.isfinite(value) or value <= 0:
        fail("BTC_INTEGRITY_MAX_PREDICTION_AGE_SECONDS must be positive and finite")
    return value


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def finite_probs(values) -> bool:
    vals = [float(x) for x in values]
    return all(math.isfinite(x) and 0.0 <= x <= 1.0 for x in vals) and abs(sum(vals) - 1.0) <= 1e-5


def check_model(horizon: str) -> dict:
    meta_path = MODEL_DIR / f"{horizon}.json"
    model_path = MODEL_DIR / f"{horizon}.joblib"
    if not meta_path.is_file() or not model_path.is_file():
        fail(f"missing production artifact for {horizon}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("horizon") != horizon:
        fail(f"{horizon}: horizon metadata mismatch")
    if meta.get("artifact") != model_path.name:
        fail(f"{horizon}: artifact metadata mismatch")
    if meta.get("classes") != CLASSES:
        fail(f"{horizon}: class order mismatch")
    if meta.get("features") != FEATURES:
        fail(f"{horizon}: feature contract mismatch")
    model = joblib.load(model_path)
    classes = [str(x) for x in getattr(model, "classes_", [])]
    if classes != CLASSES:
        fail(f"{horizon}: serialized model classes mismatch: {classes}")
    if not meta.get("model_version"):
        fail(f"{horizon}: missing model_version")
    return {"horizon": horizon, "model_version": meta["model_version"], "feature_count": len(FEATURES)}


def check_db() -> dict:
    if not DB.exists() or DB.stat().st_size <= 0:
        fail("predictions.db missing or empty")
    with sqlite3.connect(DB) as con:
        row = con.execute("SELECT created_at_utc,base_price,p_up_5m,p_down_5m,p_flat_5m,p_up_10m,p_down_10m,p_flat_10m,model_version,feature_json,scenario_json FROM predictions ORDER BY prediction_id DESC LIMIT 1").fetchone()
        if not row:
            fail("no predictions recorded")
        created = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - created).total_seconds()
        max_age = max_prediction_age_seconds()
        if age < -60 or age > max_age:
            fail(f"latest prediction is stale or future-dated: {age:.0f}s (max {max_age:.0f}s)")
        if not math.isfinite(float(row[1])) or float(row[1]) <= 0:
            fail("latest base price invalid")
        if not finite_probs(row[2:5]) or not finite_probs(row[5:8]):
            fail("latest prediction probabilities invalid")
        feature_obj = json.loads(row[9] or "{}")
        if not all(k in feature_obj and math.isfinite(float(feature_obj[k])) for k in FEATURES):
            fail("latest prediction feature snapshot is incomplete or non-finite")
        scenario = json.loads(row[10] or "{}")
        if not isinstance(scenario.get("data_quality"), dict):
            fail("latest prediction lacks data_quality metadata")
        if not row[8]:
            fail("latest prediction lacks model_version")
    return {"latest_age_seconds": int(age), "prediction_ok": True, "max_age_seconds": max_age}


def main() -> int:
    init_db()
    models = [check_model(h) for h in ("5m", "10m")]
    db = check_db()
    if HEALTH.is_file():
        health = json.loads(HEALTH.read_text(encoding="utf-8"))
        if not isinstance(health.get("ok"), bool):
            fail("research health artifact malformed")
    print(json.dumps({"status": "PASS", "models": models, "database": db}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
