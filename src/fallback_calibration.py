"""Source-native, chronological calibration for BTC fallback predictions.

Only settled fallback predictions are used. The fit/holdout split is chronological,
the holdout is never used to choose parameters, and rejection leaves the predictor
at model-only probabilities.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

from db import DB, init_db

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
CLASSES = ["UP", "DOWN", "FLAT"]
HOLDOUT_FRACTION = 0.25
MIN_ROWS = 80
MAX_WEIGHT = 0.45
GRID_W = np.linspace(0.0, MAX_WEIGHT, 19)
GRID_T = np.linspace(0.7, 2.5, 37)
MIN_LOGLOSS_GAIN = 0.001
MIN_BRIER_GAIN = 0.0005


def _norm(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def _brier(y, p):
    yi = np.asarray([CLASSES.index(v) for v in y], dtype=int)
    one = np.eye(3)[yi]
    return float(np.mean(np.sum((_norm(p) - one) ** 2, axis=1)))


def _horizon_model_version(model_version, source, horizon):
    """Extract the horizon-specific source-native model version from a combined binding."""
    marker = f"{horizon}:{source}_fallback."
    for part in str(model_version or "").split("|"):
        if part.startswith(marker):
            return part
    return None


def _rows(source, horizon):
    actual = f"actual_direction_{horizon}"
    rows = []
    with sqlite3.connect(DB) as con:
        # A persisted production prediction stores both horizon bindings in one
        # pipe-delimited model_version. Query broadly, then extract the exact
        # horizon/source binding; this is essential for 10m fallback calibration.
        raw = con.execute(
            f"""SELECT created_at_utc,model_version,scenario_json,{actual}
                FROM predictions
                WHERE {actual} IS NOT NULL
                  AND model_version LIKE ?
                  AND model_version NOT LIKE 'DEGRADED_NO_FRESH_DATA%'
                ORDER BY created_at_utc""",
            (f"%{source}_fallback.%",),
        ).fetchall()
    for created, model_version, scenario_text, y in raw:
        horizon_version = _horizon_model_version(model_version, source, horizon)
        if horizon_version is None or y not in CLASSES:
            continue
        try:
            comp = json.loads(scenario_text or "{}").get("components", {})
            model = comp.get(f"model_raw_{horizon}")
            structural = comp.get(f"structural_{horizon}")
            if not isinstance(model, dict) or not isinstance(structural, dict):
                continue
            mp = np.asarray([float(model[c]) for c in CLASSES], dtype=float)
            sp = np.asarray([float(structural[c]) for c in CLASSES], dtype=float)
            if not np.isfinite(mp).all() or not np.isfinite(sp).all():
                continue
            rows.append({
                "created": str(created),
                "model_version": str(horizon_version),
                "model": _norm(mp),
                "structural": _norm(sp),
                "y": str(y),
            })
        except Exception:
            continue
    return rows


def _temperature_mix(model, temperature):
    p = np.clip(np.asarray(model, dtype=float), 1e-7, 1.0)
    z = np.log(p) / float(temperature)
    z -= z.max()
    q = np.exp(z)
    return q / q.sum()


def calibrate(source, horizon):
    rows = _rows(source, horizon)
    versions = sorted({r["model_version"] for r in rows})
    if not rows:
        return {
            "schema_version": 1,
            "source": source,
            "horizon": horizon,
            "model_version": None,
            "n": 0,
            "status": "insufficient_history",
            "temperature": 1.0,
            "blend_weight": 0.0,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    current_version = versions[-1]
    rows = [r for r in rows if r["model_version"] == current_version]
    if len(rows) < MIN_ROWS:
        return {
            "schema_version": 1,
            "source": source,
            "horizon": horizon,
            "model_version": current_version,
            "n": len(rows),
            "status": "insufficient_history",
            "temperature": 1.0,
            "blend_weight": 0.0,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    split = int(len(rows) * (1.0 - HOLDOUT_FRACTION))
    purge = int(horizon[:-1])
    train = rows[:max(1, split - purge)]
    holdout = rows[split:]
    if len(train) < 40 or len(holdout) < 20:
        return {
            "schema_version": 1,
            "source": source,
            "horizon": horizon,
            "model_version": current_version,
            "n": len(rows),
            "status": "insufficient_split",
            "temperature": 1.0,
            "blend_weight": 0.0,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    train_y = [r["y"] for r in train]
    holdout_y = [r["y"] for r in holdout]
    yi_train = [CLASSES.index(v) for v in train_y]
    yi_holdout = [CLASSES.index(v) for v in holdout_y]
    best = None
    for t in GRID_T:
        for w in GRID_W:
            train_model = np.asarray([_temperature_mix(r["model"], t) for r in train], dtype=float)
            train_struct = np.asarray([r["structural"] for r in train], dtype=float)
            fit = _norm((1.0 - w) * train_model + w * train_struct)
            ll = float(log_loss(yi_train, fit, labels=[0, 1, 2]))
            br = _brier(train_y, fit)
            key = (ll + 0.15 * br + 0.01 * float(w), ll, br, float(t), float(w))
            if best is None or key < best[0]:
                best = (key, float(t), float(w))

    t_best, w_best = best[1], best[2]
    hold_model = np.asarray([_temperature_mix(r["model"], t_best) for r in holdout], dtype=float)
    hold_struct = np.asarray([r["structural"] for r in holdout], dtype=float)
    baseline = _norm(np.asarray([_temperature_mix(r["model"], 1.0) for r in holdout]))
    candidate = _norm((1.0 - w_best) * hold_model + w_best * hold_struct)
    baseline_ll = float(log_loss(yi_holdout, baseline, labels=[0, 1, 2]))
    candidate_ll = float(log_loss(yi_holdout, candidate, labels=[0, 1, 2]))
    baseline_br = _brier(holdout_y, baseline)
    candidate_br = _brier(holdout_y, candidate)
    accepted = candidate_ll <= baseline_ll - MIN_LOGLOSS_GAIN and candidate_br <= baseline_br - MIN_BRIER_GAIN
    return {
        "schema_version": 1,
        "source": source,
        "horizon": horizon,
        "model_version": current_version,
        "n": len(rows),
        "fit_n": len(train),
        "holdout_n": len(holdout),
        "temperature": float(t_best) if accepted else 1.0,
        "blend_weight": float(w_best) if accepted else 0.0,
        "candidate_temperature": float(t_best),
        "candidate_blend_weight": float(w_best),
        "baseline_logloss": baseline_ll,
        "candidate_logloss": candidate_ll,
        "baseline_brier": baseline_br,
        "candidate_brier": candidate_br,
        "status": "accepted" if accepted else "rejected",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main():
    init_db()
    for source in ("coinbase",):
        for horizon in ("5m", "10m"):
            result = calibrate(source, horizon)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            (MODEL_DIR / f"{source}_{horizon}.calibration.json").write_text(
                json.dumps(result, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(json.dumps(result, ensure_ascii=False))
if __name__ == "__main__":
    main()
