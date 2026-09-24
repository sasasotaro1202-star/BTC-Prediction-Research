"""Research-only prequential calibration replay for live fallback predictions.

Fallback predictions are evaluated only after their labels are settled. Each
future block receives a calibration temperature learned from strictly earlier
settled fallback observations, preserving the predict-then-observe ordering.
This artifact never changes production calibration or model state.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fallback_calibration import _horizon_model_version, _temperature_mix, _norm
from model_compare import metrics
from sklearn.metrics import log_loss

OUT = ROOT / "data" / "historical_research" / "fallback_prequential_calibration_oos.json"
DB = ROOT / "data" / "predictions.db"
CLASSES = ["UP", "DOWN", "FLAT"]
HORIZONS = ("5m", "10m")
SOURCE = "coinbase"
MIN_ROWS = 80
MIN_HISTORY = 40
TEST_BLOCK = 15
FINAL_HOLDOUT_FRAC = 0.20
MIN_BLOCKS = 3


def _rows(horizon: str):
    actual = f"actual_direction_{horizon}"
    rows = []
    with sqlite3.connect(DB) as con:
        raw = con.execute(
            f"""SELECT created_at_utc, model_version, scenario_json, {actual}
                FROM predictions
                WHERE {actual} IS NOT NULL
                  AND model_version LIKE ?
                  AND model_version NOT LIKE 'DEGRADED_NO_FRESH_DATA%'
                ORDER BY created_at_utc""",
            (f"%{SOURCE}_fallback.%",),
        ).fetchall()
    for created, version, scenario_text, y in raw:
        hv = _horizon_model_version(version, SOURCE, horizon)
        if hv is None or y not in CLASSES:
            continue
        try:
            obj = json.loads(scenario_text or "{}")
            comp = obj.get("components", {})
            raw_p = comp.get(f"model_raw_{horizon}")
            if not isinstance(raw_p, dict):
                continue
            p = _norm(np.asarray([float(raw_p[c]) for c in CLASSES], dtype=float))
            if p.shape != (3,) or not np.isfinite(p).all():
                continue
            rows.append({
                "created": str(created),
                "model_version": str(hv),
                "p": p,
                "y": str(y),
            })
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return rows


def _metrics(y, probs):
    # model_compare uses the canonical DOWN/FLAT/UP order.
    canonical_y = list(y)
    canonical_p = np.asarray(
        [[float(p[1]), float(p[2]), float(p[0])] for p in probs],
        dtype=float,
    )
    return metrics(canonical_y, canonical_p)


def _to_canonical_probability_matrix(probs):
    """Convert source-native UP/DOWN/FLAT order to canonical DOWN/FLAT/UP."""
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("probability matrix must have shape (n, 3)")
    p = _norm(p)
    return p[:, [1, 2, 0]]


def _adaptive_temperature(history):
    if len(history) < MIN_HISTORY:
        return 1.0
    p = _to_canonical_probability_matrix([r["p"] for r in history])
    y = [r["y"] for r in history]
    y_idx = np.asarray(["DOWN", "FLAT", "UP"])
    labels = {v: i for i, v in enumerate(y_idx)}
    yi = np.asarray([labels[v] for v in y], dtype=int)
    logits = np.log(np.clip(p, 1e-6, 1.0))
    best_loss = float("inf")
    best_t = 1.0
    for t in np.linspace(0.7, 2.5, 73):
        z = logits / float(t)
        z -= z.max(axis=1, keepdims=True)
        q = np.exp(z)
        q /= q.sum(axis=1, keepdims=True)
        loss = float(log_loss(yi, q, labels=[0, 1, 2]))
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return float(np.clip(best_t, 0.5, 3.0))


def evaluate(horizon: str):
    rows = _rows(horizon)
    if len(rows) < MIN_ROWS:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_settled_fallback_rows",
            "n": len(rows),
            "source": SOURCE,
            "promotion_evidence_eligible": False,
        }

    rows = rows[-min(len(rows), 2000):]
    dev_end = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    development = rows[:dev_end]
    holdout = rows[dev_end:]
    if len(development) < MIN_HISTORY + TEST_BLOCK * MIN_BLOCKS or len(holdout) < 10:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_prequential_development_history",
            "n": len(rows),
            "development_n": len(development),
            "final_holdout_n": len(holdout),
            "source": SOURCE,
            "promotion_evidence_eligible": False,
        }

    windows = []
    for end in range(MIN_HISTORY, len(development), TEST_BLOCK):
        test = development[end:min(end + TEST_BLOCK, len(development))]
        if len(test) < max(10, TEST_BLOCK // 2):
            continue
        history = development[:end]
        adaptive_t = _adaptive_temperature(history)
        raw = np.asarray([r["p"] for r in test], dtype=float)
        adaptive = np.asarray([_temperature_mix(r["p"], adaptive_t) for r in test], dtype=float)
        y = [r["y"] for r in test]
        raw_m = _metrics(y, raw)
        adaptive_m = _metrics(y, adaptive)
        windows.append({
            "n": len(test),
            "start_created": test[0]["created"],
            "end_created": test[-1]["created"],
            "history_n": len(history),
            "adaptive_temperature": adaptive_t,
            "raw": raw_m,
            "adaptive": adaptive_m,
            "delta": {
                "accuracy": adaptive_m["accuracy"] - raw_m["accuracy"],
                "logloss": adaptive_m["logloss"] - raw_m["logloss"],
                "brier": adaptive_m["brier"] - raw_m["brier"],
                "calibration_error": adaptive_m["calibration_error"] - raw_m["calibration_error"],
            },
        })

    if len(windows) < MIN_BLOCKS:
        return {
            "status": "DEFERRED",
            "reason": "too_few_prequential_windows",
            "n": len(rows),
            "source": SOURCE,
            "promotion_evidence_eligible": False,
        }

    delta = np.asarray(
        [[w["delta"]["accuracy"], w["delta"]["logloss"], w["delta"]["brier"], w["delta"]["calibration_error"]]
         for w in windows],
        dtype=float,
    )
    raw_hold = np.asarray([r["p"] for r in holdout], dtype=float)
    # Final holdout temperature is frozen from development only.
    frozen_t = _adaptive_temperature(development)
    adaptive_hold = np.asarray([_temperature_mix(r["p"], frozen_t) for r in holdout], dtype=float)
    y_hold = [r["y"] for r in holdout]
    raw_hold_m = _metrics(y_hold, raw_hold)
    adaptive_hold_m = _metrics(y_hold, adaptive_hold)

    summary = {
        "windows": len(windows),
        "development_n": len(development),
        "final_holdout_n": len(holdout),
        "mean_accuracy_delta": float(delta[:, 0].mean()),
        "mean_logloss_delta": float(delta[:, 1].mean()),
        "mean_brier_delta": float(delta[:, 2].mean()),
        "mean_calibration_error_delta": float(delta[:, 3].mean()),
        "improved_logloss_ratio": float(np.mean(delta[:, 1] < 0)),
        "improved_brier_ratio": float(np.mean(delta[:, 2] < 0)),
        "improved_calibration_ratio": float(np.mean(delta[:, 3] <= 0)),
        "non_worse_accuracy_ratio": float(np.mean(delta[:, 0] >= -0.005)),
        "frozen_temperature": frozen_t,
    }

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "policy": "coinbase_fallback_prequential_temperature_from_prior_settled_rows",
        "source": SOURCE,
        "summary": summary,
        "windows": windows,
        "final_holdout_protected": True,
        "final_holdout_used_for_selection": False,
        "final_holdout": {
            "raw": raw_hold_m,
            "adaptive": adaptive_hold_m,
            "delta": {
                "accuracy": adaptive_hold_m["accuracy"] - raw_hold_m["accuracy"],
                "logloss": adaptive_hold_m["logloss"] - raw_hold_m["logloss"],
                "brier": adaptive_hold_m["brier"] - raw_hold_m["brier"],
                "calibration_error": adaptive_hold_m["calibration_error"] - raw_hold_m["calibration_error"],
            },
            "frozen_temperature": frozen_t,
        },
    }


def main():
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "source": SOURCE,
        "horizons": {h: evaluate(h) for h in HORIZONS},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
