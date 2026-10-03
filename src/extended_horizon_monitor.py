"""Research-only monitoring for the seven extended BTC forecast horizons."""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

try:
    from db import DB, init_db
    from horizon_registry import EXTENDED_RESEARCH_HORIZONS
except ModuleNotFoundError:
    from src.db import DB, init_db
    from src.horizon_registry import EXTENDED_RESEARCH_HORIZONS

OUT = Path(DB).parent / "historical_research" / "extended_horizon_performance.json"
CLASSES = ("DOWN", "FLAT", "UP")
MIN_ROWS = 100


def metrics(rows):
    y = []
    p = []
    for row in rows:
        label = row[1]
        if label not in CLASSES:
            continue
        try:
            probs = np.asarray([float(row[2]), float(row[3]), float(row[4])], dtype=float)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(probs).all() or np.any(probs < 0) or float(probs.sum()) <= 0:
            continue
        probs /= probs.sum()
        y.append(CLASSES.index(label))
        p.append(probs)
    if len(y) < MIN_ROWS:
        return {
            "status": "insufficient_data",
            "n": len(y),
            "minimum": MIN_ROWS,
            "research_only": True,
            "calibration_status": "UNVALIDATED_UNCALIBRATED",
        }
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = p.argmax(axis=1)
    one = np.eye(3)[y]
    return {
        "status": "ok",
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "logloss": float(log_loss(y, p, labels=[0, 1, 2])),
        "brier": float(np.mean(np.sum((p - one) ** 2, axis=1))),
        "research_only": True,
        "calibration_status": "UNVALIDATED_UNCALIBRATED",
    }


def run():
    init_db()
    result = {
        "schema_version": 1,
        "research_only": True,
        "policy": "diagnostic_only_no_model_input_no_promotion_effect",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "horizons": {},
    }
    with sqlite3.connect(DB) as con:
        for h in EXTENDED_RESEARCH_HORIZONS:
            rows = con.execute(
                f"""SELECT prediction_id, actual_direction_{h},
                           p_down_{h}, p_flat_{h}, p_up_{h}
                    FROM predictions
                    WHERE actual_direction_{h} IS NOT NULL
                    ORDER BY prediction_id"""
            ).fetchall()
            result["horizons"][h] = metrics(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
