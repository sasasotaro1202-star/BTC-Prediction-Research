"""Leak-resistant calibration of the live model/structural blend.

The live predictor has two probability sources:
1. the production ML model trained on historical OHLCV features;
2. a causal structural/microstructure overlay built from current public data.

The overlay is useful only if it improves unseen settled predictions. This
module therefore learns only a single low-dimensional blend weight from past
settled predictions, using a chronological fit/holdout split. It never pools
incompatible production-model generations.
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
MIN_ROWS = 400
HOLDOUT_FRACTION = 0.25
DEFAULT_WEIGHT = 0.20
# A structural overlay must earn its production weight on unseen settled data.
# Until then (or after rejection), fail closed to the validated ML model alone.
FALLBACK_WEIGHT = 0.0
MAX_WEIGHT = 0.45
GRID = np.linspace(0.0, MAX_WEIGHT, 19)
MIN_LOGLOSS_GAIN = 0.001
MIN_BRIER_GAIN = 0.0005
# A candidate blend must improve across most chronological holdout blocks,
# not merely on one aggregate slice. This prevents a transient regime from
# activating the structural overlay for all future predictions.
BLOCK_SIZE = 25
MIN_BLOCKS = 4
MIN_IMPROVED_BLOCK_RATIO = 0.70


def _norm(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def _brier(y, p):
    idx = {c: i for i, c in enumerate(CLASSES)}
    yi = np.asarray([idx[v] for v in y])
    one = np.eye(3)[yi]
    return float(np.mean(np.sum((_norm(p) - one) ** 2, axis=1)))


def _actual_column(horizon: str) -> str:
    if horizon not in ("5m", "10m"):
        raise ValueError(f"unsupported horizon: {horizon}")
    return f"actual_direction_{horizon}"


def _current_registry_version(con, horizon: str):
    row = con.execute(
        "SELECT production_version FROM model_registry WHERE horizon=?",
        (horizon,),
    ).fetchone()
    return str(row[0]) if row and row[0] else None


def _rows(horizon: str):
    """Load settled rows for the currently registered production generation only."""
    actual = _actual_column(horizon)
    prob_suffix = horizon  # e.g. 5m -> p_up_5m; do not append another 'm'.
    rows = []
    with sqlite3.connect(DB) as con:
        model_version = _current_registry_version(con, horizon)
        if not model_version:
            return rows
        # Predictions store both horizon versions as "5m:<version>|10m:<version>".
        prefix = f"{horizon}:{model_version}|%"
        raw = con.execute(
            f"""SELECT created_at_utc,scenario_json,
                       p_up_{prob_suffix},p_down_{prob_suffix},p_flat_{prob_suffix},
                       {actual}
                FROM predictions
                WHERE {actual} IS NOT NULL
                  AND model_version LIKE ?
                  AND model_version NOT LIKE 'DEGRADED_NO_FRESH_DATA%'
                ORDER BY created_at_utc""",
            (prefix,),
        ).fetchall()

    for created, scenario_text, up, down, flat, y in raw:
        if y not in CLASSES:
            continue
        try:
            comp = json.loads(scenario_text or "{}").get("components", {})
            model = comp.get(f"model_raw_{horizon}") or comp.get("model_raw")
            structural = comp.get(f"structural_{horizon}") or comp.get("structural")
            if not isinstance(model, dict) or not isinstance(structural, dict):
                continue
            mp = [float(model[c]) for c in CLASSES]
            sp = [float(structural[c]) for c in CLASSES]
            stored = [float(up), float(down), float(flat)]
            if not all(math.isfinite(x) for x in mp + sp + stored):
                continue
            rows.append((created, mp, sp, y))
        except Exception:
            continue
    return rows



def _block_stability(y, model_probs, structural_probs, weight, block_size=BLOCK_SIZE):
    """Measure chronological holdout stability of a fixed blend weight."""
    y_idx = [CLASSES.index(v) for v in y]
    model = _norm(model_probs)
    structural = _norm(structural_probs)
    blended = _norm((1.0 - float(weight)) * model + float(weight) * structural)
    blocks = []
    for start in range(0, len(y), int(block_size)):
        stop = min(start + int(block_size), len(y))
        if stop - start < max(10, int(block_size) // 2):
            continue
        yi = y_idx[start:stop]
        base_p = model[start:stop]
        cand_p = blended[start:stop]
        base_ll = float(log_loss(yi, base_p, labels=[0, 1, 2]))
        cand_ll = float(log_loss(yi, cand_p, labels=[0, 1, 2]))
        base_br = float(_brier(y[start:stop], base_p))
        cand_br = float(_brier(y[start:stop], cand_p))
        blocks.append({
            "start": start,
            "stop": stop,
            "logloss_delta": cand_ll - base_ll,
            "brier_delta": cand_br - base_br,
        })
    if not blocks:
        return {
            "blocks": 0,
            "improved_logloss_ratio": 0.0,
            "improved_brier_ratio": 0.0,
            "stable": False,
            "deltas": [],
        }
    ll_ratio = float(np.mean([x["logloss_delta"] < 0.0 for x in blocks]))
    br_ratio = float(np.mean([x["brier_delta"] < 0.0 for x in blocks]))
    stable = (
        len(blocks) >= MIN_BLOCKS
        and ll_ratio >= MIN_IMPROVED_BLOCK_RATIO
        and br_ratio >= MIN_IMPROVED_BLOCK_RATIO
    )
    return {
        "blocks": len(blocks),
        "improved_logloss_ratio": ll_ratio,
        "improved_brier_ratio": br_ratio,
        "stable": stable,
        "deltas": blocks,
    }

def calibrate(horizon: str):
    rows = _rows(horizon)
    with sqlite3.connect(DB) as con:
        model_version = _current_registry_version(con, horizon)

    if len(rows) < MIN_ROWS:
        return {
            "horizon": horizon,
            "model_version": model_version,
            "base_weight": FALLBACK_WEIGHT,
            "n": len(rows),
            "status": "insufficient_history",
        }

    split = int(len(rows) * (1.0 - HOLDOUT_FRACTION))
    purge_gap = int(horizon[:-1])
    train = rows[:max(1, split - purge_gap)]
    holdout = rows[split:]
    y = [r[3] for r in holdout]
    model_h = _norm([r[1] for r in holdout])
    structural_h = _norm([r[2] for r in holdout])
    baseline_ll = float(log_loss([CLASSES.index(v) for v in y], model_h, labels=[0, 1, 2]))
    baseline_br = _brier(y, model_h)
    structural_ll = float(log_loss([CLASSES.index(v) for v in y], structural_h, labels=[0, 1, 2]))
    structural_br = _brier(y, structural_h)

    train_y = [r[3] for r in train]
    train_model = _norm([r[1] for r in train])
    train_struct = _norm([r[2] for r in train])
    best_w = DEFAULT_WEIGHT
    best_fit = float("inf")
    for w in GRID:
        p = _norm((1.0 - w) * train_model + w * train_struct)
        ll = float(log_loss([CLASSES.index(v) for v in train_y], p, labels=[0, 1, 2]))
        if ll < best_fit:
            best_fit = ll
            best_w = float(w)

    candidate = _norm((1.0 - best_w) * model_h + best_w * structural_h)
    cand_ll = float(log_loss([CLASSES.index(v) for v in y], candidate, labels=[0, 1, 2]))
    cand_br = _brier(y, candidate)
    stability = _block_stability(y, model_h, structural_h, best_w)
    accepted = (
        cand_ll <= baseline_ll - MIN_LOGLOSS_GAIN
        and cand_br <= baseline_br - MIN_BRIER_GAIN
        and stability["stable"]
    )
    final_w = best_w if accepted else FALLBACK_WEIGHT
    return {
        "horizon": horizon,
        "model_version": model_version,
        "base_weight": float(final_w),
        "n": len(rows),
        "fit_n": len(train),
        "holdout_n": len(holdout),
        "holdout_fraction": HOLDOUT_FRACTION,
        "baseline_logloss": baseline_ll,
        "candidate_logloss": cand_ll,
        "baseline_brier": baseline_br,
        "candidate_brier": cand_br,
        "structural_logloss": structural_ll,
        "structural_brier": structural_br,
        "candidate_weight": float(best_w),
        "stability": stability,
        "status": "accepted" if accepted else "rejected",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def save(result):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{result['horizon']}.blend.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main():
    init_db()
    for horizon in ("5m", "10m"):
        result = calibrate(horizon)
        save(result)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
