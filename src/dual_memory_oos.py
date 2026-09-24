"""Research-only dual-memory challenger for BTC short-horizon direction.

A full-history expert preserves broad regime knowledge while a recent-history
expert adapts faster to concept drift. Their soft weight is learned only from
completed earlier OOS blocks. The final holdout is never used for selection and
no production artifact is modified.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model_compare import (
    HORIZONS,
    CLASSES,
    MIN_TRAIN,
    TEST_BLOCK,
    PURGE_BARS,
    EMBARGO_BARS,
    load_archive_research_rows,
    metrics,
    aligned,
    _temperature,
    apply_temperature,
)

OUT = ROOT / "data" / "historical_research" / "dual_memory_oos.json"
MAX_ROWS = 12000
RECENT_FRACTION = 0.50
MIN_RECENT_TRAIN = 1000
FINAL_HOLDOUT_FRAC = 0.20
MIN_BLOCKS = 8


def _factory():
    return RandomForestClassifier(
        n_estimators=320,
        max_depth=10,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


def _fit_calibrated(train, test):
    if len(train) < MIN_TRAIN or len(test) < max(50, TEST_BLOCK // 2):
        return None
    y = [r["y"] for r in train]
    split = max(int(len(train) * 0.75), MIN_TRAIN - 100)
    if split < 100 or len(train) - split < 50:
        return None
    prefix = train[:split]
    cal = _factory()
    if len(set(r["y"] for r in prefix)) < 3:
        return None
    cal.fit(
        np.asarray([r["x"] for r in prefix], dtype=float),
        np.asarray([r["y"] for r in prefix]),
    )
    cal_p = aligned(cal, np.asarray([r["x"] for r in train[split:]], dtype=float))
    temp = _temperature(cal_p, [r["y"] for r in train[split:]])
    model = _factory()
    model.fit(
        np.asarray([r["x"] for r in train], dtype=float),
        np.asarray(y),
    )
    raw = aligned(model, np.asarray([r["x"] for r in test], dtype=float))
    return apply_temperature(raw, temp)


def _predict_champion(model, rows):
    return aligned(model, np.asarray([r["x"] for r in rows], dtype=float))


def _adaptive_recent_weight(history):
    if not history:
        return 0.50
    recent_ll = float(np.mean([r["recent_logloss"] for r in history[-6:]]))
    full_ll = float(np.mean([r["full_logloss"] for r in history[-6:]]))
    recent_br = float(np.mean([r["recent_brier"] for r in history[-6:]]))
    full_br = float(np.mean([r["full_brier"] for r in history[-6:]]))
    # Logistic soft routing: the lower-loss expert receives more weight.
    score_gap = 0.70 * (full_ll - recent_ll) + 0.30 * (full_br - recent_br)
    raw = 1.0 / (1.0 + np.exp(-float(score_gap) / 0.02))
    shrunk = 0.35 * 0.50 + 0.65 * raw
    return float(np.clip(shrunk, 0.20, 0.80))


def _mix(full_p, recent_p, recent_weight):
    w = float(np.clip(recent_weight, 0.20, 0.80))
    out = (1.0 - w) * np.asarray(full_p, dtype=float) + w * np.asarray(recent_p, dtype=float)
    out = np.clip(out, 1e-7, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def evaluate(horizon: str):
    rows = load_archive_research_rows(horizon, MAX_ROWS)
    if len(rows) < MIN_TRAIN + TEST_BLOCK * MIN_BLOCKS + 200:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_archive_rows",
            "n": len(rows),
            "promotion_evidence_eligible": False,
        }

    rows = sorted(rows, key=lambda r: (str(r.get("created", "")), str(r.get("id", ""))))
    split = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    development = rows[:split]
    holdout = rows[split:]

    champion_path = ROOT / "models" / f"{horizon}.joblib"
    if not champion_path.is_file():
        return {"status": "DEFERRED", "reason": "champion_artifact_missing", "n": len(rows)}
    try:
        champion = joblib.load(champion_path)
    except Exception as exc:
        return {"status": "DEFERRED", "reason": "champion_artifact_load_failed", "error": f"{type(exc).__name__}:{exc}"}

    blocks = []
    history = []
    for end in range(MIN_TRAIN, len(development), TEST_BLOCK):
        train_end = max(0, end - PURGE_BARS[horizon] - EMBARGO_BARS[horizon])
        train = development[:train_end]
        test = development[end:min(end + TEST_BLOCK, len(development))]
        if len(train) < MIN_TRAIN or len(test) < max(100, TEST_BLOCK // 2):
            continue

        recent_n = max(MIN_RECENT_TRAIN, int(len(train) * RECENT_FRACTION))
        if recent_n > len(train):
            recent_n = len(train)
        recent_train = train[-recent_n:]

        full_p = _fit_calibrated(train, test)
        recent_p = _fit_calibrated(recent_train, test)
        if full_p is None or recent_p is None:
            continue

        recent_weight = _adaptive_recent_weight(history)
        dynamic_p = _mix(full_p, recent_p, recent_weight)
        champion_p = _predict_champion(champion, test)
        y = [r["y"] for r in test]

        full_m = metrics(y, full_p)
        recent_m = metrics(y, recent_p)
        dynamic_m = metrics(y, dynamic_p)
        champion_m = metrics(y, champion_p)

        history.append({
            "recent_logloss": recent_m["logloss"],
            "full_logloss": full_m["logloss"],
            "recent_brier": recent_m["brier"],
            "full_brier": full_m["brier"],
        })
        blocks.append({
            "n": len(test),
            "start": str(test[0].get("created", "")),
            "end": str(test[-1].get("created", "")),
            "train_n": len(train),
            "recent_train_n": recent_n,
            "recent_fraction": float(recent_n / max(len(train), 1)),
            "recent_weight_before_block": recent_weight,
            "full_memory": full_m,
            "recent_memory": recent_m,
            "dynamic": dynamic_m,
            "champion": champion_m,
            "delta_dynamic_vs_champion": {
                "accuracy": dynamic_m["accuracy"] - champion_m["accuracy"],
                "logloss": dynamic_m["logloss"] - champion_m["logloss"],
                "brier": dynamic_m["brier"] - champion_m["brier"],
            },
            "delta_dynamic_vs_full_memory": {
                "accuracy": dynamic_m["accuracy"] - full_m["accuracy"],
                "logloss": dynamic_m["logloss"] - full_m["logloss"],
                "brier": dynamic_m["brier"] - full_m["brier"],
            },
        })

    if len(blocks) < MIN_BLOCKS:
        return {"status": "DEFERRED", "reason": "too_few_valid_oos_blocks", "n": len(rows)}

    d_ll = np.asarray([b["delta_dynamic_vs_champion"]["logloss"] for b in blocks], dtype=float)
    d_br = np.asarray([b["delta_dynamic_vs_champion"]["brier"] for b in blocks], dtype=float)
    d_ac = np.asarray([b["delta_dynamic_vs_champion"]["accuracy"] for b in blocks], dtype=float)
    r_ll = np.asarray([b["delta_dynamic_vs_full_memory"]["logloss"] for b in blocks], dtype=float)
    r_br = np.asarray([b["delta_dynamic_vs_full_memory"]["brier"] for b in blocks], dtype=float)

    hold_full = _fit_calibrated(development, holdout)
    hold_recent = _fit_calibrated(development[-max(MIN_RECENT_TRAIN, int(len(development) * RECENT_FRACTION)):], holdout)
    if hold_full is None or hold_recent is None:
        return {"status": "DEFERRED", "reason": "holdout_prediction_failed", "n": len(rows)}
    final_weight = _adaptive_recent_weight(history)
    hold_dynamic = _mix(hold_full, hold_recent, final_weight)
    hold_champion = _predict_champion(champion, holdout)
    y_hold = [r["y"] for r in holdout]

    dynamic_gate = bool(
        len(blocks) >= MIN_BLOCKS
        and float(np.mean(d_ll < 0)) >= 0.60
        and float(np.mean(d_br < 0)) >= 0.60
        and float(np.mean(d_ac >= -0.005)) >= 0.80
        and float(np.mean(d_ll)) <= -0.003
        and float(np.mean(d_br)) <= -0.0015
    )

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "policy": "causal_dual_memory_full_history_plus_recent_50pct_soft_routing",
        "recent_fraction_target": RECENT_FRACTION,
        "summary": {
            "blocks": len(blocks),
            "samples": int(sum(b["n"] for b in blocks)),
            "dynamic_mean_accuracy_delta_vs_champion": float(d_ac.mean()),
            "dynamic_mean_logloss_delta_vs_champion": float(d_ll.mean()),
            "dynamic_mean_brier_delta_vs_champion": float(d_br.mean()),
            "dynamic_improved_logloss_ratio": float(np.mean(d_ll < 0)),
            "dynamic_improved_brier_ratio": float(np.mean(d_br < 0)),
            "dynamic_non_worse_accuracy_ratio": float(np.mean(d_ac >= -0.005)),
            "dynamic_mean_logloss_delta_vs_full_memory": float(r_ll.mean()),
            "dynamic_mean_brier_delta_vs_full_memory": float(r_br.mean()),
            "dynamic_improved_vs_full_logloss_ratio": float(np.mean(r_ll < 0)),
            "dynamic_improved_vs_full_brier_ratio": float(np.mean(r_br < 0)),
            "final_recent_weight": final_weight,
            "promotion_style_gate": dynamic_gate,
        },
        "final_holdout_protected": True,
        "final_holdout_used_for_selection": False,
        "final_holdout": {
            "champion": metrics(y_hold, hold_champion),
            "full_memory": metrics(y_hold, hold_full),
            "recent_memory": metrics(y_hold, hold_recent),
            "dynamic": metrics(y_hold, hold_dynamic),
            "recent_weight": final_weight,
        },
        "blocks": blocks,
    }


def main():
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "horizons": {h: evaluate(h) for h in HORIZONS},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
