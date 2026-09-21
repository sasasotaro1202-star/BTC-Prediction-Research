"""Research-only rolling OOS test for adaptive ensemble weighting."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from model_compare import HORIZONS, load_rows, metrics, apply_temperature, _temperature
from ensemble_model import SoftVotingEnsemble

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "adaptive_ensemble_oos.json"
MIN_TRAIN = 2000
TEST_BLOCK = 150
MAX_BLOCKS = 16
MAX_ROWS = 7000
PURGE = {"5m": 5, "10m": 10}
EMBARGO = {"5m": 60, "10m": 60}


def _endpoints(n: int):
    raw = list(range(MIN_TRAIN, n, TEST_BLOCK))
    if len(raw) <= MAX_BLOCKS:
        return raw
    return sorted(set(int(x) for x in np.linspace(raw[0], raw[-1], MAX_BLOCKS)))


def _aligned(model, rows):
    X = np.asarray([r["x"] for r in rows], dtype=float)
    raw = np.asarray(model.predict_proba(X), dtype=float)
    out = np.full((len(rows), 3), 1e-6, dtype=float)
    idx = {"DOWN": 0, "FLAT": 1, "UP": 2}
    for j, cls in enumerate(model.classes_):
        if str(cls) in idx:
            out[:, idx[str(cls)]] = raw[:, j]
    out = np.clip(out, 1e-6, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def _predict_block(factory, train, test):
    split = int(len(train) * 0.75)
    if split < 120 or len(train) - split < 30:
        return None
    cal_model = factory()
    cal_model.fit(
        np.asarray([r["x"] for r in train[:split]], dtype=float),
        np.asarray([r["y"] for r in train[:split]]),
    )
    cal_probs = _aligned(cal_model, train[split:])
    temp = _temperature(cal_probs, [r["y"] for r in train[split:]])

    model = factory()
    model.fit(
        np.asarray([r["x"] for r in train], dtype=float),
        np.asarray([r["y"] for r in train]),
    )
    probs = _aligned(model, test)
    return apply_temperature(probs, temp)


def evaluate(horizon: str):
    rows = load_rows(horizon)
    if len(rows) > MAX_ROWS:
        rows = rows[-MAX_ROWS:]
    if len(rows) < MIN_TRAIN + TEST_BLOCK:
        return {"status": "DEFERRED", "n": len(rows), "reason": "insufficient_rows"}

    equal_factory = lambda: SoftVotingEnsemble(learn_weights=False)
    adaptive_factory = lambda: SoftVotingEnsemble(learn_weights=True)
    equal_preds = []
    adaptive_preds = []
    ys = []
    blocks = []

    for end in _endpoints(len(rows)):
        train_end = max(0, end - PURGE[horizon] - EMBARGO[horizon])
        test = rows[end:min(end + TEST_BLOCK, len(rows))]
        train = rows[:train_end]
        if len(train) < MIN_TRAIN or len(test) < max(50, TEST_BLOCK // 2):
            continue
        eq = _predict_block(equal_factory, train, test)
        ad = _predict_block(adaptive_factory, train, test)
        if eq is None or ad is None:
            continue
        y = [r["y"] for r in test]
        em = metrics(y, eq)
        am = metrics(y, ad)
        blocks.append({
            "n": len(y),
            "equal": em,
            "adaptive": am,
            "delta": {
                "accuracy": am["accuracy"] - em["accuracy"],
                "logloss": am["logloss"] - em["logloss"],
                "brier": am["brier"] - em["brier"],
            },
        })
        equal_preds.extend(eq.tolist())
        adaptive_preds.extend(ad.tolist())
        ys.extend(y)

    if not blocks:
        return {"status": "DEFERRED", "n": len(rows), "reason": "no_valid_blocks"}

    ll = np.asarray([b["delta"]["logloss"] for b in blocks], dtype=float)
    br = np.asarray([b["delta"]["brier"] for b in blocks], dtype=float)
    ac = np.asarray([b["delta"]["accuracy"] for b in blocks], dtype=float)
    summary = {
        "blocks": len(blocks),
        "samples": int(sum(b["n"] for b in blocks)),
        "mean_accuracy_delta": float(ac.mean()),
        "mean_logloss_delta": float(ll.mean()),
        "mean_brier_delta": float(br.mean()),
        "improved_logloss_ratio": float(np.mean(ll < 0)),
        "improved_brier_ratio": float(np.mean(br < 0)),
        "non_worse_accuracy_ratio": float(np.mean(ac >= -0.005)),
    }
    eligible = (
        summary["blocks"] >= 8
        and summary["improved_logloss_ratio"] >= 0.60
        and summary["improved_brier_ratio"] >= 0.60
        and summary["mean_logloss_delta"] <= -0.003
        and summary["mean_brier_delta"] <= -0.0015
        and summary["non_worse_accuracy_ratio"] >= 0.80
    )
    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "final_holdout_protected": True,
        "policy": "adaptive_ensemble_weight_learning_on_pre_test_training_window_only",
        "eligible_pending_frozen_holdout_confirmation": bool(eligible),
        "summary": summary,
        "blocks": blocks,
    }


def main():
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "horizons": {h: evaluate(h) for h in HORIZONS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
