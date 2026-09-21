"""Research-only rolling OOS test for adaptive ensemble weighting."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

try:
    from model_compare import HORIZONS, load_rows, metrics, apply_temperature, _temperature
    from ensemble_model import SoftVotingEnsemble
except ModuleNotFoundError:
    # Support both execution modes used by CI and the test suite:
    #   python src/adaptive_ensemble_oos.py
    #   from src import adaptive_ensemble_oos
    from src.model_compare import HORIZONS, load_rows, metrics, apply_temperature, _temperature
    from src.ensemble_model import SoftVotingEnsemble

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
    prefix_y = [r["y"] for r in train[:split]]
    if len(set(prefix_y)) < 3:
        return None
    cal_model = factory()
    try:
        cal_model.fit(
            np.asarray([r["x"] for r in train[:split]], dtype=float),
            np.asarray(prefix_y),
        )
    except ValueError as exc:
        if "three_classes" in str(exc):
            return None
        raise
    cal_probs = _aligned(cal_model, train[split:])
    temp = _temperature(cal_probs, [r["y"] for r in train[split:]])

    model = factory()
    model.fit(
        np.asarray([r["x"] for r in train], dtype=float),
        np.asarray([r["y"] for r in train]),
    )
    probs = _aligned(model, test)
    return apply_temperature(probs, temp)


def _evaluate_development(rows, horizon: str):
    equal_factory = lambda: SoftVotingEnsemble(learn_weights=False)
    adaptive_factory = lambda: SoftVotingEnsemble(learn_weights=True)
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
    if not blocks:
        return None

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
    return summary, blocks, bool(eligible)


def evaluate(horizon: str):
    rows = load_rows(horizon)
    if len(rows) > MAX_ROWS:
        rows = rows[-MAX_ROWS:]

    if len(rows) < MIN_TRAIN + TEST_BLOCK + 100:
        return {
            "status": "DEFERRED",
            "n": len(rows),
            "reason": "insufficient_rows_for_protected_holdout",
        }

    split = int(len(rows) * 0.80)
    development = rows[:split]
    holdout = rows[split:]
    if len(development) < MIN_TRAIN + TEST_BLOCK:
        return {
            "status": "DEFERRED",
            "n": len(rows),
            "reason": "insufficient_development_rows",
        }

    dev_result = _evaluate_development(development, horizon)
    if dev_result is None:
        return {
            "status": "DEFERRED",
            "n": len(rows),
            "reason": "no_valid_development_blocks",
        }
    summary, blocks, eligible = dev_result

    # Final 20% is protected. No threshold, weight, or eligibility decision
    # is made from it; it is evaluated once using the frozen development
    # protocol and the full development history as training data.
    equal_factory = lambda: SoftVotingEnsemble(learn_weights=False)
    adaptive_factory = lambda: SoftVotingEnsemble(learn_weights=True)
    eq = _predict_block(equal_factory, development, holdout)
    ad = _predict_block(adaptive_factory, development, holdout)
    if eq is None or ad is None:
        final_holdout = {
            "status": "DEFERRED",
            "reason": "holdout_prediction_failed",
        }
    else:
        y_hold = [r["y"] for r in holdout]
        em = metrics(y_hold, eq)
        am = metrics(y_hold, ad)
        final_holdout = {
            "equal": em,
            "adaptive": am,
            "delta": {
                "accuracy": am["accuracy"] - em["accuracy"],
                "logloss": am["logloss"] - em["logloss"],
                "brier": am["brier"] - em["brier"],
            },
        }

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "final_holdout_protected": True,
        "final_holdout_used_for_selection": False,
        "policy": "adaptive_ensemble_weight_learning_on_pre_test_training_window_only",
        "eligible_pending_frozen_holdout_confirmation": bool(eligible),
        "summary": summary,
        "development_n": len(development),
        "final_holdout_n": len(holdout),
        "final_holdout": final_holdout,
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
