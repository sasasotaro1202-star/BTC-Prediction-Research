"""Research-only fresh OOS model zoo for the Coinbase fallback domain."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from bootstrap_train import FEATURES, build_dataset
from coinbase_fallback_train import fetch_coinbase

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUT = ROOT / "data" / "historical_research" / "coinbase_fallback_model_zoo_oos.json"
CLASSES = ["DOWN", "FLAT", "UP"]

TARGET_ROWS = 30000
FINAL_HOLDOUT_FRAC = 0.20
PURGE = {"5m": 5, "10m": 10}
BLOCKS = 6
MIN_BLOCK = 250


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    yi = np.asarray([CLASSES.index(str(v)) for v in y])
    pred = p.argmax(1)
    return {
        "accuracy": float(accuracy_score(yi, pred)),
        "logloss": float(log_loss(yi, p, labels=[0, 1, 2])),
        "n": int(len(yi)),
    }


def candidates(seed=42):
    return {
        "rf": RandomForestClassifier(
            n_estimators=320, max_depth=8, min_samples_leaf=4,
            class_weight="balanced_subsample", random_state=seed,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=320, max_depth=8, min_samples_leaf=4,
            class_weight="balanced", random_state=seed,
            n_jobs=-1,
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=220, learning_rate=0.045, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=seed,
        ),
        "logreg": LogisticRegression(
            C=0.25, max_iter=2000, multi_class="auto",
            class_weight="balanced", random_state=seed,
        ),
    }


def evaluate_horizon(X, y, horizon):
    n = len(y)
    holdout_start = int(n * (1.0 - FINAL_HOLDOUT_FRAC))
    development = np.arange(0, holdout_start)
    holdout = np.arange(holdout_start, n)
    if len(development) < 1800 or len(holdout) < 800:
        return {"status": "DEFERRED", "reason": "insufficient_fresh_coinbase_history"}

    block_edges = np.linspace(0, len(development), BLOCKS + 1, dtype=int)
    records = {}
    for name in candidates().keys():
        block_rows = []
        for b in range(BLOCKS):
            test_start = int(block_edges[b])
            test_end = int(block_edges[b + 1])
            if test_end - test_start < MIN_BLOCK:
                continue
            train_end = max(0, test_start - PURGE[horizon])
            if train_end < 1000:
                continue
            model = candidates()[name]
            model.fit(X[:train_end], y[:train_end])
            p = model.predict_proba(X[test_start:test_end])
            block_rows.append(metrics(y[test_start:test_end], p))

        if not block_rows:
            continue

        et = candidates()[name]
        et.fit(X[:holdout_start - PURGE[horizon]], y[:holdout_start - PURGE[horizon]])
        p_hold = et.predict_proba(X[holdout])
        hold = metrics(y[holdout], p_hold)
        records[name] = {
            "blocks": block_rows,
            "mean_block_logloss": float(np.mean([x["logloss"] for x in block_rows])),
            "mean_block_accuracy": float(np.mean([x["accuracy"] for x in block_rows])),
            "block_logloss_improved_ratio": float(np.mean([
                x["logloss"] <= np.mean([z["logloss"] for z in block_rows]) for x in block_rows
            ])),
            "holdout": hold,
            "model": et,
        }

    champion = joblib.load(MODEL_DIR / f"coinbase_{horizon}.joblib")
    frozen_p = champion.predict_proba(X[holdout])
    frozen = metrics(y[holdout], frozen_p)
    for rec in records.values():
        rec.pop("model", None)

    eligible = []
    for name, rec in records.items():
        block_ll = np.asarray([x["logloss"] for x in rec["blocks"]])
        block_acc = np.asarray([x["accuracy"] for x in rec["blocks"]])
        rf_blocks = records["rf"]["blocks"][:len(block_ll)] if "rf" in records else []
        rf_ll = np.asarray([x["logloss"] for x in rf_blocks], dtype=float)
        rf_acc = np.asarray([x["accuracy"] for x in rf_blocks], dtype=float)
        ll_ratio = float(np.mean(block_ll < rf_ll)) if len(rf_ll) == len(block_ll) else 0.0
        acc_ratio = float(np.mean(block_acc >= (rf_acc - 0.005))) if len(rf_acc) == len(block_acc) else 0.0
        if len(block_ll) >= 4 and ll_ratio >= 0.50 and acc_ratio >= 0.50:
            eligible.append(name)

    return {
        "status": "OK",
        "n": n,
        "development_n": len(development),
        "holdout_n": len(holdout),
        "final_holdout_protected": True,
        "final_holdout_used_for_selection": False,
        "current_frozen_champion": frozen,
        "candidates": records,
        "eligible_research_candidates": eligible,
        "promotion_evidence_eligible": False,
    }


def main():
    rows = fetch_coinbase(TARGET_ROWS)
    out = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "source": "Coinbase Exchange BTC-USD 1m closed candles",
        "rows": len(rows),
        "feature_schema": list(FEATURES),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "horizons": {},
    }
    for horizon in ("5m", "10m"):
        X, y = build_dataset(rows, int(horizon[:-1]))
        out["horizons"][horizon] = evaluate_horizon(X, y, horizon)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
