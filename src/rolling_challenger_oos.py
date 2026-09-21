"""Continuous research-only rolling challenger evaluation on the current production generation."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from db import DB, init_db
    from ensemble_model import SoftVotingEnsemble
    from model_compare import (
    HORIZONS, CLASSES, MIN_TRAIN, MIN_OOS, TEST_BLOCK,
    metrics, normalize, aligned, _temperature, apply_temperature,
    hac_test, _adjusted_alpha,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "rolling_challenger_oos.json"
MAX_ROWS = 12000
FINAL_HOLDOUT_FRAC = 0.20
PURGE = {"5m": 5, "10m": 10}
EMBARGO = {"5m": 60, "10m": 60}
ALPHA = 0.05


def current_generation(horizon: str) -> str | None:
    init_db()
    with sqlite3.connect(DB) as con:
        row = con.execute(
            "SELECT production_version FROM model_registry WHERE horizon=?",
            (horizon,),
        ).fetchone()
    return str(row[0]) if row and row[0] else None


def load_current_rows(horizon: str):
    from model_compare import load_rows
    version = current_generation(horizon)
    if not version:
        return [], None
    prefix = f"{horizon}:{version}|%"
    rows = [r for r in load_rows(horizon) if str(r.get("model_version", "")).startswith(prefix)]
    rows = rows[-MAX_ROWS:]
    return rows, version


def factories():
    return {
        "logreg_c0.1": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.1, max_iter=3000)),
        ]),
        "extra_trees_500": lambda: ExtraTreesClassifier(
            n_estimators=500, max_depth=7, min_samples_leaf=8,
            max_features="sqrt", random_state=42, n_jobs=-1,
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=250, max_leaf_nodes=15, learning_rate=0.04,
            l2_regularization=1.0, random_state=42,
        ),
        "gaussian_nb": lambda: GaussianNB(),
        "adaptive_soft_ensemble": lambda: SoftVotingEnsemble(learn_weights=True),
    }


def candidate_block_predictions(train, test, factory, horizon):
    if len(train) < MIN_TRAIN or len(set(r["y"] for r in train)) < 3:
        return None
    split = max(int(len(train) * 0.75), MIN_TRAIN - 100)
    if split < 100 or len(train) - split < 30:
        return None
    cal_train, cal_rows = train[:split], train[split:]
    if len(set(r["y"] for r in cal_train)) < 3:
        return None
    try:
        cal_model = factory()
        cal_model.fit(
            np.asarray([r["x"] for r in cal_train], dtype=float),
            np.asarray([r["y"] for r in cal_train]),
        )
        cal_probs = aligned(cal_model, np.asarray([r["x"] for r in cal_rows], dtype=float))
        temperature = _temperature(cal_probs, [r["y"] for r in cal_rows])

        model = factory()
        model.fit(
            np.asarray([r["x"] for r in train], dtype=float),
            np.asarray([r["y"] for r in train]),
        )
        raw = aligned(model, np.asarray([r["x"] for r in test], dtype=float))
        return apply_temperature(raw, temperature)
    except (ValueError, RuntimeError):
        return None


def loss_diff(y, production, candidate):
    yi = np.asarray([CLASSES.index(v) for v in y], dtype=int)
    pp = normalize(production)
    cp = normalize(candidate)
    return {
        "logloss": -np.log(np.clip(cp[np.arange(len(yi)), yi], 1e-12, 1.0))
                     + np.log(np.clip(pp[np.arange(len(yi)), yi], 1e-12, 1.0)),
        "brier": (
            np.sum((cp - np.eye(3)[yi]) ** 2, axis=1)
            - np.sum((pp - np.eye(3)[yi]) ** 2, axis=1)
        ),
    }


def evaluate(horizon: str):
    rows, version = load_current_rows(horizon)
    if len(rows) < MIN_TRAIN + MIN_OOS:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_current_generation_rows",
            "n": len(rows),
            "model_version": version,
        }

    split = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    development = rows[:split]
    holdout = rows[split:]
    if len(development) < MIN_TRAIN + MIN_OOS:
        return {"status": "DEFERRED", "reason": "insufficient_development_rows", "n": len(rows), "model_version": version}

    prod_by_id = {r["id"]: r["production"] for r in rows}
    candidate_results = {}
    cand_factories = factories()
    corrected_alpha = _adjusted_alpha(ALPHA, len(cand_factories))

    for name, factory in cand_factories.items():
        ys, prod_probs, cand_probs = [], [], []
        block_rows = []
        for end in range(MIN_TRAIN, len(development), TEST_BLOCK):
            train_end = max(0, end - PURGE[horizon] - EMBARGO[horizon])
            train = development[:train_end]
            test = development[end:min(end + TEST_BLOCK, len(development))]
            if len(train) < MIN_TRAIN or len(test) < max(10, TEST_BLOCK // 2):
                continue
            cp = candidate_block_predictions(train, test, factory, horizon)
            if cp is None:
                continue
            y = [r["y"] for r in test]
            pp = [prod_by_id[r["id"]] for r in test]
            cm = metrics(y, cp)
            pm = metrics(y, pp)
            block_rows.append({
                "n": len(y),
                "logloss_delta": cm["logloss"] - pm["logloss"],
                "brier_delta": cm["brier"] - pm["brier"],
                "accuracy_delta": cm["accuracy"] - pm["accuracy"],
            })
            ys.extend(y)
            prod_probs.extend(pp)
            cand_probs.extend(cp.tolist())

        if len(ys) < MIN_OOS:
            continue
        cm = metrics(ys, cand_probs)
        pm = metrics(ys, prod_probs)
        diffs = loss_diff(ys, prod_probs, cand_probs)
        stats = {
            k: hac_test(v, PURGE[horizon], alpha=corrected_alpha)
            for k, v in diffs.items()
        }
        ll_d = np.asarray([b["logloss_delta"] for b in block_rows], dtype=float)
        br_d = np.asarray([b["brier_delta"] for b in block_rows], dtype=float)
        ac_d = np.asarray([b["accuracy_delta"] for b in block_rows], dtype=float)
        stability = {
            "blocks": len(block_rows),
            "improved_logloss_ratio": float(np.mean(ll_d < 0)),
            "improved_brier_ratio": float(np.mean(br_d < 0)),
            "non_worse_accuracy_ratio": float(np.mean(ac_d >= -0.005)),
        }
        eligible = (
            stability["blocks"] >= 8
            and stability["improved_logloss_ratio"] >= 0.60
            and stability["improved_brier_ratio"] >= 0.60
            and cm["logloss"] <= pm["logloss"] - 0.003
            and cm["brier"] <= pm["brier"] - 0.0015
            and cm["accuracy"] >= pm["accuracy"] - 0.005
            and bool(stats["logloss"]["significant"])
            and bool(stats["brier"]["significant"])
        )
        candidate_results[name] = {
            "development": {
                "candidate": cm,
                "production": pm,
                "delta": {
                    "accuracy": cm["accuracy"] - pm["accuracy"],
                    "logloss": cm["logloss"] - pm["logloss"],
                    "brier": cm["brier"] - pm["brier"],
                },
            },
            "block_stability": stability,
            "statistical_tests": stats,
            "eligible_pending_frozen_holdout_confirmation": bool(eligible),
            "blocks": block_rows,
        }

    # The latest 20% is protected and only evaluated for descriptive evidence.
    holdout_prod = metrics([r["y"] for r in holdout], [r["production"] for r in holdout])
    for name, factory in cand_factories.items():
        if name not in candidate_results:
            continue
        hp = candidate_block_predictions(development, holdout, factory, horizon)
        if hp is None:
            candidate_results[name]["final_holdout"] = {"status": "DEFERRED"}
        else:
            candidate_results[name]["final_holdout"] = {
                "candidate": metrics([r["y"] for r in holdout], hp),
                "production": holdout_prod,
            }

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "final_holdout_protected": True,
        "model_version": version,
        "n": len(rows),
        "development_n": len(development),
        "final_holdout_n": len(holdout),
        "alpha": corrected_alpha,
        "candidates": candidate_results,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
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
