"""Research-only OOS test for extending the BTC feature schema.

Tests whether already-persisted cutoff features that are currently excluded from
the production schema improve the current production model under chronological
walk-forward evaluation. No production artifact is changed.
"""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

from feature_schema import FEATURES
from model_compare import CLASSES, TEST_BLOCK, MIN_TRAIN, MIN_OOS, PURGE_BARS, EMBARGO_BARS, metrics, aligned, apply_temperature, _temperature

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "predictions.db"
OUT = ROOT / "data" / "historical_research" / "extended_features_oos.json"

EXTRA_FEATURES = ("ret_15m", "ret_30m", "range_position_30m", "trend_alignment")
ALL_FEATURES = tuple(FEATURES) + EXTRA_FEATURES
FINAL_HOLDOUT_FRAC = 0.20


def _norm(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def _metrics(y, p):
    return metrics(y, p)


def load_rows(horizon: str):
    actual = f"actual_direction_{horizon}"
    pu, pd, pf = f"p_up_{horizon}", f"p_down_{horizon}", f"p_flat_{horizon}"
    with sqlite3.connect(DB) as con:
        raw = con.execute(
            f"""SELECT prediction_id,created_at_utc,feature_json,{actual},{pu},{pd},{pf}
                FROM predictions
                WHERE {actual} IS NOT NULL
                ORDER BY created_at_utc,prediction_id"""
        ).fetchall()

    out = []
    for rid, created, feature_json, y, up, down, flat in raw:
        if y not in CLASSES:
            continue
        try:
            f = json.loads(feature_json or "{}")
            base = [float(f[k]) for k in FEATURES]
            extra = [float(f[k]) for k in EXTRA_FEATURES]
            production = [float(down), float(flat), float(up)]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        vals = base + extra + production
        if not all(math.isfinite(x) for x in vals):
            continue
        if min(production) < 0 or sum(production) <= 0:
            continue
        out.append({
            "id": rid,
            "created": created,
            "base": base,
            "extended": base + extra,
            "y": y,
            "production": _norm([production])[0].tolist(),
        })
    return out


def factories():
    return {
        "logreg_extended": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.3, max_iter=3000)),
        ]),
        "extra_trees_extended": lambda: ExtraTreesClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "hgb_extended": lambda: HistGradientBoostingClassifier(
            max_iter=220,
            max_leaf_nodes=15,
            learning_rate=0.04,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


def candidate_block(factory, train, test):
    if len(train) < MIN_TRAIN or len(set(r["y"] for r in train)) < 3:
        return None
    split = max(int(len(train) * 0.75), MIN_TRAIN - 100)
    if split < MIN_TRAIN - 200 or len(train) - split < 30:
        return None
    cal_train = train[:split]
    cal_eval = train[split:]
    if len(set(r["y"] for r in cal_train)) < 3:
        return None

    cal_model = factory()
    cal_model.fit(np.asarray([r["extended"] for r in cal_train], dtype=float),
                  np.asarray([r["y"] for r in cal_train]))
    cal_probs = aligned(cal_model, np.asarray([r["extended"] for r in cal_eval], dtype=float))
    temperature = _temperature(cal_probs, [r["y"] for r in cal_eval])

    model = factory()
    model.fit(np.asarray([r["extended"] for r in train], dtype=float),
              np.asarray([r["y"] for r in train]))
    probs = aligned(model, np.asarray([r["extended"] for r in test], dtype=float))
    return apply_temperature(probs, temperature)


def evaluate(horizon: str):
    rows = load_rows(horizon)
    if len(rows) < MIN_TRAIN + MIN_OOS:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_rows",
            "n": len(rows),
        }

    split = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    development, holdout = rows[:split], rows[split:]
    if len(development) < MIN_TRAIN + MIN_OOS:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_development_rows",
            "n": len(rows),
        }

    prod_by_id = {r["id"]: r["production"] for r in rows}
    results = {}

    for name, factory in factories().items():
        ys, prod_probs, cand_probs, blocks = [], [], [], []
        for end in range(MIN_TRAIN, len(development), TEST_BLOCK):
            train_end = max(0, end - PURGE_BARS[horizon] - EMBARGO_BARS[horizon])
            train = development[:train_end]
            test = development[end:min(end + TEST_BLOCK, len(development))]
            if len(train) < MIN_TRAIN or len(test) < max(10, TEST_BLOCK // 2):
                continue
            cp = candidate_block(factory, train, test)
            if cp is None:
                continue
            y = [r["y"] for r in test]
            pp = [prod_by_id[r["id"]] for r in test]
            cm = _metrics(y, cp)
            pm = _metrics(y, pp)
            blocks.append({
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

        cm = _metrics(ys, cand_probs)
        pm = _metrics(ys, prod_probs)
        ll = np.asarray([b["logloss_delta"] for b in blocks], dtype=float)
        br = np.asarray([b["brier_delta"] for b in blocks], dtype=float)
        ac = np.asarray([b["accuracy_delta"] for b in blocks], dtype=float)
        results[name] = {
            "development": {
                "candidate": cm,
                "production": pm,
                "delta": {
                    "accuracy": cm["accuracy"] - pm["accuracy"],
                    "logloss": cm["logloss"] - pm["logloss"],
                    "brier": cm["brier"] - pm["brier"],
                },
            },
            "block_stability": {
                "blocks": len(blocks),
                "improved_logloss_ratio": float(np.mean(ll < 0)) if len(ll) else 0.0,
                "improved_brier_ratio": float(np.mean(br < 0)) if len(br) else 0.0,
                "non_worse_accuracy_ratio": float(np.mean(ac >= -0.005)) if len(ac) else 0.0,
            },
            "blocks_detail": blocks,
            "eligible_pending_frozen_holdout_confirmation": bool(
                len(blocks) >= 8
                and float(np.mean(ll < 0)) >= 0.60
                and float(np.mean(br < 0)) >= 0.60
                and cm["logloss"] <= pm["logloss"] - 0.003
                and cm["brier"] <= pm["brier"] - 0.0015
                and cm["accuracy"] >= pm["accuracy"] - 0.005
            ),
        }

    holdout_result = {
        "production": _metrics([r["y"] for r in holdout], [r["production"] for r in holdout]),
        "candidate_evaluations": {},
    }
    for name, factory in factories().items():
        if name not in results:
            continue
        cp = candidate_block(factory, development, holdout)
        if cp is None:
            holdout_result["candidate_evaluations"][name] = {"status": "DEFERRED"}
        else:
            holdout_result["candidate_evaluations"][name] = {
                "candidate": _metrics([r["y"] for r in holdout], cp),
                "production": holdout_result["production"],
            }

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "final_holdout_protected": True,
        "horizon": horizon,
        "n": len(rows),
        "development_n": len(development),
        "final_holdout_n": len(holdout),
        "base_features": list(FEATURES),
        "extra_features": list(EXTRA_FEATURES),
        "candidate_results": results,
        "final_holdout": holdout_result,
    }


def main():
    if not DB.exists():
        raise SystemExit("prediction database missing")
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "feature_set_policy": "existing_cutoff_persisted_features_only_no_future_inputs",
        "horizons": {h: evaluate(h) for h in ("5m", "10m")},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
