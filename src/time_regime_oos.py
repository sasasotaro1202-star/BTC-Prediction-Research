"""Research-only OOS evaluation of deterministic 24/7 crypto time-regime features.

Adds only UTC intraday/weekly cyclical features and a weekend indicator. The
final holdout is descriptive only; no production artifact is modified.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from feature_schema import FEATURES
from model_compare import (
    HORIZONS,
    MIN_TRAIN,
    MIN_OOS,
    TEST_BLOCK,
    PURGE_BARS,
    EMBARGO_BARS,
    aligned,
    metrics,
    walk_forward,
    _adjusted_alpha,
    loss_arrays,
    hac_test,
    load_primary_production_strict_rows,
    load_archive_research_rows,
)

OUT = ROOT / "data" / "historical_research" / "time_regime_oos.json"
FINAL_HOLDOUT_FRAC = 0.20
EXTRA_FEATURES = (
    "utc_minute_sin",
    "utc_minute_cos",
    "utc_week_sin",
    "utc_week_cos",
    "utc_weekend",
)

def _timestamp_features(value: str) -> list[float]:
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    minute_of_week = dt.weekday() * 1440 + dt.hour * 60 + dt.minute
    minute_of_day = dt.hour * 60 + dt.minute
    return [
        math.sin(2.0 * math.pi * minute_of_day / 1440.0),
        math.cos(2.0 * math.pi * minute_of_day / 1440.0),
        math.sin(2.0 * math.pi * minute_of_week / 10080.0),
        math.cos(2.0 * math.pi * minute_of_week / 10080.0),
        1.0 if dt.weekday() >= 5 else 0.0,
    ]

def _augment(rows):
    out = []
    for row in rows:
        try:
            extra = _timestamp_features(row["created"])
            x = np.asarray(row["x"], dtype=float)
            if x.shape != (len(FEATURES),) or not np.isfinite(x).all():
                continue
            out.append({**row, "x": np.concatenate([x, np.asarray(extra, dtype=float)]).tolist()})
        except (TypeError, ValueError, OverflowError):
            continue
    return out

def _freeze_archive_to_champion(horizon, rows):
    """Attach frozen Champion probabilities to only post-training archive rows."""
    import json
    import joblib

    model_path = ROOT / "models" / f"{horizon}.joblib"
    meta_path = ROOT / "models" / f"{horizon}.json"
    if not model_path.is_file() or not meta_path.is_file():
        return []
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        trained_raw = meta.get("trained_at_utc")
        trained_at = (
            datetime.fromisoformat(str(trained_raw).replace("Z", "+00:00"))
            if trained_raw else None
        )
        if trained_at is None:
            return []
        model = joblib.load(model_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return []

    out = []
    for row in rows:
        try:
            created = datetime.fromisoformat(str(row["created"]).replace("Z", "+00:00"))
            if created <= trained_at:
                continue
            x = np.asarray([row["x"]], dtype=float)
            p = aligned(model, x)[0].tolist()
            out.append({**row, "production": p})
        except (TypeError, ValueError, OverflowError, OSError):
            continue
    return out


def factories():
    return {
        "logreg": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.20, max_iter=3000)),
        ]),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=420,
            max_depth=10,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=240,
            max_leaf_nodes=15,
            learning_rate=0.035,
            l2_regularization=1.5,
            random_state=42,
        ),
    }

def _candidate_holdout(factory, train, holdout):
    if len(train) < MIN_TRAIN or len(holdout) < 100:
        return None
    split = max(int(len(train) * 0.75), MIN_TRAIN - 100)
    if len(train) - split < 50:
        return None
    cal = factory()
    cal.fit(
        np.asarray([r["x"] for r in train[:split]], dtype=float),
        np.asarray([r["y"] for r in train[:split]]),
    )
    cal_probs = aligned(
        cal,
        np.asarray([r["x"] for r in train[split:]], dtype=float),
    )
    from model_compare import _temperature, apply_temperature
    temp = _temperature(cal_probs, [r["y"] for r in train[split:]])
    model = factory()
    model.fit(
        np.asarray([r["x"] for r in train], dtype=float),
        np.asarray([r["y"] for r in train]),
    )
    raw = aligned(model, np.asarray([r["x"] for r in holdout], dtype=float))
    return apply_temperature(raw, temp)

def evaluate(horizon: str):
    rows = load_primary_production_strict_rows(horizon)
    source = "live_binance_primary"
    evidence_eligible = True

    if len(rows) < MIN_TRAIN + MIN_OOS:
        archive = load_archive_research_rows(
            horizon,
            max(12000, MIN_TRAIN + MIN_OOS + 200),
        )
        archive = _freeze_archive_to_champion(horizon, archive)
        if len(archive) > len(rows):
            rows = archive
            source = "binance_vision_archive_frozen_champion"
            evidence_eligible = False

    if len(rows) < MIN_TRAIN + MIN_OOS:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_chronological_rows",
            "n": len(rows),
            "data_source": source,
            "promotion_evidence_eligible": evidence_eligible,
        }

    rows = sorted(_augment(rows), key=lambda r: (str(r["created"]), int(r["id"])))
    if len(rows) < MIN_TRAIN + MIN_OOS:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_valid_augmented_rows",
            "n": len(rows),
            "data_source": source,
            "promotion_evidence_eligible": evidence_eligible,
        }

    split = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    development, holdout = rows[:split], rows[split:]
    production_by_id = {r["id"]: r["production"] for r in rows}

    results = {}
    alpha = _adjusted_alpha(0.05, len(factories()))
    for name, factory in factories().items():
        wf = walk_forward(development, factory, horizon)
        if wf is None:
            continue
        prod = [production_by_id[i] for i in wf["ids"]]
        cm = metrics(wf["ys"], wf["probs"])
        pm = metrics(wf["ys"], prod)
        diffs = loss_arrays(wf["ys"], prod, wf["probs"])
        stats = {
            key: hac_test(value, PURGE_BARS[horizon], alpha=alpha)
            for key, value in diffs.items()
        }

        block_ll, block_br, block_ac = [], [], []
        for start in range(0, len(wf["ys"]), TEST_BLOCK):
            ys = wf["ys"][start:start + TEST_BLOCK]
            if len(ys) < max(10, TEST_BLOCK // 2):
                continue
            cand = metrics(ys, wf["probs"][start:start + TEST_BLOCK])
            base = metrics(ys, prod[start:start + TEST_BLOCK])
            block_ll.append(cand["logloss"] - base["logloss"])
            block_br.append(cand["brier"] - base["brier"])
            block_ac.append(cand["accuracy"] - base["accuracy"])

        eligible = bool(
            len(block_ll) >= 8
            and float(np.mean(np.asarray(block_ll) < 0)) >= 0.60
            and float(np.mean(np.asarray(block_br) < 0)) >= 0.60
            and cm["logloss"] <= pm["logloss"] - 0.003
            and cm["brier"] <= pm["brier"] - 0.0015
            and cm["accuracy"] >= pm["accuracy"] - 0.005
            and stats["logloss"]["significant"]
            and stats["brier"]["significant"]
        )

        item = {
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
                "blocks": len(block_ll),
                "improved_logloss_ratio": float(np.mean(np.asarray(block_ll) < 0)) if block_ll else 0.0,
                "improved_brier_ratio": float(np.mean(np.asarray(block_br) < 0)) if block_br else 0.0,
                "non_worse_accuracy_ratio": float(np.mean(np.asarray(block_ac) >= -0.005)) if block_ac else 0.0,
            },
            "statistical_tests": stats,
            "eligible_pending_frozen_holdout_confirmation": eligible,
        }
        hold = _candidate_holdout(factory, development, holdout)
        if hold is None:
            item["final_holdout"] = {"status": "DEFERRED"}
        else:
            y_hold = [r["y"] for r in holdout]
            p_hold = [r["production"] for r in holdout]
            cand_hold = metrics(y_hold, hold)
            prod_hold = metrics(y_hold, p_hold)
            item["final_holdout"] = {
                "candidate": cand_hold,
                "production": prod_hold,
                "delta": {
                    "accuracy": cand_hold["accuracy"] - prod_hold["accuracy"],
                    "logloss": cand_hold["logloss"] - prod_hold["logloss"],
                    "brier": cand_hold["brier"] - prod_hold["brier"],
                },
            }
        results[name] = item

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "final_holdout_protected": True,
        "final_holdout_used_for_selection": False,
        "data_source": source,
        "promotion_evidence_eligible": evidence_eligible,
        "n": len(rows),
        "development_n": len(development),
        "final_holdout_n": len(holdout),
        "extra_features": list(EXTRA_FEATURES),
        "corrected_alpha": alpha,
        "candidates": results,
    }

def main():
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "feature_schema": list(FEATURES),
        "extra_features": list(EXTRA_FEATURES),
        "horizons": {h: evaluate(h) for h in HORIZONS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
