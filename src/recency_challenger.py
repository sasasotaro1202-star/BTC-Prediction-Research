"""Research-only recency challenger for the BTC production model.

Uses the exact 15-feature production schema on the newest contiguous free BTC
1m history. It never writes production model artifacts. Model selection is
chronological and the final holdout is untouched until one descriptive score.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bootstrap_train import fetch_history, make_features
from label_policy import CLASSES, NEUTRAL_RETURN

OUT = ROOT / "data" / "historical_research" / "recency_challenger.json"
CLASSES = tuple(CLASSES)
TARGET_ROWS = 30_000
MIN_ROWS = 20_000
HOLDOUT = 5_000
VALIDATION = 5_000
HORIZONS = {"5m": 5, "10m": 10}


def metrics(y, p):
    yi = np.asarray([CLASSES.index(str(v)) for v in y], dtype=int)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-8, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    hit = np.argmax(p, axis=1) == yi
    one = np.eye(3)[yi]
    ll = -np.mean(np.log(np.clip(p[np.arange(len(y)), yi], 1e-8, 1.0)))
    br = np.mean(np.sum((p - one) ** 2, axis=1))
    return {"n": int(len(y)), "accuracy": float(hit.mean()), "logloss": float(ll), "brier": float(br)}


def factory(name):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=10,
            max_features="sqrt", random_state=42, n_jobs=-1
        )
    if name == "rf_balanced":
        return RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=10,
            max_features="sqrt", class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        )
    if name == "extra":
        return ExtraTreesClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=10,
            max_features="sqrt", random_state=42, n_jobs=-1
        )
    if name == "extra_balanced":
        return ExtraTreesClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=10,
            max_features="sqrt", class_weight="balanced",
            random_state=42, n_jobs=-1
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=300, max_leaf_nodes=31, learning_rate=0.035,
            l2_regularization=1.5, random_state=42
        )
    if name == "logreg":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.5, max_iter=3000)),
        ])
    raise KeyError(name)


def build(rows, horizon):
    X, y, ts = [], [], []
    step = horizon * 60_000
    by_ts = {int(r[0]): r for r in rows}
    for i, row in enumerate(rows):
        t = int(row[0])
        future = by_ts.get(t + step)
        if future is None or i < 30:
            continue
        base = float(row[4])
        future_price = float(future[4])
        if not (math.isfinite(base) and base > 0 and math.isfinite(future_price)):
            continue
        x = make_features(rows[max(0, i - 30): i + 1])
        ret = future_price / base - 1.0
        y.append("UP" if ret > NEUTRAL_RETURN else "DOWN" if ret < -NEUTRAL_RETURN else "FLAT")
        X.append(x)
        ts.append(t)
    return np.asarray(X, float), np.asarray(y, dtype=object), np.asarray(ts, dtype=np.int64)


def bootstrap_ci_accuracy(y, cand_p, base_p, seed=42):
    yi = np.asarray([CLASSES.index(str(v)) for v in y], dtype=int)
    ch = (np.argmax(cand_p, axis=1) == yi).astype(float)
    bh = (np.argmax(base_p, axis=1) == yi).astype(float)
    diff = ch - bh
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(2000, len(diff)))
    means = diff[idx].mean(axis=1)
    return {
        "mean_delta": float(diff.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def evaluate(horizon, rows):
    X, y, ts = build(rows, horizon)
    n = len(y)
    if n < MIN_ROWS:
        raise RuntimeError(f"insufficient_rows:{horizon}:{n}")
    hold_start = n - HOLDOUT
    val_start = hold_start - VALIDATION
    if val_start < 10_000:
        raise RuntimeError(f"insufficient_validation:{horizon}:{n}")
    X_train, y_train = X[:val_start], y[:val_start]
    X_val, y_val = X[val_start:hold_start], y[val_start:hold_start]
    X_hold, y_hold, t_hold = X[hold_start:], y[hold_start:], ts[hold_start:]

    names = ["rf", "rf_balanced", "extra", "extra_balanced", "hgb", "logreg"]
    dev = []
    for name in names:
        m = factory(name)
        m.fit(X_train, y_train)
        p = m.predict_proba(X_val)
        dev.append({"model": name, **metrics(y_val, p)})
    dev.sort(key=lambda z: (-z["accuracy"], z["logloss"], z["brier"]))
    selected = dev[0]["model"]

    # Refit on all development rows only.
    cand = factory(selected)
    cand.fit(X[:hold_start], y[:hold_start])
    cand_p = cand.predict_proba(X_hold)

    champion_path = ROOT / "models" / f"{horizon}m.joblib"
    champion = joblib.load(champion_path)
    champion_p = champion.predict_proba(X_hold)

    counts = {c: int(np.sum(y_hold == c)) for c in CLASSES}
    freq = np.asarray([counts[c] for c in CLASSES], dtype=float)
    freq /= freq.sum()
    freq_p = np.tile(freq, (len(y_hold), 1))

    cm = metrics(y_hold, champion_p)
    rm = metrics(y_hold, cand_p)
    fm = metrics(y_hold, freq_p)
    ci = bootstrap_ci_accuracy(y_hold, cand_p, champion_p)

    accuracy_delta = rm["accuracy"] - cm["accuracy"]
    rel_acc = accuracy_delta / max(abs(cm["accuracy"]), 1e-12)
    rel_ll = (cm["logloss"] - rm["logloss"]) / max(abs(cm["logloss"]), 1e-12)
    rel_br = (cm["brier"] - rm["brier"]) / max(abs(cm["brier"]), 1e-12)
    eligible = bool(
        rm["accuracy"] >= cm["accuracy"] + 0.005
        and rel_acc >= 0.03
        and rel_ll >= 0.03
        and rel_br >= 0.01
        and ci["ci95_low"] > 0.0
        and rm["accuracy"] >= fm["accuracy"]
    )

    return {
        "n": n, "development_n": hold_start, "validation_n": VALIDATION,
        "frozen_holdout_n": HOLDOUT, "selected_model": selected,
        "validation_results": dev,
        "frozen_holdout": {
            "current_champion": cm,
            "recency_candidate": rm,
            "frequency_baseline": fm,
            "accuracy_delta_vs_champion": accuracy_delta,
            "accuracy_relative_gain_vs_champion": rel_acc,
            "logloss_relative_gain_vs_champion": rel_ll,
            "brier_relative_gain_vs_champion": rel_br,
            "accuracy_bootstrap_ci_vs_champion": ci,
            "holdout_class_counts": counts,
            "holdout_start_utc": datetime.fromtimestamp(int(t_hold[0]) / 1000, timezone.utc).isoformat(),
            "holdout_end_utc": datetime.fromtimestamp(int(t_hold[-1]) / 1000, timezone.utc).isoformat(),
        },
        "eligibility": eligible,
        "production_changed": False,
        "research_only": True,
    }


def main():
    rows, source = fetch_history(TARGET_ROWS)
    if len(rows) < MIN_ROWS:
        raise RuntimeError(f"history_too_short:{len(rows)}")
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "history_rows": len(rows),
        "features": 15,
        "research_only": True,
        "production_changed": False,
        "horizons": {},
    }
    for horizon in HORIZONS:
        report["horizons"][horizon] = evaluate(HORIZONS[horizon], rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
