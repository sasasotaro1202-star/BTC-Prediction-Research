from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bootstrap_train import CLASSES, FEATURES, build_dataset, fetch_history

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "historical_oos_report.json"
MIN_TRAIN = 7_000
TEST_BLOCK = 2_500
PURGE = 70


def normalize(probs):
    p = np.clip(np.asarray(probs, float), 1e-8, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def score(y, probs):
    idx = {c: i for i, c in enumerate(CLASSES)}
    yi = np.asarray([idx[v] for v in y])
    p = normalize(probs)
    one = np.eye(3)[yi]
    return {
        "accuracy": float(np.mean(p.argmax(1) == yi)),
        "logloss": float(log_loss(yi, p, labels=[0, 1, 2])),
        "brier": float(np.mean(np.sum((p - one) ** 2, axis=1))),
    }


def factories():
    return {
        "logreg_c1": lambda: Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=1.0, max_iter=3000))]),
        "rf": lambda: RandomForestClassifier(n_estimators=250, max_depth=7, min_samples_leaf=12, max_features="sqrt", random_state=42, n_jobs=-1),
        "hgb": lambda: HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=15, learning_rate=0.04, l2_regularization=1.5, random_state=42),
    }


def walk_forward(X, y, factory):
    folds = []
    pooled_y, pooled_p = [], []
    for end in range(MIN_TRAIN, len(y), TEST_BLOCK):
        train_end = end - PURGE
        test_end = min(end + TEST_BLOCK, len(y))
        if train_end < MIN_TRAIN or test_end <= end:
            break
        model = factory()
        model.fit(X[:train_end], y[:train_end])
        p = normalize(model.predict_proba(X[end:test_end]))
        yy = y[end:test_end]
        folds.append(score(yy, p))
        pooled_y.extend(yy.tolist())
        pooled_p.extend(p.tolist())
    if not pooled_y:
        return None
    return {
        "pooled": score(pooled_y, pooled_p),
        "folds": folds,
        "fold_count": len(folds),
        "worst_accuracy": min(f["accuracy"] for f in folds),
        "accuracy_std": float(np.std([f["accuracy"] for f in folds])),
        "positive_accuracy_fold_share": float(np.mean([f["accuracy"] > 1 / 3 for f in folds])),
    }


def baseline_walk_forward(y):
    folds = []
    pooled_y, pooled_p = [], []
    for end in range(MIN_TRAIN, len(y), TEST_BLOCK):
        train_end = end - PURGE
        test_end = min(end + TEST_BLOCK, len(y))
        if train_end < MIN_TRAIN or test_end <= end:
            break
        pri = np.asarray([np.mean(y[:train_end] == c) for c in CLASSES])
        p = np.tile(pri, (test_end - end, 1))
        yy = y[end:test_end]
        folds.append(score(yy, p))
        pooled_y.extend(yy.tolist())
        pooled_p.extend(p.tolist())
    if not pooled_y:
        return None
    return {"pooled": score(pooled_y, pooled_p), "folds": folds, "fold_count": len(folds)}


def main():
    rows, source = fetch_history(25_000)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "raw_rows": len(rows),
        "feature_count": len(FEATURES),
        "classes": CLASSES,
        "purge_rows": PURGE,
        "test_block_rows": TEST_BLOCK,
        "min_train_rows": MIN_TRAIN,
        "production_untouched": True,
        "horizons": {},
    }
    for horizon in (5, 10):
        X, y = build_dataset(rows, horizon)
        name = f"{horizon}m"
        result = {"rows": len(y), "class_counts": {c: int(np.sum(y == c)) for c in CLASSES}}
        if len(y) < MIN_TRAIN + TEST_BLOCK:
            result["status"] = "insufficient_history"
            report["horizons"][name] = result
            continue
        baseline = baseline_walk_forward(y)
        candidates = {}
        for model_name, factory in factories().items():
            try:
                candidates[model_name] = walk_forward(X, y, factory)
            except Exception as exc:
                candidates[model_name] = {"status": "error", "error": type(exc).__name__}
        eligible = []
        for model_name, candidate in candidates.items():
            if not candidate or candidate.get("status") == "error":
                continue
            p = candidate["pooled"]
            b = baseline["pooled"]
            if p["logloss"] < b["logloss"] - 0.002 and p["brier"] < b["brier"] - 0.001 and candidate["positive_accuracy_fold_share"] >= 0.60:
                eligible.append((model_name, p["logloss"], p["brier"]))
        result.update({"status": "ok", "baseline": baseline, "candidates": candidates, "eligible": sorted(eligible, key=lambda z: (z[1], z[2]))})
        report["horizons"][name] = result
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "source": source, "raw_rows": len(rows), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
