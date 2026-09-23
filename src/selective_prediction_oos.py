"""Research-only selective prediction OOS for BTC short-horizon signals.

The selector never uses current/future labels. A model is fitted on the
strictly earlier training block, and the acceptance threshold for the next
block is derived only from confidence scores observed in an earlier block.
Production models/artifacts are never modified.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for p in (ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model_compare import HORIZONS, load_archive_research_rows  # noqa: E402
from ensemble_model import SoftVotingEnsemble  # noqa: E402

OUT = ROOT / "data" / "historical_research" / "selective_prediction_oos.json"
MIN_TRAIN = 2000
TEST_BLOCK = 300
MAX_ROWS = 9000
TARGET_COVERAGE = 0.30
MIN_SELECTED = 30
CLASSES = ("DOWN", "FLAT", "UP")


def _aligned(model, rows):
    X = np.asarray([r["x"] for r in rows], dtype=float)
    raw = np.asarray(model.predict_proba(X), dtype=float)
    out = np.full((len(rows), 3), 1e-7, dtype=float)
    idx = {c: i for i, c in enumerate(CLASSES)}
    for j, cls in enumerate(model.classes_):
        if str(cls) in idx:
            out[:, idx[str(cls)]] = raw[:, j]
    out = np.clip(out, 1e-7, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def _factories():
    return {
        "logreg": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.3, max_iter=3000)),
        ]),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=220,
            max_depth=12,
            min_samples_leaf=12,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=160,
            max_leaf_nodes=15,
            learning_rate=0.04,
            l2_regularization=1.5,
            random_state=42,
        ),
        "soft_ensemble": lambda: SoftVotingEnsemble(learn_weights=False),
    }


def _fit_predict(factory, train, test):
    X = np.asarray([r["x"] for r in train], dtype=float)
    y = np.asarray([r["y"] for r in train])
    if len(set(y.tolist())) < 3:
        return None
    model = factory()
    model.fit(X, y)
    return _aligned(model, test)


def _metrics(y, probs):
    y = np.asarray(y, dtype=str)
    p = np.asarray(probs, dtype=float)
    yi = np.asarray([CLASSES.index(v) for v in y], dtype=int)
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    hit = pred == yi
    return {
        "n": int(len(y)),
        "accuracy": float(hit.mean()) if len(y) else None,
        "mean_confidence": float(conf.mean()) if len(conf) else None,
    }


def _adaptive_threshold(previous_confidence, target_coverage):
    prev = np.asarray(previous_confidence, dtype=float)
    prev = prev[np.isfinite(prev)]
    if len(prev) < 20:
        return 0.60
    q = float(1.0 - target_coverage)
    # Higher threshold = fewer accepted predictions; derived solely from
    # earlier score distribution, never from current labels.
    return float(np.quantile(prev, q))


def _evaluate_horizon(horizon):
    rows = load_archive_research_rows(horizon, MAX_ROWS)
    if len(rows) < MIN_TRAIN + 2 * TEST_BLOCK:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_archive_rows",
            "n": len(rows),
        }

    blocks = []
    for end in range(MIN_TRAIN, len(rows), TEST_BLOCK):
        train = rows[:end]
        test = rows[end:min(end + TEST_BLOCK, len(rows))]
        if len(test) < TEST_BLOCK // 2:
            continue

        block_models = {}
        for name, factory in _factories().items():
            probs = _fit_predict(factory, train, test)
            if probs is None:
                continue
            block_models[name] = probs

        if not block_models:
            continue

        # The threshold uses only the immediately preceding block's confidence
        # distribution; current-block outcomes are never used for selection.
        previous_conf = {}
        if blocks:
            for name in block_models:
                previous_conf[name] = blocks[-1]["confidence_by_model"].get(name, [])

        by_model = {}
        for name, probs in block_models.items():
            y = [r["y"] for r in test]
            conf = probs.max(axis=1)
            fixed = conf >= 0.60
            fixed_metrics = _metrics(np.asarray(y)[fixed], probs[fixed]) if fixed.any() else {
                "n": 0, "accuracy": None, "mean_confidence": None
            }

            threshold = _adaptive_threshold(previous_conf.get(name, []), TARGET_COVERAGE)
            adaptive = conf >= threshold
            if adaptive.sum() < MIN_SELECTED and len(conf) >= MIN_SELECTED:
                # Keep a minimum diagnostic sample without reading labels: take
                # the top-confidence observations in the current block.
                k = min(MIN_SELECTED, len(conf))
                order = np.argsort(-conf)[:k]
                adaptive = np.zeros(len(conf), dtype=bool)
                adaptive[order] = True

            adaptive_metrics = _metrics(np.asarray(y)[adaptive], probs[adaptive]) if adaptive.any() else {
                "n": 0, "accuracy": None, "mean_confidence": None
            }

            by_model[name] = {
                "all": _metrics(y, probs),
                "fixed_0.60": fixed_metrics,
                "fixed_0.60_coverage": float(fixed.mean()),
                "adaptive": adaptive_metrics,
                "adaptive_threshold": threshold,
                "adaptive_coverage": float(adaptive.mean()),
                "adaptive_selected_index_count": int(adaptive.sum()),
                "confidence_mean": float(conf.mean()),
            }

        blocks.append({
            "start_index": end,
            "n": len(test),
            "models": by_model,
            "confidence_by_model": {
                name: np.asarray(probs.max(axis=1), dtype=float).tolist()
                for name, probs in block_models.items()
            },
        })

    if not blocks:
        return {"status": "DEFERRED", "reason": "no_valid_blocks", "n": len(rows)}

    summary = {}
    for name in _factories():
        selected = [
            b["models"][name]["adaptive"]
            for b in blocks
            if name in b["models"] and b["models"][name]["adaptive"]["n"] >= MIN_SELECTED
        ]
        if not selected:
            continue
        acc = np.asarray([m["accuracy"] for m in selected], dtype=float)
        cov = np.asarray([b["models"][name]["adaptive_coverage"] for b in blocks if name in b["models"]], dtype=float)
        fixed_acc = np.asarray([
            b["models"][name]["fixed_0.60"]["accuracy"]
            for b in blocks
            if name in b["models"] and b["models"][name]["fixed_0.60"]["accuracy"] is not None
        ], dtype=float)
        summary[name] = {
            "blocks": int(len(selected)),
            "mean_adaptive_selected_accuracy": float(acc.mean()),
            "mean_adaptive_coverage": float(cov.mean()),
            "mean_fixed_0.60_selected_accuracy": float(fixed_acc.mean()) if len(fixed_acc) else None,
            "coverage_in_0.20_0.40_ratio": float(np.mean((cov >= 0.20) & (cov <= 0.40))),
        }

    return {
        "status": "OK",
        "research_only": True,
        "production_changed": False,
        "policy": "causal_prequential_confidence_quantile_selection; current_block_labels_never_used_for_thresholds",
        "n": len(rows),
        "target_coverage": TARGET_COVERAGE,
        "summary": summary,
        "blocks": blocks,
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "horizons": {h: _evaluate_horizon(h) for h in HORIZONS},
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
