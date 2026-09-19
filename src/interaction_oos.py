"""BTC interaction-feature walk-forward research gate.

Research-only: production artifacts are never modified here. Candidate selection
uses chronological development OOS only. The final holdout is evaluated once
with frozen choices and is never used for promotion decisions.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bootstrap_train import (
    CLASSES,
    MIN_OOS,
    MIN_TRAIN,
    TARGET_ROWS,
    fetch_history,
    make_features,
)
from interaction_features import add_interactions
from selective_prediction import choose_threshold, evaluate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "interaction_oos_report.json"

BASE_FEATURES = [
    "ret_1m","ret_3m","ret_5m","ret_10m","acceleration",
    "volatility_5m","volatility_10m","range_position_10m",
    "body_1m","upper_wick_1m","lower_wick_1m","volume_ratio",
    "volume_trend","ema_gap_5m","ema_gap_10m",
]
IX_FEATURES = [
    "ix_ret1_vol10","ix_ret5_vol10","ix_trend_vol","ix_ema_agreement",
    "ix_momentum_agreement","ix_momentum_extension","ix_ret1_volume",
    "ix_ret5_volume","ix_volume_vol","ix_volume_trend_vol",
]
TEST_BLOCK = 250
FINAL_HOLDOUT_FRAC = 0.20
PURGE = {"5m": 5, "10m": 10}


def build_dataset_dicts(rows, horizon):
    h = int(horizon[:-1])
    fs, ys = [], []
    for i in range(30, len(rows) - h):
        history = rows[:i + 1]
        d = dict(zip(BASE_FEATURES, make_features(history)))
        close = np.asarray([r[4] for r in history], float)
        d["ret_15m"] = close[-1] / close[-16] - 1.0
        d["ret_30m"] = close[-1] / close[-31] - 1.0
        future_return = rows[i + h][4] / rows[i][4] - 1.0
        y = "UP" if future_return > 0.00020 else "DOWN" if future_return < -0.00020 else "FLAT"
        fs.append(d)
        ys.append(y)
    return fs, np.asarray(ys)


def metrics(y, p):
    yi = np.array([CLASSES.index(v) for v in y])
    p = np.clip(np.asarray(p, float), 1e-7, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    pred = p.argmax(1)
    one = np.eye(3)[yi]
    conf = p.max(1)
    return {
        "n": int(len(y)),
        "accuracy": float((pred == yi).mean()),
        "logloss": float(log_loss(yi, p, labels=[0, 1, 2])),
        "brier": float(np.mean(np.sum((p - one) ** 2, axis=1))),
        "mean_confidence": float(conf.mean()),
    }


def aligned(model, X):
    raw = model.predict_proba(X)
    out = np.full((len(X), 3), 1e-7)
    for j, c in enumerate(model.classes_):
        out[:, CLASSES.index(str(c))] = raw[:, j]
    return out / out.sum(axis=1, keepdims=True)


def factories():
    return {
        "logreg": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.3, max_iter=2000)),
        ]),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=15, learning_rate=0.04,
            l2_regularization=1.5, random_state=42,
        ),
    }


def walk_forward(X, y, horizon, start, stop):
    base_preds, ix_preds, ys, block_deltas = [], [], [], []
    for end in range(start, stop, TEST_BLOCK):
        train_end = max(0, end - PURGE[horizon])
        test_end = min(end + TEST_BLOCK, stop)
        if test_end <= end or train_end < MIN_TRAIN:
            continue
        model_preds_base, model_preds_ix = [], []
        for factory in factories().values():
            mb, mi = factory(), factory()
            mb.fit(X["base"][:train_end], y[:train_end])
            mi.fit(X["ix"][:train_end], y[:train_end])
            model_preds_base.append(aligned(mb, X["base"][end:test_end]))
            model_preds_ix.append(aligned(mi, X["ix"][end:test_end]))
        pb = np.mean(model_preds_base, axis=0)
        pi = np.mean(model_preds_ix, axis=0)
        block_y = y[end:test_end]
        base_preds.extend(pb.tolist())
        ix_preds.extend(pi.tolist())
        ys.extend(block_y.tolist())
        bm, xm = metrics(block_y, pb), metrics(block_y, pi)
        block_deltas.append({
            "accuracy": xm["accuracy"] - bm["accuracy"],
            "logloss": xm["logloss"] - bm["logloss"],
            "brier": xm["brier"] - bm["brier"],
        })
    if len(ys) < MIN_OOS:
        return None
    return {
        "baseline": metrics(ys, base_preds),
        "interaction": metrics(ys, ix_preds),
        "blocks": block_deltas,
        "y": np.asarray(ys),
        "base_predictions": np.asarray(base_preds, float),
        "interaction_predictions": np.asarray(ix_preds, float),
    }


def gate(result):
    b, x, blocks = result["baseline"], result["interaction"], result["blocks"]
    improved_blocks = sum(
        d["logloss"] < 0 and d["brier"] < 0 and d["accuracy"] >= -0.002
        for d in blocks
    )
    ratio = improved_blocks / max(1, len(blocks))
    return {
        "eligible_for_review": (
            x["logloss"] < b["logloss"]
            and x["brier"] < b["brier"]
            and x["accuracy"] >= b["accuracy"] - 0.002
            and ratio >= 0.55
        ),
        "improved_block_ratio": ratio,
        "production_changed": False,
        "reason": "development-only gate: lower logloss and brier, near-non-decreasing accuracy, and stability across chronological blocks",
    }


def selective_report(dev, holdout):
    """Choose thresholds on development OOS, then freeze them for holdout."""
    y_dev = np.array([CLASSES.index(v) for v in dev["y"]], dtype=int)
    y_hold = np.array([CLASSES.index(v) for v in holdout["y"]], dtype=int)
    chosen = choose_threshold(y_dev, dev["interaction_predictions"])
    frozen = {}
    for target, item in chosen.items():
        threshold = float(item["threshold"])
        frozen[target] = {
            "threshold": threshold,
            "development": evaluate(
                y_dev, dev["interaction_predictions"], threshold
            ),
            "final_holdout": evaluate(
                y_hold, holdout["interaction_predictions"], threshold
            ),
        }
    return {
        "selection_source": "development_walk_forward_only",
        "final_holdout_used_for_threshold_selection": False,
        "frozen_thresholds": frozen,
    }


def main():
    rows, source = fetch_history(TARGET_ROWS)
    report = {
        "schema_version": 3,
        "research_only": True,
        "production_artifacts_modified": False,
        "source": source,
        "rows": len(rows),
        "base_features": BASE_FEATURES,
        "interaction_features": IX_FEATURES,
        "test_block": TEST_BLOCK,
        "purge_bars": PURGE,
        "final_holdout_fraction": FINAL_HOLDOUT_FRAC,
        "horizons": {},
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }

    for horizon in ("5m", "10m"):
        fs, y = build_dataset_dicts(rows, horizon)
        n = len(y)
        final_start = int(n * (1.0 - FINAL_HOLDOUT_FRAC))
        base = np.asarray([[d[k] for k in BASE_FEATURES] for d in fs], float)
        ix = np.asarray(
            [[add_interactions(d)[k] for k in BASE_FEATURES + IX_FEATURES] for d in fs],
            float,
        )

        dev = walk_forward({"base": base, "ix": ix}, y, horizon, MIN_TRAIN, final_start)
        holdout = walk_forward({"base": base, "ix": ix}, y, horizon, final_start, n)

        if dev is None or holdout is None:
            report["horizons"][horizon] = {"status": "insufficient_oos", "samples": n}
            continue

        # IMPORTANT: only development OOS can determine whether a candidate is
        # eligible for review. The final holdout is descriptive evidence only.
        report["horizons"][horizon] = {
            "status": "evaluated",
            "samples": n,
            "development": {
                "baseline": dev["baseline"],
                "interaction": dev["interaction"],
                "delta": {
                    k: dev["interaction"][k] - dev["baseline"][k]
                    for k in ("accuracy", "logloss", "brier")
                },
                "gate": gate(dev),
            },
            "final_holdout": {
                "baseline": holdout["baseline"],
                "interaction": holdout["interaction"],
                "delta": {
                    k: holdout["interaction"][k] - holdout["baseline"][k]
                    for k in ("accuracy", "logloss", "brier")
                },
                "gate": {
                    "eligible_for_review": None,
                    "production_changed": False,
                    "reason": "protected final holdout is descriptive only; no promotion decision is made from it",
                },
            },
            "selective_prediction": selective_report(dev, holdout),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
