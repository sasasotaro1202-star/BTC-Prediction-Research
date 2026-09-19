"""BTC interaction-feature walk-forward research gate.

This is intentionally research-only. It compares the current 15-feature schema
with a small, predeclared interaction extension under identical chronological
walk-forward splits. It never changes production artifacts.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bootstrap_train import CLASSES, MIN_OOS, MIN_TRAIN, fetch_history, build_dataset, TARGET_ROWS
from interaction_features import add_interactions

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
TEST_BLOCK = 50
PURGE = {"5m": 5, "10m": 10}


def feature_dicts(rows, horizon):
    h = int(horizon[:-1])
    X, y = build_dataset(rows, h)
    # Rebuild dictionaries through the same feature function used by the
    # bootstrap trainer so interaction candidates use exactly the same inputs.
    from bootstrap_train import make_features
    dicts = [dict(zip(BASE_FEATURES, make_features(rows[i + 30:i + 31]))) for i in range(len(y))]
    return dicts, y


def metrics(y, p):
    yi = np.array([CLASSES.index(v) for v in y])
    p = np.clip(np.asarray(p, float), 1e-7, 1)
    p /= p.sum(axis=1, keepdims=True)
    pred = p.argmax(1)
    one = np.eye(3)[yi]
    return {
        "n": int(len(y)),
        "accuracy": float((pred == yi).mean()),
        "logloss": float(log_loss(yi, p, labels=[0, 1, 2])),
        "brier": float(np.mean(np.sum((p - one) ** 2, axis=1))),
    }


def aligned(model, X):
    raw = model.predict_proba(X)
    out = np.full((len(X), 3), 1e-7)
    for j, c in enumerate(model.classes_):
        out[:, CLASSES.index(str(c))] = raw[:, j]
    out /= out.sum(axis=1, keepdims=True)
    return out


def factories():
    return {
        "logreg": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.3, max_iter=2000)),
        ]),
        "extra": lambda: ExtraTreesClassifier(
            n_estimators=240, max_depth=12, min_samples_leaf=12,
            max_features="sqrt", random_state=42, n_jobs=-1,
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=180, max_leaf_nodes=15, learning_rate=0.04,
            l2_regularization=1.5, random_state=42,
        ),
    }


def walk_forward(X, y, horizon):
    preds_base, preds_ix, ys = [], [], []
    for end in range(MIN_TRAIN, len(y), TEST_BLOCK):
        train_end = max(0, end - PURGE[horizon])
        test_end = min(end + TEST_BLOCK, len(y))
        if test_end <= end or train_end < MIN_TRAIN:
            continue
        for name, factory in factories().items():
            mb, mi = factory(), factory()
            mb.fit(X["base"][:train_end], y[:train_end])
            mi.fit(X["ix"][:train_end], y[:train_end])
            pb = aligned(mb, X["base"][end:test_end])
            pi = aligned(mi, X["ix"][end:test_end])
            preds_base.extend(pb.tolist())
            preds_ix.extend(pi.tolist())
            ys.extend(y[end:test_end])
            break  # one robust, regularized baseline architecture per block
    if len(ys) < MIN_OOS:
        return None
    return metrics(ys, preds_base), metrics(ys, preds_ix), len(ys)


def main():
    rows, source = fetch_history(TARGET_ROWS)
    report = {
        "schema_version": 1,
        "research_only": True,
        "source": source,
        "rows": len(rows),
        "base_features": BASE_FEATURES,
        "interaction_features": IX_FEATURES,
        "horizons": {},
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    for horizon in ("5m", "10m"):
        h = int(horizon[:-1])
        # Build feature vectors directly from each historical prediction point.
        from bootstrap_train import make_features
        dicts = []
        labels = []
        for i in range(30, len(rows) - h):
            d = dict(zip(BASE_FEATURES, make_features(rows[:i + 1])))
            dicts.append(d)
            future_return = rows[i + h][4] / rows[i][4] - 1
            labels.append("UP" if future_return > 0.00020 else "DOWN" if future_return < -0.00020 else "FLAT")
        base = np.asarray([[d[k] for k in BASE_FEATURES] for d in dicts], float)
        ix = np.asarray([[add_interactions(d)[k] for k in BASE_FEATURES + IX_FEATURES] for d in dicts], float)
        result = walk_forward({"base": base, "ix": ix}, np.asarray(labels), horizon)
        if result is None:
            report["horizons"][horizon] = {"status": "insufficient_oos"}
            continue
        b, x, n = result
        report["horizons"][horizon] = {
            "status": "evaluated",
            "n": n,
            "baseline": b,
            "interaction": x,
            "delta": {
                "accuracy": x["accuracy"] - b["accuracy"],
                "logloss": x["logloss"] - b["logloss"],
                "brier": x["brier"] - b["brier"],
            },
            "promotion": {
                "eligible_for_review": (
                    x["logloss"] < b["logloss"]
                    and x["brier"] < b["brier"]
                ),
                "production_changed": False,
            },
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
