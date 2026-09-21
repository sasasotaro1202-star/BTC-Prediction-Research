"""Research-only chronological ablation for PIT-safe GDELT features.

This never modifies production models. It compares a fixed market-only feature
set against the same estimator plus exogenous features on rows for which the
research event archive actually covers the prediction timestamp. If coverage
is insufficient, the evaluator writes DEFERRED instead of fabricating joins.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

from model_compare import CLASSES, HORIZONS, load_rows
from exogenous_feature_join import aggregate_records_for_prediction


EVENTS = ROOT / "data" / "exogenous" / "gdelt_events.jsonl"
OUT = ROOT / "data" / "historical_research" / "exogenous_oos_ablation.json"
EXO_KEYS = (
    "news_event_count",
    "news_weighted_sentiment",
    "policy_event_count",
    "macro_event_count",
    "information_importance",
)


def _read_events():
    if not EVENTS.exists():
        return []
    records = []
    with EVENTS.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def _metric(y, p):
    idx = {c: i for i, c in enumerate(CLASSES)}
    yi = np.asarray([idx[v] for v in y])
    p = np.asarray(p, dtype=float)
    one = np.eye(3)[yi]
    return {
        "accuracy": float(np.mean(np.argmax(p, axis=1) == yi)),
        "logloss": float(log_loss(yi, p, labels=[0, 1, 2])),
        "brier": float(np.mean(np.sum((p - one) ** 2, axis=1))),
        "n": int(len(y)),
    }


def _evaluate(horizon, records):
    rows = load_rows(horizon)
    joined = []
    for row in rows:
        prediction_time = datetime.fromisoformat(str(row["created"]).replace("Z", "+00:00"))
        features = aggregate_records_for_prediction(records, prediction_time)
        # Require at least one event-window observation. Zero-event rows are
        # valid in production research, but cannot establish archive coverage.
        if features["news_event_count"] <= 0:
            continue
        joined.append((row, [float(features[k]) for k in EXO_KEYS]))

    if len(joined) < 500:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_event_covered_rows",
            "covered_rows": len(joined),
            "required": 500,
            "research_only": True,
            "final_holdout_protected": True,
            "production_changed": False,
        }

    split = int(len(joined) * 0.80)
    train = joined[:split]
    test = joined[split:]
    if len(test) < 100:
        return {"status": "DEFERRED", "reason": "insufficient_test_rows", "covered_rows": len(joined)}

    X0 = np.asarray([r["x"] for r, _ in train], dtype=float)
    X1 = np.asarray([r["x"] + exo for r, exo in train], dtype=float)
    y_train = np.asarray([r["y"] for r, _ in train])
    T0 = np.asarray([r["x"] for r, _ in test], dtype=float)
    T1 = np.asarray([r["x"] + exo for r, exo in test], dtype=float)
    y_test = [r["y"] for r, _ in test]

    def factory():
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.1, max_iter=3000)),
        ])

    base = factory()
    aug = factory()
    base.fit(X0, y_train)
    aug.fit(X1, y_train)
    return {
        "status": "OK",
        "research_only": True,
        "final_holdout_protected": True,
        "production_changed": False,
        "covered_rows": len(joined),
        "train_rows": len(train),
        "test_rows": len(test),
        "market_only": _metric(y_test, base.predict_proba(T0)),
        "market_plus_exogenous": _metric(y_test, aug.predict_proba(T1)),
    }


def main():
    records = _read_events()
    result = {
        "schema_version": 1,
        "research_only": True,
        "policy": "diagnostic_only_no_model_input_no_promotion_effect",
        "final_holdout_protected": True,
        "production_changed": False,
        "event_records": len(records),
        "horizons": {h: _evaluate(h, records) for h in HORIZONS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
