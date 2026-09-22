"""Research-only chronological ablation for PIT-safe GDELT features.

The evaluator is deliberately conservative:
- only prediction rows whose complete 1-hour GDELT archive coverage is known are used;
- zero-event windows remain valid observations instead of being silently dropped;
- market-only and market+exogenous models use identical chronological OOS rows;
- development OOS uses the project's purge/embargo walk-forward evaluator;
- the latest 20% of covered rows is a descriptive holdout and is never used for
  model selection or fitting;
- this module never changes production prediction state or production models.
"""
from __future__ import annotations

import json
import sys
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

from model_compare import (
    CLASSES,
    HORIZONS,
    load_primary_production_strict_rows,
    walk_forward,
    metrics,
)
from exogenous_information import aggregate_event_features, events_from_records
from gdelt_exogenous_research import coverage_complete, _load_coverage

EVENTS = ROOT / "data" / "exogenous" / "gdelt_events.jsonl"
COVERAGE = ROOT / "data" / "exogenous" / "gdelt_coverage.json"
OUT = ROOT / "data" / "historical_research" / "exogenous_oos_ablation.json"
EXO_KEYS = (
    "news_event_count",
    "news_weighted_sentiment",
    "policy_event_count",
    "macro_event_count",
    "information_importance",
)
LOOKBACK = timedelta(hours=1)
MIN_COVERED = 1500
HOLDOUT_FRAC = 0.20


def _prediction_rows_available():
    from db import DB
    if not DB.exists():
        return False
    try:
        with sqlite3.connect(DB) as con:
            row = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='predictions'").fetchone()
        return row is not None
    except sqlite3.Error:
        return False


def _read_events():
    if not EVENTS.exists():
        return []
    records = []
    with EVENTS.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
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


def _factory():
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.1, max_iter=3000)),
    ])


def _event_features_by_bucket(records):
    events = events_from_records(records)
    buckets = defaultdict(list)
    for event in events:
        ts = event.available_at.astimezone(timezone.utc)
        bucket = ts.replace(minute=(ts.minute // 15) * 15, second=0, microsecond=0)
        buckets[bucket.strftime("%Y%m%d%H%M%S")].append(event)
    return buckets


def _features_for_prediction(row, buckets, coverage):
    prediction_time = datetime.fromisoformat(
        str(row["created"]).replace("Z", "+00:00")
    )
    if not coverage_complete(coverage, prediction_time, LOOKBACK):
        return None
    selected = []
    lower = prediction_time.astimezone(timezone.utc) - LOOKBACK
    upper = prediction_time.astimezone(timezone.utc)
    cur = lower.replace(minute=(lower.minute // 15) * 15, second=0, microsecond=0)
    end = upper.replace(minute=(upper.minute // 15) * 15, second=0, microsecond=0)
    while cur <= end:
        selected.extend(buckets.get(cur.strftime("%Y%m%d%H%M%S"), ()))
        cur += timedelta(minutes=15)
    f = aggregate_event_features(
        [e for e in selected if lower <= e.available_at.astimezone(timezone.utc) <= upper],
        prediction_time,
    )
    return [float(f[k]) for k in EXO_KEYS]


def _block_comparison(base_wf, aug_wf, block_size=25):
    base_y = base_wf["ys"]
    aug_y = aug_wf["ys"]
    if base_y != aug_y or base_wf["ids"] != aug_wf["ids"]:
        raise RuntimeError("market-only and exogenous OOS rows diverged")
    deltas = []
    for start in range(0, len(base_y), block_size):
        y = base_y[start:start + block_size]
        if len(y) < max(10, block_size // 2):
            continue
        bm = metrics(y, base_wf["probs"][start:start + block_size])
        am = metrics(y, aug_wf["probs"][start:start + block_size])
        deltas.append({
            "logloss_delta": am["logloss"] - bm["logloss"],
            "brier_delta": am["brier"] - bm["brier"],
            "accuracy_delta": am["accuracy"] - bm["accuracy"],
        })
    if not deltas:
        return {"blocks": 0}
    return {
        "blocks": len(deltas),
        "improved_logloss_ratio": float(np.mean([d["logloss_delta"] < 0 for d in deltas])),
        "improved_brier_ratio": float(np.mean([d["brier_delta"] < 0 for d in deltas])),
        "non_worse_accuracy_ratio": float(np.mean([d["accuracy_delta"] >= 0 for d in deltas])),
        "mean_logloss_delta": float(np.mean([d["logloss_delta"] for d in deltas])),
        "mean_brier_delta": float(np.mean([d["brier_delta"] for d in deltas])),
        "mean_accuracy_delta": float(np.mean([d["accuracy_delta"] for d in deltas])),
    }


def _load_prediction_rows(horizon):
    """Use only strict-PIT Binance-primary observations for exogenous OOS."""
    return load_primary_production_strict_rows(horizon)


def _evaluate(horizon, records, coverage):
    rows = sorted(_load_prediction_rows(horizon), key=lambda r: str(r["created"]))
    buckets = _event_features_by_bucket(records)
    joined = []
    event_positive = 0
    for row in rows:
        prediction_time = datetime.fromisoformat(
            str(row["created"]).replace("Z", "+00:00")
        )
        if not coverage_complete(coverage, prediction_time, LOOKBACK):
            continue
        exo = _features_for_prediction(row, buckets, coverage)
        if exo is None:
            continue
        joined.append((row, exo))
        if sum(exo[0:1]) > 0 or sum(exo[2:4]) > 0:
            event_positive += 1

    coverage_ratio = len(joined) / max(len(rows), 1)
    if len(joined) < MIN_COVERED:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_complete_archive_coverage",
            "covered_rows": len(joined),
            "event_positive_rows": event_positive,
            "source_rows": len(rows),
            "coverage_ratio": coverage_ratio,
            "required": MIN_COVERED,
        }

    holdout_n = max(1, int(len(joined) * HOLDOUT_FRAC))
    development = joined[:-holdout_n]
    holdout = joined[-holdout_n:]
    if len(development) < 1500 or len(holdout) < 100:
        return {"status": "DEFERRED", "reason": "insufficient_development_or_holdout", "covered_rows": len(joined)}

    base_rows = [{"id": r["id"], "x": r["x"], "y": r["y"]} for r, _ in development]
    aug_rows = [{"id": r["id"], "x": r["x"] + exo, "y": r["y"]} for r, exo in development]
    base_wf = walk_forward(base_rows, _factory, horizon)
    aug_wf = walk_forward(aug_rows, _factory, horizon)
    if base_wf is None or aug_wf is None:
        return {"status": "DEFERRED", "reason": "insufficient_chronological_oos", "covered_rows": len(joined)}

    X0 = np.asarray([r["x"] for r, _ in development], dtype=float)
    X1 = np.asarray([r["x"] + exo for r, exo in development], dtype=float)
    y = np.asarray([r["y"] for r, _ in development])
    H0 = np.asarray([r["x"] for r, _ in holdout], dtype=float)
    H1 = np.asarray([r["x"] + exo for r, exo in holdout], dtype=float)
    yh = [r["y"] for r, _ in holdout]
    base = _factory()
    aug = _factory()
    base.fit(X0, y)
    aug.fit(X1, y)

    return {
        "status": "OK",
        "research_only": True,
        "final_holdout_protected": True,
        "frozen_future_holdout": False,
        "holdout_is_descriptive_only": True,
        "covered_rows": len(joined),
        "event_positive_rows": event_positive,
        "source_rows": len(rows),
        "coverage_ratio": coverage_ratio,
        "development_rows": len(development),
        "holdout_rows": len(holdout),
        "development_walk_forward": {
            "market_only": base_wf["metrics"],
            "market_plus_exogenous": aug_wf["metrics"],
            "comparison": _block_comparison(base_wf, aug_wf),
        },
        "final_holdout_descriptive": {
            "market_only": _metric(yh, base.predict_proba(H0)),
            "market_plus_exogenous": _metric(yh, aug.predict_proba(H1)),
        },
    }


def main():
    records = _read_events()
    coverage = _load_coverage(COVERAGE)
    db_ready = _prediction_rows_available()
    result = {
        "schema_version": 2,
        "research_only": True,
        "policy": "diagnostic_only_no_model_input_no_promotion_effect",
        "final_holdout_protected": True,
        "frozen_future_holdout": False,
        "production_changed": False,
        "coverage_contract": {
            "lookback": "1h",
            "zero_event_windows_included": True,
            "requires_complete_15m_archive_slices": True,
        },
        "event_records": len(records),
        "coverage_slices": len(coverage),
        "database_ready": db_ready,
        "prediction_input_policy": "strict_primary_binance_pit_only",
        "horizons": ({h: _evaluate(h, records, coverage) for h in HORIZONS} if db_ready else {h: {"status": "DEFERRED", "reason": "production_prediction_database_unavailable"} for h in HORIZONS}),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
