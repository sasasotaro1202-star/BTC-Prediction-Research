"""Research-only time-context probability correction for BTC short-horizon direction.

The frozen production model remains the baseline. A lightweight chronological
meta-model sees only the frozen production probabilities plus prediction-time
UTC cyclical calendar features. Training labels are strictly prior and purged
by the existing horizon embargo. A protected final holdout is scored once.
Archive publication timing is unknown, therefore promotion is disabled.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from binance_history import binance_archive_rows
from model_compare import CLASSES, EMBARGO_BARS, metrics, normalize, load_archive_research_rows

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUT = ROOT / "data" / "historical_research" / "time_context_correction_oos.json"

HORIZONS = ("5m", "10m")
MAX_ROWS = 12000
TEST_BLOCK = 500
FINAL_HOLDOUT_FRAC = 0.20
MIN_TRAIN = 3000
MIN_BLOCKS = 8
BLEND_WEIGHT = 0.50
EPS = 1e-8


def _dt(value: str | datetime) -> datetime | None:
    try:
        dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc)


def _time_features(rows):
    x = []
    for row in rows:
        dt = _dt(row["created"])
        if dt is None:
            raise ValueError("time_context_invalid_timestamp")
        hour_of_week = dt.weekday() * 24.0 + dt.hour + dt.minute / 60.0
        theta = 2.0 * math.pi * hour_of_week / 168.0
        x.append([
            math.sin(theta),
            math.cos(theta),
            math.sin(2.0 * theta),
            math.cos(2.0 * theta),
        ])
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4 or not np.isfinite(arr).all():
        raise ValueError("time_context_features_nonfinite")
    return arr


def _baseline_probs(rows, model):
    X = np.asarray([row["x"] for row in rows], dtype=float)
    raw = np.asarray(model.predict_proba(X), dtype=float)
    out = np.full((len(rows), 3), EPS, dtype=float)
    for j, cls in enumerate(model.classes_):
        if str(cls) in CLASSES:
            out[:, CLASSES.index(str(cls))] = raw[:, j]
    return normalize(out)


def _meta_features(rows, baseline):
    p = normalize(baseline)
    logits = np.log(np.clip(p, EPS, 1.0))
    return np.column_stack([logits, _time_features(rows)])


def _fit_meta(train_rows, train_base):
    if len(train_rows) < MIN_TRAIN:
        return None
    y = np.asarray([row["y"] for row in train_rows])
    if len(np.unique(y)) < 3:
        return None
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.25, max_iter=3000)),
    ])
    model.fit(_meta_features(train_rows, train_base), y)
    return model


def _causal_train(rows, test_start, horizon):
    cutoff = _dt(test_start)
    if cutoff is None:
        return []
    embargo_minutes = int(EMBARGO_BARS[horizon])
    target_cutoff = cutoff - timedelta(minutes=embargo_minutes)
    out = []
    for row in rows:
        created = _dt(row["created"])
        target = _dt(row["target"])
        if created is None or target is None:
            continue
        if created < target and target < target_cutoff:
            out.append(row)
    return out


def _evaluate_horizon(horizon):
    meta_path = MODEL_DIR / f"{horizon}.json"
    model_path = MODEL_DIR / f"{horizon}.joblib"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_trained_at = _dt(meta["trained_at_utc"])
    if model_trained_at is None:
        return {"status": "DEFERRED", "reason": "invalid_production_trained_at"}

    rows = [
        row for row in load_archive_research_rows(horizon, MAX_ROWS)
        if (_dt(row["created"]) is not None and _dt(row["created"]) > model_trained_at)
    ]
    if len(rows) < MIN_TRAIN + TEST_BLOCK:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_post_training_archive_rows",
            "n": len(rows),
            "research_only": True,
            "production_changed": False,
            "promotion_evidence_eligible": False,
        }

    model = joblib.load(model_path)
    split = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    development = rows[:split]
    holdout = rows[split:]

    blocks = []
    for end in range(MIN_TRAIN, len(development), TEST_BLOCK):
        test = development[end:min(end + TEST_BLOCK, len(development))]
        if len(test) < TEST_BLOCK:
            continue
        train = _causal_train(development[:end], test[0]["created"], horizon)
        if len(train) < MIN_TRAIN:
            continue
        train_base = _baseline_probs(train, model)
        test_base = _baseline_probs(test, model)
        meta_model = _fit_meta(train, train_base)
        if meta_model is None:
            continue
        raw = meta_model.predict_proba(_meta_features(test, test_base))
        aligned = np.full((len(test), 3), EPS, dtype=float)
        for j, cls in enumerate(meta_model.classes_):
            if str(cls) in CLASSES:
                aligned[:, CLASSES.index(str(cls))] = raw[:, j]
        aligned = normalize(aligned)
        blended = normalize((1.0 - BLEND_WEIGHT) * test_base + BLEND_WEIGHT * aligned)
        y = [row["y"] for row in test]
        b = metrics(y, test_base)
        c = metrics(y, aligned)
        g = metrics(y, blended)
        blocks.append({
            "n": len(test),
            "baseline": b,
            "raw_candidate": c,
            "candidate": g,
            "raw_delta": {k: float(c[k] - b[k]) for k in ("accuracy", "logloss", "brier", "calibration_error")},
            "delta": {k: float(g[k] - b[k]) for k in ("accuracy", "logloss", "brier", "calibration_error")},
        })

    if len(blocks) < MIN_BLOCKS:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_oos_blocks",
            "n": len(rows),
            "blocks": len(blocks),
            "research_only": True,
            "production_changed": False,
            "promotion_evidence_eligible": False,
        }

    def aggregate(side, key):
        return float(sum(block["n"] * block[side][key] for block in blocks) / sum(block["n"] for block in blocks))

    baseline = {k: aggregate("baseline", k) for k in ("accuracy", "logloss", "brier", "calibration_error")}
    candidate = {k: aggregate("candidate", k) for k in ("accuracy", "logloss", "brier", "calibration_error")}
    raw_candidate = {k: aggregate("raw_candidate", k) for k in ("accuracy", "logloss", "brier", "calibration_error")}

    hold_train = _causal_train(development, holdout[0]["created"], horizon)
    if len(hold_train) < MIN_TRAIN:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_purged_holdout_training_rows",
            "n": len(rows),
            "blocks": len(blocks),
            "research_only": True,
            "production_changed": False,
            "promotion_evidence_eligible": False,
        }

    hold_base = _baseline_probs(holdout, model)
    hold_train_base = _baseline_probs(hold_train, model)
    hold_meta = _fit_meta(hold_train, hold_train_base)
    if hold_meta is None:
        return {"status": "DEFERRED", "reason": "holdout_meta_training_unavailable"}

    hp = np.asarray(hold_meta.predict_proba(_meta_features(holdout, hold_base)), dtype=float)
    aligned_hp = np.full((len(holdout), 3), EPS, dtype=float)
    for j, cls in enumerate(hold_meta.classes_):
        if str(cls) in CLASSES:
            aligned_hp[:, CLASSES.index(str(cls))] = hp[:, j]
    aligned_hp = normalize(aligned_hp)
    hold_c = normalize((1.0 - BLEND_WEIGHT) * hold_base + BLEND_WEIGHT * aligned_hp)

    hold_bm = metrics([row["y"] for row in holdout], hold_base)
    hold_cm = metrics([row["y"] for row in holdout], hold_c)
    hold_delta = {k: float(hold_cm[k] - hold_bm[k]) for k in ("accuracy", "logloss", "brier", "calibration_error")}

    deltas = np.asarray(\n        [[float(block["delta"][k]) for k in ("accuracy", "logloss", "brier", "calibration_error")] for block in blocks],\n        dtype=float,\n    )
    eligibility = bool(
        (baseline["logloss"] - candidate["logloss"]) / max(abs(baseline["logloss"]), EPS) >= 0.03
        and (baseline["brier"] - candidate["brier"]) / max(abs(baseline["brier"]), EPS) >= 0.01
        and candidate["accuracy"] >= baseline["accuracy"] - 0.005
        and float(np.mean(deltas[:, 1] < 0.0)) >= 0.70
        and float(np.mean(deltas[:, 2] < 0.0)) >= 0.70
        and float(np.mean(deltas[:, 0] >= -0.005)) >= 0.70
        and hold_cm["accuracy"] >= hold_bm["accuracy"] - 0.005
        and hold_cm["logloss"] <= hold_bm["logloss"]
        and hold_cm["brier"] <= hold_bm["brier"]
    )

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "strict_pit": False,
        "promotion_evidence_eligible": False,
        "horizon": horizon,
        "n": len(rows),
        "development_n": len(development),
        "final_holdout_n": len(holdout),
        "config": {"test_block": TEST_BLOCK, "blend_weight": BLEND_WEIGHT, "min_train": MIN_TRAIN},
        "development": {
            "blocks": len(blocks),
            "baseline": baseline,
            "raw_candidate": raw_candidate,
            "candidate": candidate,
            "delta": {k: float(candidate[k] - baseline[k]) for k in baseline},
            "stability": {
                "improved_logloss_ratio": float(np.mean(deltas[:, 1] < 0.0)),
                "improved_brier_ratio": float(np.mean(deltas[:, 2] < 0.0)),
                "non_worse_accuracy_ratio": float(np.mean(deltas[:, 0] >= -0.005)),
            },
        },
        "final_holdout": {
            "protected": True,
            "used_for_selection": False,
            "baseline": hold_bm,
            "candidate": hold_cm,
            "delta": hold_delta,
        },
        "eligibility": eligibility,
    }


def main():
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "horizons": {h: _evaluate_horizon(h) for h in HORIZONS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
