"""Research-only rolling-retrain RF for BTC 10m direction.

Validation revision: identical algorithm/configuration, no production changes. Purpose: test whether model staleness, rather than model family alone, explains
the recent future-OOS advantage of retrained RF. Hyperparameters are fixed
before evaluation. Each test block uses only a trailing causal training window
ending before the test target window. A final 20% future holdout is descriptive
only and cannot influence any choice.

Binance Vision archive publication timing is not equivalent to live PIT
metadata; therefore this remains research-only and promotion-ineligible.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

from binance_history import binance_archive_rows
from bootstrap_train import make_features
from feature_schema import FEATURES
from label_policy import CLASSES, NEUTRAL_RETURN

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUT = ROOT / "data" / "historical_research" / "rolling_retrain_10m_oos.json"

HORIZON = "10m"
HORIZON_STEPS = 10
TARGET_ROWS = 50_000
TRAIN_WINDOW = 20_000
TEST_BLOCK = 750
GAP_BARS = HORIZON_STEPS
FINAL_HOLDOUT_FRAC = 0.20
MIN_DEVELOPMENT = 10_000
MIN_BLOCKS = 10
MIN_TEST_BLOCK = 500
RF_TREES = 250
EPS = 1e-8


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _norm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    p = np.clip(p, EPS, 1.0)
    s = p.sum(axis=1, keepdims=True)
    if not np.isfinite(p).all() or np.any(s <= 0):
        raise ValueError("invalid_probability_matrix")
    return p / s


def _align(model, X: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(X), dtype=float)
    out = np.full((len(X), 3), EPS, dtype=float)
    for j, cls in enumerate(model.classes_):
        name = str(cls)
        if name in CLASSES:
            out[:, CLASSES.index(name)] = raw[:, j]
    return _norm(out)


def _ece(y: list[str], p: np.ndarray) -> float:
    yi = np.asarray([CLASSES.index(v) for v in y], dtype=int)
    p = _norm(p)
    pred = np.argmax(p, axis=1)
    conf = p.max(axis=1)
    value = 0.0
    for k in range(10):
        lo, hi = k / 10.0, (k + 1) / 10.0
        mask = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if np.any(mask):
            value += float(mask.mean()) * abs(
                float((pred[mask] == yi[mask]).mean()) - float(conf[mask].mean())
            )
    return float(value)


def _metrics(y: list[str], p: np.ndarray) -> dict[str, float | int]:
    yi = np.asarray([CLASSES.index(v) for v in y], dtype=int)
    p = _norm(p)
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(np.argmax(p, axis=1) == yi)),
        "logloss": float(log_loss(yi, p, labels=[0, 1, 2])),
        "brier": float(np.mean(np.sum((p - np.eye(3)[yi]) ** 2, axis=1))),
        "ece": _ece(y, p),
        "mean_confidence": float(p.max(axis=1).mean()),
    }


def _rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=RF_TREES,
        max_depth=10,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


def _build_dataset(rows: list[list[float]], trained_at: datetime) -> tuple[np.ndarray, list[str], list[datetime]]:
    X, y, ts = [], [], []
    for i in range(30, len(rows) - HORIZON_STEPS):
        created = datetime.fromtimestamp(int(rows[i][0]) / 1000.0, timezone.utc)
        if created <= trained_at:
            continue
        try:
            feat = make_features(rows[i - 29 : i + 1])
        except Exception:
            continue
        if len(feat) != len(FEATURES) or not all(math.isfinite(float(v)) for v in feat):
            continue
        future_return = float(rows[i + HORIZON_STEPS][4]) / float(rows[i][4]) - 1.0
        label = (
            "UP" if future_return > NEUTRAL_RETURN
            else "DOWN" if future_return < -NEUTRAL_RETURN
            else "FLAT"
        )
        X.append(feat)
        y.append(label)
        ts.append(created)
    if not X:
        return np.empty((0, len(FEATURES))), [], []
    return np.asarray(X, dtype=float), y, ts


def _block_delta(candidate: dict, baseline: dict) -> dict[str, float]:
    return {
        "accuracy": float(candidate["accuracy"] - baseline["accuracy"]),
        "logloss": float(candidate["logloss"] - baseline["logloss"]),
        "brier": float(candidate["brier"] - baseline["brier"]),
        "ece": float(candidate["ece"] - baseline["ece"]),
    }


def _bootstrap_mean_ci(values: np.ndarray, seed: int = 42, samples: int = 2000) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return {"low": float("nan"), "high": float("nan"), "mean": float(values.mean()) if len(values) else float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[idx].mean(axis=1)
    return {
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "mean": float(values.mean()),
    }


def evaluate() -> dict:
    meta = json.loads((MODEL_DIR / f"{HORIZON}.json").read_text(encoding="utf-8"))
    trained_at = _parse_dt(meta["trained_at_utc"])
    rows = binance_archive_rows(TARGET_ROWS)
    X, y, timestamps = _build_dataset(rows, trained_at)

    if len(y) < MIN_DEVELOPMENT + MIN_TEST_BLOCK * MIN_BLOCKS:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_future_rows",
            "n": int(len(y)),
            "research_only": True,
            "production_changed": False,
            "strict_pit": False,
            "promotion_evidence_eligible": False,
        }

    holdout_start = int(round(len(y) * (1.0 - FINAL_HOLDOUT_FRAC)))
    development_end = holdout_start
    if development_end < MIN_DEVELOPMENT:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_development_rows",
            "n": int(len(y)),
        }

    champion = joblib.load(MODEL_DIR / f"{HORIZON}.joblib")
    blocks: list[dict] = []

    # Fixed preregistered schedule: each rolling model trains on the immediately
    # preceding TRAIN_WINDOW rows, ending GAP_BARS before the test start.
    first_test = max(MIN_DEVELOPMENT, TRAIN_WINDOW + GAP_BARS)
    for test_start in range(first_test, development_end, TEST_BLOCK):
        test_end = min(test_start + TEST_BLOCK, development_end)
        if test_end - test_start < MIN_TEST_BLOCK:
            continue
        train_end = test_start - GAP_BARS
        train_start = max(0, train_end - TRAIN_WINDOW)
        if train_end - train_start < TRAIN_WINDOW:
            continue

        X_train = X[train_start:train_end]
        y_train = np.asarray(y[train_start:train_end])
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        if len(set(y_train)) < 3:
            continue

        challenger = _rf()
        challenger.fit(X_train, y_train)
        cp = _align(challenger, X_test)
        bp = _align(champion, X_test)

        cm = _metrics(y_test, cp)
        bm = _metrics(y_test, bp)
        blocks.append(
            {
                "train_start": timestamps[train_start].isoformat(),
                "train_end": timestamps[train_end - 1].isoformat(),
                "test_start": timestamps[test_start].isoformat(),
                "test_end": timestamps[test_end - 1].isoformat(),
                "train_n": int(len(X_train)),
                "test_n": int(len(X_test)),
                "baseline": bm,
                "candidate": cm,
                "delta": _block_delta(cm, bm),
            }
        )

    if len(blocks) < MIN_BLOCKS:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_oos_blocks",
            "n": int(len(y)),
            "blocks": len(blocks),
            "research_only": True,
            "production_changed": False,
            "strict_pit": False,
            "promotion_evidence_eligible": False,
        }

    def aggregate(name: str) -> dict[str, float | int]:
        keys = ("accuracy", "logloss", "brier", "ece")
        total = sum(int(b["test_n"]) for b in blocks)
        return {
            "n": total,
            **{
                key: float(sum(int(b["test_n"]) * float(b[name][key]) for b in blocks) / total)
                for key in keys
            },
        }

    baseline_dev = aggregate("baseline")
    candidate_dev = aggregate("candidate")
    ll_delta = np.asarray([b["delta"]["logloss"] for b in blocks], dtype=float)
    br_delta = np.asarray([b["delta"]["brier"] for b in blocks], dtype=float)
    acc_delta = np.asarray([b["delta"]["accuracy"] for b in blocks], dtype=float)
    ece_delta = np.asarray([b["delta"]["ece"] for b in blocks], dtype=float)

    # Protected final holdout: use the same fixed training-window protocol,
    # carrying no information backward from the holdout.
    hold_start = development_end
    hold_end = len(y)
    hold_blocks = []
    for test_start in range(hold_start, hold_end, TEST_BLOCK):
        test_end = min(test_start + TEST_BLOCK, hold_end)
        if test_end - test_start < 250:
            continue
        train_end = test_start - GAP_BARS
        train_start = max(0, train_end - TRAIN_WINDOW)
        if train_end - train_start < TRAIN_WINDOW:
            continue
        if len(set(y[train_start:train_end])) < 3:
            continue
        model = _rf()
        model.fit(X[train_start:train_end], np.asarray(y[train_start:train_end]))
        cp = _align(model, X[test_start:test_end])
        bp = _align(champion, X[test_start:test_end])
        yy = y[test_start:test_end]
        hold_blocks.append({
            "n": int(test_end - test_start),
            "baseline": _metrics(yy, bp),
            "candidate": _metrics(yy, cp),
        })

    hold_baseline = aggregate_holdout(hold_blocks, "baseline")
    hold_candidate = aggregate_holdout(hold_blocks, "candidate")

    ll_rel_gain = (baseline_dev["logloss"] - candidate_dev["logloss"]) / max(abs(baseline_dev["logloss"]), EPS)
    br_rel_gain = (baseline_dev["brier"] - candidate_dev["brier"]) / max(abs(baseline_dev["brier"]), EPS)
    eligible = bool(
        ll_rel_gain >= 0.03
        and br_rel_gain >= 0.01
        and float(np.mean(ll_delta <= 0.0)) >= 0.70
        and float(np.mean(br_delta <= 0.0)) >= 0.70
        and float(np.mean(acc_delta >= -0.005)) >= 0.70
        and float(np.mean(ece_delta <= 0.0)) >= 0.70
        and hold_candidate["accuracy"] >= hold_baseline["accuracy"] - 0.005
        and hold_candidate["logloss"] <= hold_baseline["logloss"]
        and hold_candidate["brier"] <= hold_baseline["brier"]
    )

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "strict_pit": False,
        "promotion_evidence_eligible": False,
        "archive_publication_time_unknown": True,
        "horizon": HORIZON,
        "model_version_under_test": "rolling_rf_fixed_window",
        "production_model_version": meta.get("model_version"),
        "trained_at_utc": meta.get("trained_at_utc"),
        "features": list(FEATURES),
        "config": {
            "train_window": TRAIN_WINDOW,
            "test_block": TEST_BLOCK,
            "gap_bars": GAP_BARS,
            "rf_trees": RF_TREES,
            "rf_max_depth": 10,
            "rf_min_samples_leaf": 10,
            "random_state": 42,
            "final_holdout_fraction": FINAL_HOLDOUT_FRAC,
        },
        "n": int(len(y)),
        "development_n": int(development_end),
        "final_holdout_n": int(hold_end - hold_start),
        "development": {
            "baseline": baseline_dev,
            "candidate": candidate_dev,
            "delta": _block_delta(candidate_dev, baseline_dev),
            "block_stability": {
                "blocks": len(blocks),
                "improved_logloss_ratio": float(np.mean(ll_delta <= 0.0)),
                "improved_brier_ratio": float(np.mean(br_delta <= 0.0)),
                "non_worse_accuracy_ratio": float(np.mean(acc_delta >= -0.005)),
                "non_worse_ece_ratio": float(np.mean(ece_delta <= 0.0)),
                "logloss_delta_bootstrap_ci": _bootstrap_mean_ci(ll_delta),
                "brier_delta_bootstrap_ci": _bootstrap_mean_ci(br_delta),
            },
            "blocks_detail": blocks,
        },
        "final_holdout": {
            "protected": True,
            "used_for_selection": False,
            "blocks": len(hold_blocks),
            "baseline": hold_baseline,
            "candidate": hold_candidate,
            "delta": _block_delta(hold_candidate, hold_baseline),
        },
        "eligibility": eligible,
    }


def aggregate_holdout(blocks: list[dict], name: str) -> dict[str, float | int]:
    if not blocks:
        raise ValueError("empty_holdout_blocks")
    keys = ("accuracy", "logloss", "brier", "ece")
    total = sum(int(b["n"]) for b in blocks)
    return {
        "n": total,
        **{
            key: float(sum(int(b["n"]) * float(b[name][key]) for b in blocks) / total)
            for key in keys
        },
    }


def main() -> None:
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "strict_pit": False,
        "promotion_evidence_eligible": False,
        "horizons": {HORIZON: evaluate()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
