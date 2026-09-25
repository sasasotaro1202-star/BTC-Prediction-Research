"""Research-only apples-to-apples OOS for the production core model.

Compares the frozen production artifact, retrained RF, and SoftVotingEnsemble
using the canonical 15 production features and canonical 2bp target labels on
a strictly future Binance Vision archive cohort after the production artifact's
training timestamp. Static archive publication time is not available, so this
cohort is not promotion-eligible and is explicitly marked non-strict-PIT.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from binance_history import binance_archive_rows
from bootstrap_train import make_features
from ensemble_model import SoftVotingEnsemble
from feature_schema import FEATURES
from label_policy import CLASSES, NEUTRAL_RETURN

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUT = ROOT / "data" / "historical_research" / "production_core_oos.json"

HORIZONS = ("5m", "10m")
TARGET_ROWS = 55_000
TRAIN_ROWS = 30_000
BLOCK = 2_000
PURGE_BARS = {"5m": 5, "10m": 10}
MIN_FUTURE_ROWS = 6_000
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


def _metrics(y: list[str], p: np.ndarray) -> dict[str, float | int]:
    yi = np.asarray([CLASSES.index(v) for v in y], dtype=int)
    p = _norm(p)
    pred = np.argmax(p, axis=1)
    hit = pred == yi
    conf = p.max(axis=1)
    ece = 0.0
    for k in range(10):
        lo, hi = k / 10.0, (k + 1) / 10.0
        mask = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(hit[mask].mean()) - float(conf[mask].mean()))
    return {
        "n": int(len(y)),
        "accuracy": float(hit.mean()),
        "logloss": float(log_loss(yi, p, labels=[0, 1, 2])),
        "brier": float(np.mean(np.sum((p - np.eye(3)[yi]) ** 2, axis=1))),
        "ece": float(ece),
        "mean_confidence": float(conf.mean()),
    }


def _future_dataset(rows: list[list[float]], horizon: str, trained_at: datetime) -> tuple[np.ndarray, list[str], list[datetime]]:
    steps = int(horizon[:-1])
    X, y, ts = [], [], []
    for i in range(30, len(rows) - steps):
        created = datetime.fromtimestamp(int(rows[i][0]) / 1000.0, timezone.utc)
        if created <= trained_at:
            continue
        future_return = float(rows[i + steps][4]) / float(rows[i][4]) - 1.0
        label = "UP" if future_return > NEUTRAL_RETURN else "DOWN" if future_return < -NEUTRAL_RETURN else "FLAT"
        features = make_features(rows[: i + 1])
        if len(features) != len(FEATURES) or not all(math.isfinite(float(v)) for v in features):
            continue
        X.append(features)
        y.append(label)
        ts.append(created)
    if not X:
        return np.empty((0, len(FEATURES))), [], []
    return np.asarray(X, dtype=float), y, ts


def _rf():
    return RandomForestClassifier(
        n_estimators=350,
        max_depth=10,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


def _cohort_metrics(y, probs, timestamps):
    blocks = []
    for start in range(0, len(y), BLOCK):
        end = min(start + BLOCK, len(y))
        if end - start < 250:
            continue
        m = _metrics(y[start:end], probs[start:end])
        m["start"] = timestamps[start].isoformat()
        m["end"] = timestamps[end - 1].isoformat()
        blocks.append(m)
    return blocks


def _stability(blocks: list[dict[str, float | int]]) -> dict[str, float | int | None]:
    if not blocks:
        return {"blocks": 0, "worst_accuracy": None, "accuracy_std": None, "worst_logloss": None}
    acc = np.asarray([float(b["accuracy"]) for b in blocks], dtype=float)
    ll = np.asarray([float(b["logloss"]) for b in blocks], dtype=float)
    return {
        "blocks": int(len(blocks)),
        "worst_accuracy": float(acc.min()),
        "accuracy_std": float(acc.std()),
        "worst_logloss": float(ll.max()),
        "logloss_std": float(ll.std()),
    }


def _compare(candidate: dict[str, dict], champion_name: str = "frozen_production") -> dict:
    champion = candidate[champion_name]
    out = {}
    for name, item in candidate.items():
        base = champion["overall"]
        cur = item["overall"]
        out[name] = {
            "accuracy_delta": float(cur["accuracy"] - base["accuracy"]),
            "logloss_delta": float(cur["logloss"] - base["logloss"]),
            "brier_delta": float(cur["brier"] - base["brier"]),
            "ece_delta": float(cur["ece"] - base["ece"]),
            "improved_logloss_ratio": float(
                np.mean(np.asarray(item["block_metrics"]["logloss"]) < np.asarray(champion["block_metrics"]["logloss"]))
            ) if item["block_metrics"]["logloss"] else 0.0,
            "non_worse_accuracy_ratio": float(
                np.mean(
                    np.asarray(item["block_metrics"]["accuracy"])
                    >= np.asarray(champion["block_metrics"]["accuracy"]) - 0.005
                )
            ) if item["block_metrics"]["accuracy"] else 0.0,
        }
    return out


def evaluate(horizon: str) -> dict:
    meta = json.loads((MODEL_DIR / f"{horizon}.json").read_text(encoding="utf-8"))
    trained_at = _parse_dt(meta["trained_at_utc"])
    rows = binance_archive_rows(TARGET_ROWS)
    X, y, timestamps = _future_dataset(rows, horizon, trained_at)
    if len(y) < MIN_FUTURE_ROWS:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_future_rows_after_model_generation",
            "n": int(len(y)),
            "strict_pit": False,
            "promotion_evidence_eligible": False,
            "research_only": True,
            "production_changed": False,
        }

    # Use a clean post-training future cohort. The first TRAIN_ROWS are used
    # only to train retrainable challengers; their outcomes never influence the
    # held future block. The production artifact remains completely frozen.
    split = min(TRAIN_ROWS, max(3_000, len(y) - MIN_FUTURE_ROWS))
    if len(y) - split < MIN_FUTURE_ROWS:
        split = len(y) - MIN_FUTURE_ROWS
    train_X, train_y = X[:split], np.asarray(y[:split])
    test_X, test_y = X[split:], y[split:]
    test_ts = timestamps[split:]

    champion = joblib.load(MODEL_DIR / f"{horizon}.joblib")
    champion_p = _align(champion, test_X)

    rf = _rf()
    rf.fit(train_X, train_y)
    rf_p = _align(rf, test_X)

    soft = SoftVotingEnsemble(learn_weights=True)
    soft.fit(train_X, train_y)
    soft_p = _align(soft, test_X)

    models = {
        "frozen_production": champion_p,
        "retrained_rf": rf_p,
        "retrained_soft_ensemble": soft_p,
    }
    result = {}
    for name, p in models.items():
        blocks = _cohort_metrics(test_y, p, test_ts)
        result[name] = {
            "overall": _metrics(test_y, p),
            "block_metrics": {
                "accuracy": [float(b["accuracy"]) for b in blocks],
                "logloss": [float(b["logloss"]) for b in blocks],
                "brier": [float(b["brier"]) for b in blocks],
                "ece": [float(b["ece"]) for b in blocks],
            },
            "blocks": blocks,
            "stability": _stability(blocks),
        }

    # Diagnostic high-risk bucket: use only model disagreement between the
    # retrained RF and SoftVoting probabilities. This is descriptive, never a
    # selection input for the future test cohort.
    disagreement = np.mean((rf_p - soft_p) ** 2, axis=1)
    cut = np.quantile(disagreement[: max(1, len(disagreement) // 2)], 0.80)
    high = disagreement >= cut
    if np.any(high):
        result["high_disagreement"] = {
            "threshold": float(cut),
            "n": int(high.sum()),
            "production": _metrics([test_y[i] for i in np.flatnonzero(high)], champion_p[high]),
            "retrained_rf": _metrics([test_y[i] for i in np.flatnonzero(high)], rf_p[high]),
            "retrained_soft_ensemble": _metrics([test_y[i] for i in np.flatnonzero(high)], soft_p[high]),
        }

    comparison = _compare(result)
    eligible = False
    for name in ("retrained_rf", "retrained_soft_ensemble"):
        c = comparison[name]
        # Research gate: meaningful proper-scoring gain + no major accuracy
        # loss + at least 70% non-worse blocks. This still does not authorize
        # production promotion because archive PIT publication time is unknown.
        if (
            c["logloss_delta"] <= -0.03 * max(abs(result["frozen_production"]["overall"]["logloss"]), EPS)
            and c["brier_delta"] <= -0.01 * max(abs(result["frozen_production"]["overall"]["brier"]), EPS)
            and c["non_worse_accuracy_ratio"] >= 0.70
            and result[name]["overall"]["accuracy"] >= result["frozen_production"]["overall"]["accuracy"] - 0.005
        ):
            eligible = True

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "strict_pit": False,
        "promotion_evidence_eligible": False,
        "archive_publication_time_unknown": True,
        "horizon": horizon,
        "production_generation": meta.get("model_version"),
        "trained_at_utc": meta.get("trained_at_utc"),
        "train_n": int(split),
        "future_test_n": int(len(test_y)),
        "future_test_start": test_ts[0].isoformat(),
        "future_test_end": test_ts[-1].isoformat(),
        "features": list(FEATURES),
        "models": result,
        "comparison_to_frozen_production": comparison,
        "research_gate_satisfied": bool(eligible),
    }


def main():
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "strict_pit": False,
        "promotion_evidence_eligible": False,
        "horizons": {h: evaluate(h) for h in HORIZONS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
