"""Research-only targeted uncertainty routing for BTC direction.

Uses a prequential reliability router, but activates it only on observations
whose uncertainty exceeds a threshold derived from a previous block. Low-risk
observations keep the frozen soft-equal baseline unchanged. Current/future
outcomes never influence the current route.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import joblib

from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_compare import CLASSES, EMBARGO_BARS, PURGE_BARS, load_archive_research_rows, metrics as core_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/historical_research/targeted_uncertainty_routing_oos.json"

HORIZONS = ("5m", "10m")
EXPERTS = ("production", "logreg", "extra_trees", "hgb")
MIN_TRAIN = 3000
META_BLOCK = 500
TEST_BLOCK = 500
MIN_META_ROWS = 250
MAX_ROWS = 12000
FINAL_HOLDOUT_FRAC = 0.20
UNCERTAINTY_QUANTILE = 0.75
EPS = 1e-7


def metrics(y, probs):
    raw = core_metrics(y, probs)
    return {**raw, "ece": float(raw["calibration_error"])}


def _align(model: Any, rows: list[dict[str, Any]]) -> np.ndarray:
    X = np.asarray([r["x"] for r in rows], dtype=float)
    raw = np.asarray(model.predict_proba(X), dtype=float)
    out = np.full((len(rows), 3), EPS, dtype=float)
    idx = {c: i for i, c in enumerate(CLASSES)}
    for j, cls in enumerate(model.classes_):
        if str(cls) in idx:
            out[:, idx[str(cls)]] = raw[:, j]
    out = np.clip(out, EPS, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def _factories():
    return {
        "logreg": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.3, max_iter=2500)),
        ]),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=140, max_depth=10, min_samples_leaf=15,
            max_features="sqrt", random_state=42, n_jobs=-1
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=180, max_leaf_nodes=15, learning_rate=0.04,
            l2_regularization=1.5, random_state=42
        ),
    }


def _fit(train):
    if len(train) < MIN_TRAIN or len({r["y"] for r in train}) < 3:
        raise ValueError("insufficient_or_single_class_training_rows")
    X = np.asarray([r["x"] for r in train], dtype=float)
    y = np.asarray([r["y"] for r in train], dtype=str)
    models = {}
    for name, factory in _factories().items():
        model = factory()
        model.fit(X, y)
        models[name] = model
    return models


def _champion_probs(rows, horizon: str) -> np.ndarray:
    model_path = ROOT / "models" / f"{horizon}.joblib"
    meta_path = ROOT / "models" / f"{horizon}.json"
    if not model_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"production_champion_artifact_missing:{horizon}")
    model = joblib.load(model_path)
    X = np.asarray([r["x"] for r in rows], dtype=float)
    raw = np.asarray(model.predict_proba(X), dtype=float)
    out = np.full((len(rows), 3), EPS, dtype=float)
    index = {c: j for j, c in enumerate(CLASSES)}
    for j, cls in enumerate(getattr(model, "classes_", [])):
        if str(cls) in index:
            out[:, index[str(cls)]] = raw[:, j]
    if out.shape != (len(rows), 3) or not np.isfinite(out).all():
        raise ValueError(f"production_champion_probability_invalid:{horizon}")
    out = np.clip(out, EPS, 1.0)
    out /= out.sum(axis=1, keepdims=True)
    return out


def _probs(models, rows, horizon: str):
    p = {name: _align(model, rows) for name, model in models.items()}
    p["production"] = _champion_probs(rows, horizon)
    return p

