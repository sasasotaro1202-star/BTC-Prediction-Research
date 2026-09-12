"""Strict walk-forward research helpers for BTC short-horizon forecasts."""
from __future__ import annotations
from dataclasses import dataclass
from math import log
from typing import Iterable, Sequence
import numpy as np

@dataclass(frozen=True)
class ForecastMetrics:
    n: int
    accuracy: float
    logloss: float
    brier: float
    ece: float

def clip_probs(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    return p / p.sum(axis=1, keepdims=True)

def multiclass_metrics(y: Sequence[int], probs: np.ndarray, bins: int = 10) -> ForecastMetrics:
    y = np.asarray(y, dtype=int)
    p = clip_probs(np.asarray(probs, dtype=float))
    n = len(y)
    if n == 0:
        return ForecastMetrics(0, float("nan"), float("nan"), float("nan"), float("nan"))
    pred = np.argmax(p, axis=1)
    accuracy = float(np.mean(pred == y))
    logloss = float(-np.mean([log(max(p[i, y[i]], 1e-6)) for i in range(n)]))
    onehot = np.eye(p.shape[1])[y]
    brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
    conf = p.max(axis=1)
    correct = (pred == y).astype(float)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return ForecastMetrics(n, accuracy, logloss, brier, float(ece))

def purge_embargo_splits(times: Sequence[int], train_size: int, test_size: int, embargo: int) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Expanding walk-forward splits with an explicit time embargo."""
    t = np.asarray(times)
    start = train_size
    while start + embargo + test_size <= len(t):
        yield np.arange(0, start), np.arange(start + embargo, start + embargo + test_size)
        start += test_size

def direction_from_prices(base: float, future: float, neutral_bps: float = 2.0) -> int:
    """0=DOWN, 1=FLAT, 2=UP using a fixed ex-ante neutral band."""
    r_bps = (future / base - 1.0) * 10000.0
    if r_bps > neutral_bps:
        return 2
    if r_bps < -neutral_bps:
        return 0
    return 1
