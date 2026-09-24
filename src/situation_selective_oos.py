"""Research-only situation-conditioned selective prediction gate for BTC.

The gate score uses only information available in the prediction row:
probability confidence/margin plus causal market-state features. Thresholds are
chosen on chronological development OOS and frozen before the protected
holdout is evaluated. This module never changes production artifacts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from model_compare import HORIZONS, load_primary_production_strict_rows, load_archive_research_rows, aligned

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "situation_selective_oos.json"
TARGET_COVERAGES = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
FINAL_HOLDOUT_FRAC = 0.20
CLASSES = ("DOWN", "FLAT", "UP")
# Runtime feature indices are pinned to feature_schema.FEATURES order.
IX_VOL5 = 5
IX_VOL10 = 6
IX_TREND5 = 2
IX_EMA5 = 13
IX_EMA10 = 14


def _entropy(p: np.ndarray) -> float:
    q = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0)
    q /= q.sum()
    return float(-np.sum(q * np.log(q)) / math.log(3.0))


def _margin(p: np.ndarray) -> float:
    q = np.sort(np.asarray(p, dtype=float))
    return float(max(0.0, q[-1] - q[-2]))


def _state_score(row: dict) -> float:
    p = np.asarray(row["production"], dtype=float)
    p = np.clip(p, 1e-9, 1.0)
    p /= p.sum()
    x = np.asarray(row["x"], dtype=float)
    vol = max(abs(float(x[IX_VOL10])), 1e-8)
    trend = abs(float(x[IX_TREND5])) / vol
    ema_agreement = 1.0 if float(x[IX_EMA5]) * float(x[IX_EMA10]) >= 0.0 else 0.0
    # Confidence dominates. Trend strength and EMA agreement reward coherent
    # states, while entropy directly penalizes ambiguous distributions.
    confidence = float(p.max())
    margin = _margin(p)
    entropy = _entropy(p)
    return (
        0.55 * confidence
        + 0.30 * margin
        + 0.10 * min(trend / 3.0, 1.0)
        + 0.05 * ema_agreement
        - 0.20 * entropy
    )


def _score_components(row: dict) -> dict:
    p = np.asarray(row["production"], dtype=float)
    p = np.clip(p, 1e-9, 1.0)
    p /= p.sum()
    x = np.asarray(row["x"], dtype=float)
    return {
        "confidence": float(p.max()),
        "margin": _margin(p),
        "entropy": _entropy(p),
        "trend_strength": abs(float(x[IX_TREND5])) / max(abs(float(x[IX_VOL10])), 1e-8),
        "volatility_ratio": abs(float(x[IX_VOL10])) / max(abs(float(x[IX_VOL5])), 1e-8),
        "ema_agreement": bool(float(x[IX_EMA5]) * float(x[IX_EMA10]) >= 0.0),
    }


def _accuracy(y: Iterable[str], rows: list[dict], threshold: float) -> dict:
    selected = [r for r in rows if _state_score(r) >= threshold]
    if not selected:
        return {"coverage": 0.0, "n": 0, "accuracy": None, "mean_score": None}
    hits = [
        int(np.argmax(np.asarray(r["production"], dtype=float))) == CLASSES.index(str(r["y"]))
        for r in selected
    ]
    return {
        "coverage": float(len(selected) / max(1, len(rows))),
        "n": len(selected),
        "accuracy": float(np.mean(hits)),
        "mean_score": float(np.mean([_state_score(r) for r in selected])),
    }


def _choose_thresholds(dev: list[dict]) -> dict[str, dict]:
    if len(dev) < 200:
        return {}
    scores = np.asarray([_state_score(r) for r in dev], dtype=float)
    candidates = np.unique(np.round(scores, 8))
    chosen = {}
    for target in TARGET_COVERAGES:
        scored = []
        for threshold in candidates:
            out = _accuracy([r["y"] for r in dev], dev, float(threshold))
            if out["n"] < max(100, int(len(dev) * 0.03)):
                continue
            key = (
                abs(out["coverage"] - target),
                -(out["accuracy"] if out["accuracy"] is not None else -1.0),
                -float(threshold),
            )
            scored.append((key, float(threshold), out))
        if scored:
            _, threshold, metrics = min(scored, key=lambda z: z[0])
            chosen[str(target)] = {"threshold": threshold, "development": metrics}
    return chosen


def _run(horizon: str) -> dict:
    rows = load_primary_production_strict_rows(horizon)
    source = "live_binance_primary"
    promotion_evidence_eligible = True
    if len(rows) < 1500:
        # Descriptive-only fallback: freeze the current Champion on closed archive
        # candles. These rows accelerate research but can never become promotion evidence.
        try:
            import joblib
            meta = json.loads((ROOT / "models" / f"{horizon}.json").read_text(encoding="utf-8"))
            model = joblib.load(ROOT / "models" / f"{horizon}.joblib")
            trained_raw = meta.get("trained_at_utc")
            trained_at = None
            if trained_raw:
                from datetime import datetime, timezone
                trained_at = datetime.fromisoformat(str(trained_raw).replace("Z", "+00:00"))
            archive = load_archive_research_rows(horizon, max(12000, 1500))
            frozen = []
            for row in archive:
                try:
                    created = datetime.fromisoformat(str(row["created"]).replace("Z", "+00:00"))
                    if trained_at is not None and created <= trained_at:
                        continue
                    x = np.asarray([row["x"]], dtype=float)
                    if x.shape != (1, 15) or not np.isfinite(x).all():
                        continue
                    p = aligned(model, x)[0].tolist()
                    frozen.append({**row, "production": p})
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue
            if len(frozen) > len(rows):
                rows = frozen
                source = "binance_archive_frozen_champion"
                promotion_evidence_eligible = False
        except (OSError, ValueError, TypeError, ImportError, json.JSONDecodeError):
            pass
    if len(rows) < 1500:
        return {
            "status": "DEFERRED",
            "n": len(rows),
            "reason": "insufficient_chronological_rows",
        }
    split = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    dev, holdout = rows[:split], rows[split:]
    choices = _choose_thresholds(dev)
    if not choices:
        return {
            "status": "DEFERRED",
            "n": len(rows),
            "reason": "insufficient_development_rows",
        }

    result = {}
    y_all = [r["y"] for r in rows]
    base = _accuracy(y_all, rows, -1e9)
    for target, item in choices.items():
        threshold = float(item["threshold"])
        dev_metrics = item["development"]
        hold_metrics = _accuracy([r["y"] for r in holdout], holdout, threshold)
        result[target] = {
            "threshold": threshold,
            "development": dev_metrics,
            "final_holdout": hold_metrics,
            "holdout_selection_frozen": True,
            "delta_accuracy_vs_all_holdout": (
                hold_metrics["accuracy"] - base["accuracy"]
                if hold_metrics["accuracy"] is not None and base["accuracy"] is not None
                else None
            ),
        }

    # Descriptive failure-mode table for the complete protected holdout.
    buckets = []
    for name, lo, hi in (
        ("LOW_SCORE", -1e9, 0.35),
        ("MID_SCORE", 0.35, 0.60),
        ("HIGH_SCORE", 0.60, 1e9),
    ):
        subset = [r for r in holdout if lo <= _state_score(r) < hi]
        if subset:
            hits = [
                int(np.argmax(np.asarray(r["production"], dtype=float))) == CLASSES.index(str(r["y"]))
                for r in subset
            ]
            buckets.append({
                "bucket": name,
                "n": len(subset),
                "accuracy": float(np.mean(hits)),
                "mean_score": float(np.mean([_state_score(r) for r in subset])),
            })

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "final_holdout_protected": True,
        "final_holdout_used_for_selection": False,
        "selection_source": "development_chronological_oos_only",
        "data_source": source,
        "promotion_evidence_eligible": promotion_evidence_eligible,
        "score_definition": "confidence+margin+trend_strength+ema_agreement-entropy",
        "n": len(rows),
        "development_n": len(dev),
        "final_holdout_n": len(holdout),
        "baseline_all_holdout": base,
        "thresholds": result,
        "holdout_state_buckets": buckets,
    }


def main() -> None:
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "horizons": {h: _run(h) for h in HORIZONS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
