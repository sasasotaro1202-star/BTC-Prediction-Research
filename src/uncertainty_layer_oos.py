"""Research-only uncertainty / upset-risk layer for BTC short-horizon OOS.

This layer consumes already-generated chronological OOS probabilities. It does
not access realized labels until after the current probability vector is fixed,
and it never changes production artifacts.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "data" / "historical_research"
OUT = RESEARCH / "uncertainty_layer_oos.json"
CLASSES = ("DOWN", "FLAT", "UP")
MODELS = ("logreg", "extra", "rf", "hgb")
HORIZONS = ("5m", "10m")
EPS = 1e-8
DEV_FRACTION = 0.80
INNER_FRACTION = 0.75
HOLDOUT_BLOCKS = 8
ALPHA_GRID = (0.05, 0.10, 0.20, 0.30, 0.40)
MIN_ROWS = 2000


def _norm(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0)
    s = p.sum(axis=1, keepdims=True)
    if not np.isfinite(p).all() or np.any(s <= 0):
        raise ValueError("probability_nonfinite")
    return p / s


def _metrics(y: list[str], p: np.ndarray) -> dict[str, float | int]:
    yi = np.asarray([CLASSES.index(v) for v in y], dtype=int)
    p = _norm(p)
    one = np.eye(3)[yi]
    hit = (np.argmax(p, axis=1) == yi).astype(float)
    conf = p.max(axis=1)
    logloss = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), yi], EPS, 1.0))))
    brier = float(np.mean(np.sum((p - one) ** 2, axis=1)))
    ece = 0.0
    for k in range(10):
        lo, hi = k / 10.0, (k + 1) / 10.0
        mask = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(hit[mask].mean()) - float(conf[mask].mean()))
    return {
        "n": int(len(y)),
        "accuracy": float(hit.mean()),
        "logloss": logloss,
        "brier": brier,
        "ece": float(ece),
        "mean_confidence": float(conf.mean()),
    }


def _read(path: Path) -> dict[int, tuple[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    out: dict[int, tuple[str, np.ndarray]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"timestamp", "actual", "p_down", "p_flat", "p_up"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"schema_missing:{path.name}")
        for row in reader:
            ts = int(row["timestamp"])
            y = row["actual"]
            p = np.asarray(
                [float(row["p_down"]), float(row["p_flat"]), float(row["p_up"])],
                dtype=float,
            )
            if y not in CLASSES or not np.isfinite(p).all() or np.any(p < 0) or float(p.sum()) <= 0:
                continue
            if ts in out:
                raise ValueError(f"duplicate_timestamp:{path.name}:{ts}")
            out[ts] = (y, _norm(p[None, :])[0])
    return out


def load_horizon(horizon: str) -> tuple[list[int], list[str], np.ndarray, np.ndarray]:
    paths = {name: RESEARCH / f"oos_{horizon}_{name}.csv" for name in MODELS}
    paths["ensemble"] = RESEARCH / f"oos_{horizon}_ensemble.csv"
    data = {name: _read(path) for name, path in paths.items()}
    common = sorted(set.intersection(*(set(v) for v in data.values())))
    if len(common) < MIN_ROWS:
        raise ValueError(f"insufficient_common_oos_rows:{horizon}:{len(common)}")
    ys: list[str] = []
    probs: dict[str, list[np.ndarray]] = {name: [] for name in data}
    for ts in common:
        labels = {data[name][ts][0] for name in data}
        if len(labels) != 1:
            raise ValueError(f"label_mismatch:{horizon}:{ts}")
        ys.append(next(iter(labels)))
        for name in data:
            probs[name].append(data[name][ts][1])
    stacked = np.asarray([probs[name] for name in MODELS], dtype=float)
    ensemble = np.asarray(probs["ensemble"], dtype=float)
    if not np.all(np.diff(np.asarray(common, dtype=np.int64)) > 0):
        raise ValueError(f"timestamps_not_strictly_increasing:{horizon}")
    return common, ys, stacked, ensemble


def uncertainty_features(model_probs: np.ndarray, ensemble: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate prediction-time uncertainty only from available probabilities."""
    ensemble = _norm(ensemble)
    confidence = ensemble.max(axis=1)
    entropy = -np.sum(ensemble * np.log(np.clip(ensemble, EPS, 1.0)), axis=1) / math.log(3.0)
    sorted_p = np.sort(ensemble, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    # Mean squared dispersion across model probability vectors.
    center = model_probs.mean(axis=0)
    disagreement = np.mean(np.sum((model_probs - center[None, :, :]) ** 2, axis=2), axis=0)
    # Agreement is the fraction of model argmax directions matching the ensemble.
    model_argmax = np.argmax(model_probs, axis=2)
    ensemble_argmax = np.argmax(ensemble, axis=1)
    agreement = np.mean(model_argmax == ensemble_argmax[None, :], axis=0)
    disagreement_scaled = np.clip(disagreement / 0.10, 0.0, 1.0)
    uncertainty = np.clip(
        0.55 * entropy
        + 0.25 * disagreement_scaled
        + 0.20 * (1.0 - margin),
        0.0,
        1.0,
    )
    return {
        "confidence": confidence,
        "entropy": entropy,
        "margin": margin,
        "disagreement": disagreement_scaled,
        "agreement": agreement,
        "uncertainty": uncertainty,
    }


def _shrink(ensemble: np.ndarray, uncertainty: np.ndarray, alpha: float) -> np.ndarray:
    w = np.clip(float(alpha) * uncertainty, 0.0, 0.80)[:, None]
    uniform = np.full_like(ensemble, 1.0 / 3.0)
    return _norm((1.0 - w) * ensemble + w * uniform)


def _block_metrics(y: list[str], p: np.ndarray) -> dict[str, Any]:
    n = len(y)
    if n < HOLDOUT_BLOCKS * 25:
        return {"blocks": 0, "non_worse_logloss_ratio": 0.0, "non_worse_brier_ratio": 0.0}
    edges = np.linspace(0, n, HOLDOUT_BLOCKS + 1, dtype=int)
    ll, br = [], []
    for i in range(HOLDOUT_BLOCKS):
        a, b = int(edges[i]), int(edges[i + 1])
        if b - a < 25:
            continue
        m = _metrics(y[a:b], p[a:b])
        ll.append(float(m["logloss"]))
        br.append(float(m["brier"]))
    return {
        "blocks": len(ll),
        "non_worse_logloss_ratio": float(np.mean(np.asarray(ll) <= np.inf)) if not ll else 0.0,
        "non_worse_brier_ratio": float(np.mean(np.asarray(br) <= np.inf)) if not br else 0.0,
    }


def evaluate(horizon: str) -> dict[str, Any]:
    timestamps, y, model_probs, ensemble = load_horizon(horizon)
    n = len(y)
    split = int(n * DEV_FRACTION)
    development = list(range(split))
    holdout = list(range(split, n))
    if len(holdout) < 400:
        raise ValueError(f"holdout_too_small:{horizon}:{len(holdout)}")

    dev_inner_end = int(len(development) * INNER_FRACTION)
    train_idx = development[:dev_inner_end]
    tune_idx = development[dev_inner_end:]
    feats = uncertainty_features(model_probs, ensemble)
    dev_tune_y = [y[i] for i in tune_idx]

    best_alpha = None
    best_ll = float("inf")
    alpha_scores: list[dict[str, float]] = []
    for alpha in ALPHA_GRID:
        p = _shrink(ensemble[tune_idx], feats["uncertainty"][tune_idx], alpha)
        score = float(_metrics(dev_tune_y, p)["logloss"])
        alpha_scores.append({"alpha": float(alpha), "tune_logloss": score})
        if score < best_ll:
            best_ll, best_alpha = score, alpha

    assert best_alpha is not None
    dev_y = [y[i] for i in development]
    dev_base = _metrics(dev_y, ensemble[development])
    dev_candidate = _metrics(
        dev_y,
        _shrink(ensemble[development], feats["uncertainty"][development], best_alpha),
    )

    hold_y = [y[i] for i in holdout]
    hold_base_p = ensemble[holdout]
    hold_candidate_p = _shrink(hold_base_p, feats["uncertainty"][holdout], best_alpha)
    hold_base = _metrics(hold_y, hold_base_p)
    hold_candidate = _metrics(hold_y, hold_candidate_p)

    # Upset-risk identification threshold is learned only from development.
    risk_threshold = float(np.quantile(feats["uncertainty"][development], 0.75))
    high = feats["uncertainty"][holdout] >= risk_threshold
    high_risk_n = int(high.sum())
    low_risk_n = int((~high).sum())
    hold_hit = (np.argmax(hold_base_p, axis=1) == np.asarray([CLASSES.index(v) for v in hold_y])).astype(float)
    risk_stats = {
        "development_threshold": risk_threshold,
        "holdout_high_uncertainty_n": high_risk_n,
        "holdout_low_uncertainty_n": low_risk_n,
        "high_uncertainty_error_rate": float(1.0 - hold_hit[high].mean()) if high_risk_n else None,
        "low_uncertainty_error_rate": float(1.0 - hold_hit[~high].mean()) if low_risk_n else None,
    }

    # Eligibility is decided ONLY on development data. The final holdout is
    # descriptive and cannot influence promotion/selection.
    dev_ll_rel = (float(dev_base["logloss"]) - float(dev_candidate["logloss"])) / max(
        EPS, abs(float(dev_base["logloss"]))
    )
    dev_br_rel = (float(dev_base["brier"]) - float(dev_candidate["brier"])) / max(
        EPS, abs(float(dev_base["brier"]))
    )
    dev_block = _block_metrics(dev_y, _shrink(ensemble[development], feats["uncertainty"][development], best_alpha))
    dev_block["candidate_vs_baseline_logloss_non_worse_ratio"] = float(
        np.mean(
            [
                _metrics(dev_y[a:b], _shrink(ensemble[development][a:b], feats["uncertainty"][development][a:b], best_alpha))["logloss"]
                <= _metrics(dev_y[a:b], ensemble[development][a:b])["logloss"]
                for a, b in [
                    (int(x), int(z))
                    for x, z in zip(
                        np.linspace(0, len(dev_y), HOLDOUT_BLOCKS + 1, dtype=int)[:-1],
                        np.linspace(0, len(dev_y), HOLDOUT_BLOCKS + 1, dtype=int)[1:],
                    )
                    if int(z) - int(x) >= 25
                ]
            ]
        )
    ) if len(dev_y) >= HOLDOUT_BLOCKS * 25 else 0.0

    high_dev = feats["uncertainty"][development] >= risk_threshold
    high_dev_n = int(high_dev.sum())
    dev_hit = (
        np.argmax(ensemble[development], axis=1)
        == np.asarray([CLASSES.index(v) for v in dev_y])
    ).astype(float)
    dev_risk_error = float(1.0 - dev_hit[high_dev].mean()) if high_dev_n else None

    # Final holdout numbers remain descriptive only.
    ll_rel = (float(hold_base["logloss"]) - float(hold_candidate["logloss"])) / max(
        EPS, abs(float(hold_base["logloss"]))
    )
    br_rel = (float(hold_base["brier"]) - float(hold_candidate["brier"])) / max(
        EPS, abs(float(hold_base["brier"]))
    )
    eligible = bool(
        dev_ll_rel >= 0.03
        and dev_br_rel >= 0.01
        and dev_block["candidate_vs_baseline_logloss_non_worse_ratio"] >= 0.70
        and float(dev_candidate["ece"]) <= float(dev_base["ece"])
        and float(dev_candidate["accuracy"]) >= float(dev_base["accuracy"]) - 0.005
        and high_dev_n >= 100
    )

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "final_holdout_protected": True,
        "final_holdout_used_for_selection": False,
        "promotion_evidence_eligible": False,
        "horizon": horizon,
        "n": n,
        "development_n": len(development),
        "tune_n": len(tune_idx),
        "final_holdout_n": len(holdout),
        "alpha_grid": list(map(float, ALPHA_GRID)),
        "selected_alpha_from_development_only": float(best_alpha),
        "development": {
            "baseline": dev_base,
            "candidate": dev_candidate,
            "tune_scores": alpha_scores,
        },
        "final_holdout": {
            "baseline": hold_base,
            "candidate": hold_candidate,
            "relative_logloss_improvement": float(ll_rel),
            "relative_brier_improvement": float(br_rel),
            "block_stability": block,
            "risk_detection": risk_stats,
        },
        "feature_policy": {
            "inputs": [
                "ensemble_probability",
                "model_entropy",
                "model_disagreement",
                "probability_margin",
            ],
            "outcome_used_after_prediction_fixed": True,
            "no_future_features": True,
        },
        "development_eligibility_metrics": {
            "relative_logloss_improvement": float(dev_ll_rel),
            "relative_brier_improvement": float(dev_br_rel),
            "block_stability": dev_block,
            "high_uncertainty_n": high_dev_n,
            "high_uncertainty_error_rate": dev_risk_error,
        },
        "eligibility": eligible,
        "timestamp_start": int(timestamps[0]),
        "timestamp_end": int(timestamps[-1]),
    }


def main() -> None:
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "horizons": {h: evaluate(h) for h in HORIZONS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
