"""Research-only risk-gated selective routing for BTC.

Stage 1 predicts whether the production ensemble is likely to err using only
prediction-time uncertainty/disagreement features learned from prior rows.
Stage 2 is activated only for high-risk rows and chooses one frozen expert
using prior high-risk loss. Current outcomes are incorporated only after the
current prediction is fixed. The final holdout is never used for selection.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "data" / "historical_research"
OUT = RESEARCH / "risk_gated_selective_routing_oos.json"

CLASSES = ("DOWN", "FLAT", "UP")
EXPERTS = ("logreg", "extra", "rf", "hgb", "ensemble")
THRESHOLDS = (0.60, 0.65, 0.70, 0.75, 0.80)
HORIZONS = ("5m", "10m")
META_BLOCK = 900
TEST_BLOCK = 600
GAP_BARS = {"5m": 65, "10m": 70}
FINAL_HOLDOUT_FRAC = 0.20
MIN_ROWS = 5000
MIN_TEST = 100
EPS = 1e-8


def _norm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    p = np.clip(p, EPS, 1.0)
    s = p.sum(axis=1, keepdims=True)
    if not np.isfinite(p).all() or np.any(s <= 0):
        raise ValueError("invalid_probability_matrix")
    return p / s


def _metrics(y: list[str], p: np.ndarray) -> dict[str, float | int]:
    yi = np.asarray([CLASSES.index(v) for v in y], dtype=int)
    p = _norm(p)
    hit = np.argmax(p, axis=1) == yi
    conf = p.max(axis=1)
    return {
        "n": int(len(y)),
        "accuracy": float(hit.mean()),
        "logloss": float(-np.mean(np.log(np.clip(p[np.arange(len(y)), yi], EPS, 1.0)))),
        "brier": float(np.mean(np.sum((p - np.eye(3)[yi]) ** 2, axis=1))),
        "ece": _ece(yi, p),
        "mean_confidence": float(conf.mean()),
    }


def _ece(yi: np.ndarray, p: np.ndarray) -> float:
    conf = p.max(axis=1)
    pred = np.argmax(p, axis=1)
    value = 0.0
    for k in range(10):
        lo, hi = k / 10.0, (k + 1) / 10.0
        mask = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if np.any(mask):
            value += float(mask.mean()) * abs(float((pred[mask] == yi[mask]).mean()) - float(conf[mask].mean()))
    return float(value)


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
            p = np.asarray([float(row["p_down"]), float(row["p_flat"]), float(row["p_up"])], dtype=float)
            if y not in CLASSES:
                continue
            if not np.isfinite(p).all() or np.any(p < 0) or float(p.sum()) <= 0:
                continue
            if ts in out:
                raise ValueError(f"duplicate_timestamp:{path.name}:{ts}")
            out[ts] = (y, _norm(p)[0])
    return out


def load_horizon(horizon: str) -> tuple[list[int], list[str], dict[str, np.ndarray]]:
    data = {name: _read(RESEARCH / f"oos_{horizon}_{name}.csv") for name in EXPERTS}
    common = sorted(set.intersection(*(set(v) for v in data.values())))
    if len(common) < MIN_ROWS:
        raise ValueError(f"insufficient_common_oos_rows:{horizon}:{len(common)}")
    ys: list[str] = []
    probs: dict[str, list[np.ndarray]] = {name: [] for name in EXPERTS}
    for ts in common:
        labels = {data[name][ts][0] for name in data}
        if len(labels) != 1:
            raise ValueError(f"label_mismatch:{horizon}:{ts}")
        ys.append(next(iter(labels)))
        for name in EXPERTS:
            probs[name].append(data[name][ts][1])
    arrays = {name: np.asarray(value, dtype=float) for name, value in probs.items()}
    if not np.all(np.diff(np.asarray(common, dtype=np.int64)) > 0):
        raise ValueError(f"timestamps_not_strictly_increasing:{horizon}")
    return common, ys, arrays


def uncertainty_features(probs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ensemble = _norm(probs["ensemble"])
    names = ("logreg", "extra", "rf", "hgb")
    stacked = np.stack([_norm(probs[name]) for name in names], axis=0)
    confidence = ensemble.max(axis=1)
    entropy = -np.sum(ensemble * np.log(np.clip(ensemble, EPS, 1.0)), axis=1) / math.log(3.0)
    ordered = np.sort(ensemble, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    center = stacked.mean(axis=0)
    disagreement = np.mean(np.sum((stacked - center[None, :, :]) ** 2, axis=2), axis=0)
    disagreement = np.clip(disagreement / 0.10, 0.0, 1.0)
    argmaxes = np.argmax(stacked, axis=2)
    ensemble_argmax = np.argmax(ensemble, axis=1)
    agreement = np.mean(argmaxes == ensemble_argmax[None, :], axis=0)
    uncertainty = np.clip(
        0.55 * entropy + 0.25 * disagreement + 0.20 * (1.0 - margin),
        0.0,
        1.0,
    )
    return {
        "confidence": confidence,
        "entropy": entropy,
        "margin": margin,
        "disagreement": disagreement,
        "agreement": agreement,
        "uncertainty": uncertainty,
    }


def _risk_matrix(features: dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
    return np.column_stack([
        features["confidence"][idx],
        features["entropy"][idx],
        features["margin"][idx],
        features["disagreement"][idx],
        features["agreement"][idx],
        features["uncertainty"][idx],
    ])


def _fit_risk_model(features: dict[str, np.ndarray], ensemble: np.ndarray, y: list[str], idx: np.ndarray):
    yi = np.asarray([CLASSES.index(v) for v in y], dtype=int)
    error = (np.argmax(ensemble[idx], axis=1) != yi[idx]).astype(int)
    if len(idx) < 300 or len(np.unique(error)) < 2:
        return None
    model = Pipeline([
        ("scale", StandardScaler()),
        ("risk", LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")),
    ])
    model.fit(_risk_matrix(features, idx), error)
    return model


def _risk_predict(model, features: dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
    if model is None:
        return np.clip(features["uncertainty"][idx], 0.0, 1.0)
    out = np.asarray(model.predict_proba(_risk_matrix(features, idx))[:, 1], dtype=float)
    if not np.isfinite(out).all():
        raise ValueError("risk_probability_nonfinite")
    return np.clip(out, 0.0, 1.0)


def _select_expert(
    probs: dict[str, np.ndarray],
    y: list[str],
    features: dict[str, np.ndarray],
    idx: np.ndarray,
) -> str:
    risk_cut = float(np.quantile(features["uncertainty"][idx], 0.75))
    high = idx[features["uncertainty"][idx] >= risk_cut]
    if len(high) < 50:
        high = idx
    scores: dict[str, float] = {}
    yi = np.asarray([CLASSES.index(v) for v in y], dtype=int)
    for name in EXPERTS:
        p = _norm(probs[name][high])
        loss = -np.log(np.clip(p[np.arange(len(high)), yi[high]], EPS, 1.0))
        scores[name] = float(np.mean(loss))
    return min(scores, key=scores.get)


def _apply_gate(
    ensemble: np.ndarray,
    expert_probs: np.ndarray,
    risk_prob: np.ndarray,
    threshold: float,
) -> np.ndarray:
    choose = np.asarray(risk_prob >= float(threshold), dtype=bool)
    out = ensemble.copy()
    out[choose] = expert_probs[choose]
    return _norm(out)


def _safe_gate_score(
    y: list[str],
    baseline: np.ndarray,
    expert: np.ndarray,
    risk_prob: np.ndarray,
    threshold: float,
) -> tuple[float, dict[str, Any]]:
    candidate = _apply_gate(baseline, expert, risk_prob, threshold)
    metrics = _metrics(y, candidate)
    metrics["risk_coverage"] = float(np.mean(risk_prob >= threshold))
    return float(metrics["logloss"]), metrics


def _choose_threshold(
    y: list[str],
    baseline: np.ndarray,
    expert: np.ndarray,
    risk_prob: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    best = None
    scores = []
    for threshold in THRESHOLDS:
        score, metric = _safe_gate_score(y, baseline, expert, risk_prob, threshold)
        item = {"threshold": threshold, "logloss": score, **metric}
        scores.append(item)
        key = (score, -metric["accuracy"], -metric["risk_coverage"])
        if best is None or key < best[0]:
            best = (key, threshold, item)
    if best is None:
        return 0.75, {"threshold": 0.75, "reason": "fallback"}
    return float(best[1]), {"selected": best[2], "scores": scores}


def evaluate(horizon: str) -> dict[str, Any]:
    timestamps, y, probs = load_horizon(horizon)
    features = uncertainty_features(probs)
    n = len(y)
    holdout_start = int(round(n * (1.0 - FINAL_HOLDOUT_FRAC)))
    development_idx = np.arange(holdout_start, dtype=int)
    holdout_idx = np.arange(holdout_start, n, dtype=int)
    if len(development_idx) < MIN_ROWS or len(holdout_idx) < 400:
        return {"status": "DEFERRED", "reason": "insufficient_rows", "n": n}

    blocks = []
    dev_start = META_BLOCK + GAP_BARS[horizon] + TEST_BLOCK
    for test_start in range(dev_start, holdout_start, TEST_BLOCK):
        test_end = min(test_start + TEST_BLOCK, holdout_start)
        if test_end - test_start < MIN_TEST:
            continue
        test_gap_end = test_start - GAP_BARS[horizon]
        tune_start = max(META_BLOCK, test_gap_end - TEST_BLOCK)
        meta_end = tune_start - GAP_BARS[horizon]
        meta_start = max(0, meta_end - META_BLOCK)
        meta_idx = np.arange(meta_start, meta_end, dtype=int)
        tune_idx = np.arange(tune_start, test_gap_end, dtype=int)
        test_idx = np.arange(test_start, test_end, dtype=int)
        if len(meta_idx) < 300 or len(tune_idx) < MIN_TEST:
            continue

        risk_model = _fit_risk_model(features, probs["ensemble"], y, meta_idx)
        risk_tune = _risk_predict(risk_model, features, tune_idx)
        expert_name = _select_expert(probs, y, features, meta_idx)
        baseline_tune = probs["ensemble"][tune_idx]
        expert_tune = probs[expert_name][tune_idx]
        threshold, threshold_info = _choose_threshold(
            [y[i] for i in tune_idx],
            baseline_tune,
            expert_tune,
            risk_tune,
        )

        test_risk = _risk_predict(risk_model, features, test_idx)
        baseline_test = probs["ensemble"][test_idx]
        expert_test = probs[expert_name][test_idx]
        candidate_test = _apply_gate(baseline_test, expert_test, test_risk, threshold)

        y_test = [y[i] for i in test_idx]
        baseline_m = _metrics(y_test, baseline_test)
        candidate_m = _metrics(y_test, candidate_test)
        high = test_risk >= threshold
        low = ~high
        high_m = _metrics([y_test[i] for i in np.flatnonzero(high)], candidate_test[high]) if high.any() else {"n": 0}
        high_b = _metrics([y_test[i] for i in np.flatnonzero(high)], baseline_test[high]) if high.any() else {"n": 0}
        low_m = _metrics([y_test[i] for i in np.flatnonzero(low)], candidate_test[low]) if low.any() else {"n": 0}
        low_b = _metrics([y_test[i] for i in np.flatnonzero(low)], baseline_test[low]) if low.any() else {"n": 0}

        blocks.append({
            "start_timestamp": timestamps[test_start],
            "end_timestamp": timestamps[test_end - 1],
            "n": len(test_idx),
            "expert": expert_name,
            "threshold": threshold,
            "risk_coverage": float(high.mean()),
            "baseline": baseline_m,
            "candidate": candidate_m,
            "delta": {
                "accuracy": candidate_m["accuracy"] - baseline_m["accuracy"],
                "logloss": candidate_m["logloss"] - baseline_m["logloss"],
                "brier": candidate_m["brier"] - baseline_m["brier"],
                "ece": candidate_m["ece"] - baseline_m["ece"],
            },
            "high_risk": {
                "n": int(high.sum()),
                "baseline": high_b,
                "candidate": high_m,
                "delta_logloss": high_m["logloss"] - high_b["logloss"] if high.any() else None,
                "delta_brier": high_m["brier"] - high_b["brier"] if high.any() else None,
                "accuracy_delta": high_m["accuracy"] - high_b["accuracy"] if high.any() else None,
            },
            "low_risk": {
                "n": int(low.sum()),
                "baseline": low_b,
                "candidate": low_m,
                "delta_logloss": low_m["logloss"] - low_b["logloss"] if low.any() else None,
                "delta_brier": low_m["brier"] - low_b["brier"] if low.any() else None,
                "accuracy_delta": low_m["accuracy"] - low_b["accuracy"] if low.any() else None,
            },
        })

    if len(blocks) < 8:
        return {"status": "DEFERRED", "reason": "insufficient_oos_blocks", "n": n, "blocks": len(blocks)}

    def aggregate(key: str) -> dict[str, Any]:
        ys, bp, cp = [], [], []
        for b in blocks:
            start = timestamps.index(b["start_timestamp"])
            end = timestamps.index(b["end_timestamp"]) + 1
            ys.extend(y[start:end])
            bp.extend(probs["ensemble"][start:end].tolist())
            cp.extend(_apply_gate(
                probs["ensemble"][start:end],
                probs[b["expert"]][start:end],
                np.full(end - start, b["threshold"], dtype=float),
                b["threshold"],
            ).tolist())
        return _metrics(ys, np.asarray(cp if key == "candidate" else bp))

    baseline_dev = aggregate("baseline")
    candidate_dev = aggregate("candidate")
    deltas = np.asarray([b["delta"]["logloss"] for b in blocks], dtype=float)
    brier_deltas = np.asarray([b["delta"]["brier"] for b in blocks], dtype=float)
    accuracy_deltas = np.asarray([b["delta"]["accuracy"] for b in blocks], dtype=float)
    eligible = bool(
        np.mean(deltas < 0) >= 0.70
        and np.mean(brier_deltas <= 0) >= 0.70
        and candidate_dev["logloss"] <= baseline_dev["logloss"] - 0.003
        and candidate_dev["brier"] <= baseline_dev["brier"] - 0.0015
        and candidate_dev["accuracy"] >= baseline_dev["accuracy"] - 0.005
    )

    # Final holdout: selection uses development only. The selected threshold is
    # obtained from the last development validation block, then the risk model
    # and expert reliability are refit on all development rows.
    final_train_end = holdout_start - TEST_BLOCK
    final_train_idx = np.arange(0, final_train_end, dtype=int)
    final_tune_idx = np.arange(final_train_end, holdout_start, dtype=int)
    final_risk_model = _fit_risk_model(features, probs["ensemble"], y, final_train_idx)
    final_risk_tune = _risk_predict(final_risk_model, features, final_tune_idx)
    final_expert = _select_expert(probs, y, features, final_train_idx)
    final_threshold, final_threshold_info = _choose_threshold(
        [y[i] for i in final_tune_idx],
        probs["ensemble"][final_tune_idx],
        probs[final_expert][final_tune_idx],
        final_risk_tune,
    )
    full_risk_model = _fit_risk_model(features, probs["ensemble"], y, development_idx)
    hold_risk = _risk_predict(full_risk_model, features, holdout_idx)
    hold_candidate = _apply_gate(
        probs["ensemble"][holdout_idx],
        probs[final_expert][holdout_idx],
        hold_risk,
        final_threshold,
    )
    hold_y = [y[i] for i in holdout_idx]
    hold_baseline = _metrics(hold_y, probs["ensemble"][holdout_idx])
    hold_candidate_m = _metrics(hold_y, hold_candidate)

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "chronological": True,
        "strict_common_oos_csv": True,
        "final_holdout_protected": True,
        "final_holdout_used_for_selection": False,
        "policy": "risk_model_and_expert_selection_use_only_prior_rows; gate_activates_challenger_only_on_high_risk",
        "horizon": horizon,
        "n": n,
        "development_n": len(development_idx),
        "final_holdout_n": len(holdout_idx),
        "blocks": len(blocks),
        "development": {
            "baseline": baseline_dev,
            "candidate": candidate_dev,
            "delta": {
                "accuracy": candidate_dev["accuracy"] - baseline_dev["accuracy"],
                "logloss": candidate_dev["logloss"] - baseline_dev["logloss"],
                "brier": candidate_dev["brier"] - baseline_dev["brier"],
                "ece": candidate_dev["ece"] - baseline_dev["ece"],
            },
            "block_stability": {
                "improved_logloss_ratio": float(np.mean(deltas < 0)),
                "improved_brier_ratio": float(np.mean(brier_deltas < 0)),
                "non_worse_accuracy_ratio": float(np.mean(accuracy_deltas >= -0.005)),
            },
        },
        "final_holdout": {
            "baseline": hold_baseline,
            "candidate": hold_candidate_m,
            "delta": {
                "accuracy": hold_candidate_m["accuracy"] - hold_baseline["accuracy"],
                "logloss": hold_candidate_m["logloss"] - hold_baseline["logloss"],
                "brier": hold_candidate_m["brier"] - hold_baseline["brier"],
                "ece": hold_candidate_m["ece"] - hold_baseline["ece"],
            },
            "risk_coverage": float(np.mean(hold_risk >= final_threshold)),
            "expert": final_expert,
            "threshold": final_threshold,
            "threshold_validation": final_threshold_info,
        },
        "eligibility": eligible,
        "blocks_detail": blocks,
    }


def main() -> None:
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "horizons": {h: evaluate(h) for h in HORIZONS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
