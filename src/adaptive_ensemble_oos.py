"""Research-only rolling OOS test for adaptive ensemble weighting."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
for _path in (ROOT_DIR, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from src.model_compare import HORIZONS, load_primary_production_strict_rows, load_archive_research_rows, metrics, apply_temperature, _temperature
    load_rows = load_primary_production_strict_rows
    from src.ensemble_model import SoftVotingEnsemble
except ModuleNotFoundError:
    from model_compare import HORIZONS, load_primary_production_strict_rows, load_archive_research_rows, metrics, apply_temperature, _temperature
    from ensemble_model import SoftVotingEnsemble

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "adaptive_ensemble_oos.json"
MIN_TRAIN = 2000
TEST_BLOCK = 150
MAX_BLOCKS = 16
MAX_ROWS = 7000
PURGE = {"5m": 5, "10m": 10}
EMBARGO = {"5m": 60, "10m": 60}


def _endpoints(n: int):
    raw = list(range(MIN_TRAIN, n, TEST_BLOCK))
    if len(raw) <= MAX_BLOCKS:
        return raw
    return sorted(set(int(x) for x in np.linspace(raw[0], raw[-1], MAX_BLOCKS)))


def _rolling_validation_weights(parts, y, min_weight=0.10, shrinkage=0.25):
    """Choose stable weights from multiple historical validation windows."""
    n = len(y)
    if len(parts) != 3 or n < 240:
        return None
    edges = [int(n * 0.50), int(n * 0.65), int(n * 0.80)]
    windows = [(edges[0], edges[1]), (edges[1], edges[2]), (edges[2], n)]
    values = np.arange(0.10, 0.71, 0.10)
    grid = []
    for a in values:
        for b in values:
            c = 1.0 - float(a) - float(b)
            if 0.10 - 1e-12 <= c <= 0.70 + 1e-12:
                grid.append((float(a), float(b), float(c)))
    best = None
    for weights in grid:
        fold_scores = []
        for start, end in windows:
            yi = np.asarray([("DOWN", "FLAT", "UP").index(str(v)) for v in y[start:end]], dtype=int)
            if len(yi) < 20:
                break
            mix = sum(w * p[start:end] for w, p in zip(weights, parts))
            mix = np.clip(mix, 1e-7, 1.0)
            mix /= mix.sum(axis=1, keepdims=True)
            one = np.eye(3)[yi]
            ll = float(-np.mean(np.log(mix[np.arange(len(yi)), yi])))
            br = float(np.mean(np.sum((mix - one) ** 2, axis=1)))
            fold_scores.append(ll + 0.15 * br)
        if len(fold_scores) != 3:
            continue
        prior = np.asarray((0.34, 0.33, 0.33))
        mean_score = float(np.mean(fold_scores))
        stability = float(np.std(fold_scores))
        distance = float(np.sum((np.asarray(weights) - prior) ** 2))
        key = (mean_score + 0.20 * stability + 0.01 * distance, mean_score, stability, weights)
        if best is None or key < best[0]:
            best = (key, weights)
    if best is None:
        return None
    chosen = np.asarray(best[1], dtype=float)
    chosen = (1.0 - shrinkage) * chosen + shrinkage * (1.0 / 3.0)
    chosen = np.maximum(chosen, float(min_weight))
    chosen /= chosen.sum()
    return tuple(float(v) for v in chosen)


def _predict_block_with_weights(train, test, weights):
    if weights is None:
        return None
    if len(train) < 120 or len(test) < 25:
        return None
    factory = SoftVotingEnsemble(learn_weights=False)
    models = []
    X = np.asarray([r["x"] for r in train], dtype=float)
    y = np.asarray([r["y"] for r in train])
    for sub_factory in factory._factories():
        model = sub_factory()
        model.fit(X, y)
        models.append(model)
    parts = [_aligned(model, test) for model in models]
    return sum(w * p for w, p in zip(weights, parts))


def _aligned(model, rows):
    X = np.asarray([r["x"] for r in rows], dtype=float)
    raw = np.asarray(model.predict_proba(X), dtype=float)
    out = np.full((len(rows), 3), 1e-6, dtype=float)
    idx = {"DOWN": 0, "FLAT": 1, "UP": 2}
    for j, cls in enumerate(model.classes_):
        if str(cls) in idx:
            out[:, idx[str(cls)]] = raw[:, j]
    out = np.clip(out, 1e-6, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def _predict_block(factory, train, test):
    split = int(len(train) * 0.75)
    if split < 120 or len(train) - split < 30:
        return None
    prefix_y = [r["y"] for r in train[:split]]
    if len(set(prefix_y)) < 3:
        return None
    cal_model = factory()
    try:
        cal_model.fit(
            np.asarray([r["x"] for r in train[:split]], dtype=float),
            np.asarray(prefix_y),
        )
    except ValueError as exc:
        if "three_classes" in str(exc):
            return None
        raise
    cal_probs = _aligned(cal_model, train[split:])
    temp = _temperature(cal_probs, [r["y"] for r in train[split:]])

    model = factory()
    model.fit(
        np.asarray([r["x"] for r in train], dtype=float),
        np.asarray([r["y"] for r in train]),
    )
    probs = _aligned(model, test)
    return apply_temperature(probs, temp)


def _evaluate_development(rows, horizon: str):
    equal_factory = lambda: SoftVotingEnsemble(learn_weights=False)
    adaptive_factory = lambda: SoftVotingEnsemble(learn_weights=True)
    blocks = []
    stable_blocks = []

    for end in _endpoints(len(rows)):
        train_end = max(0, end - PURGE[horizon] - EMBARGO[horizon])
        test = rows[end:min(end + TEST_BLOCK, len(rows))]
        train = rows[:train_end]
        if len(train) < MIN_TRAIN or len(test) < max(50, TEST_BLOCK // 2):
            continue
        eq = _predict_block(equal_factory, train, test)
        ad = _predict_block(adaptive_factory, train, test)
        probe_split = int(len(train) * 0.75)
        probe_parts = []
        if probe_split >= 120 and len(train) - probe_split >= 30:
            probe_factory = SoftVotingEnsemble(learn_weights=False)
            prefix = np.asarray([r["x"] for r in train[:probe_split]], dtype=float)
            prefix_y = np.asarray([r["y"] for r in train[:probe_split]])
            for sub_factory in probe_factory._factories():
                probe = sub_factory()
                probe.fit(prefix, prefix_y)
                probe_parts.append(_aligned(probe, train[probe_split:]))
        stable_weights = _rolling_validation_weights(
            probe_parts, [r["y"] for r in train[probe_split:]]
        ) if len(probe_parts) == 3 else None
        stable = _predict_block_with_weights(train, test, stable_weights)
        if eq is None or ad is None:
            continue
        y = [r["y"] for r in test]
        em = metrics(y, eq)
        am = metrics(y, ad)
        sm = metrics(y, stable) if stable is not None else None
        if sm is not None:
            stable_blocks.append({
                "n": len(y),
                "equal": em,
                "stable": sm,
                "delta": {
                    "accuracy": sm["accuracy"] - em["accuracy"],
                    "logloss": sm["logloss"] - em["logloss"],
                    "brier": sm["brier"] - em["brier"],
                },
            })
        blocks.append({
            "n": len(y),
            "equal": em,
            "adaptive": am,
            "delta": {
                "accuracy": am["accuracy"] - em["accuracy"],
                "logloss": am["logloss"] - em["logloss"],
                "brier": am["brier"] - em["brier"],
            },
        })
    if not blocks:
        return None

    stable_ll = np.asarray([b["delta"]["logloss"] for b in stable_blocks], dtype=float) if stable_blocks else np.asarray([], dtype=float)
    stable_br = np.asarray([b["delta"]["brier"] for b in stable_blocks], dtype=float) if stable_blocks else np.asarray([], dtype=float)
    stable_ac = np.asarray([b["delta"]["accuracy"] for b in stable_blocks], dtype=float) if stable_blocks else np.asarray([], dtype=float)
    ll = np.asarray([b["delta"]["logloss"] for b in blocks], dtype=float)
    br = np.asarray([b["delta"]["brier"] for b in blocks], dtype=float)
    ac = np.asarray([b["delta"]["accuracy"] for b in blocks], dtype=float)
    summary = {
        "blocks": len(blocks),
        "stable_blocks": len(stable_blocks),
        "stable_mean_accuracy_delta": float(stable_ac.mean()) if len(stable_ac) else None,
        "stable_mean_logloss_delta": float(stable_ll.mean()) if len(stable_ll) else None,
        "stable_mean_brier_delta": float(stable_br.mean()) if len(stable_br) else None,
        "stable_improved_logloss_ratio": float(np.mean(stable_ll < 0)) if len(stable_ll) else None,
        "stable_improved_brier_ratio": float(np.mean(stable_br < 0)) if len(stable_br) else None,
        "samples": int(sum(b["n"] for b in blocks)),
        "mean_accuracy_delta": float(ac.mean()),
        "mean_logloss_delta": float(ll.mean()),
        "mean_brier_delta": float(br.mean()),
        "improved_logloss_ratio": float(np.mean(ll < 0)),
        "improved_brier_ratio": float(np.mean(br < 0)),
        "non_worse_accuracy_ratio": float(np.mean(ac >= -0.005)),
    }
    eligible = (
        summary["blocks"] >= 8
        and summary["improved_logloss_ratio"] >= 0.60
        and summary["improved_brier_ratio"] >= 0.60
        and summary["mean_logloss_delta"] <= -0.003
        and summary["mean_brier_delta"] <= -0.0015
        and summary["non_worse_accuracy_ratio"] >= 0.80
    )
    stable_eligible = (
        summary["stable_blocks"] >= 8
        and summary["stable_improved_logloss_ratio"] is not None
        and summary["stable_improved_brier_ratio"] is not None
        and summary["stable_mean_logloss_delta"] <= -0.003
        and summary["stable_mean_brier_delta"] <= -0.0015
        and summary["stable_improved_logloss_ratio"] >= 0.60
        and summary["stable_improved_brier_ratio"] >= 0.60
        and summary["non_worse_accuracy_ratio"] >= 0.80
    )
    return summary, blocks, bool(eligible), bool(stable_eligible), stable_blocks


def evaluate(horizon: str):
    # Compatibility seam for deterministic tests; load_rows is the strict
    # primary-PIT loader used by production research evaluation.
    rows = load_rows(horizon)
    data_source = "live_binance_primary"
    if len(rows) < MIN_TRAIN + TEST_BLOCK + 100:
        archive = load_archive_research_rows(horizon, MAX_ROWS)
        if len(archive) > len(rows):
            rows = archive
            data_source = "binance_vision_archive"
    if len(rows) > MAX_ROWS:
        rows = rows[-MAX_ROWS:]

    if len(rows) < MIN_TRAIN + TEST_BLOCK + 100:
        return {
            "status": "DEFERRED",
            "n": len(rows),
            "data_source": data_source,
            "reason": "insufficient_research_rows_for_protected_holdout",
        }

    split = int(len(rows) * 0.80)
    development = rows[:split]
    holdout = rows[split:]
    if len(development) < MIN_TRAIN + TEST_BLOCK:
        return {
            "status": "DEFERRED",
            "n": len(rows),
            "reason": "insufficient_development_rows",
        }

    dev_result = _evaluate_development(development, horizon)
    if dev_result is None:
        return {
            "status": "DEFERRED",
            "n": len(rows),
            "reason": "no_valid_development_blocks",
        }
    summary, blocks, eligible, stable_eligible, stable_blocks = dev_result

    # Final 20% is protected. No threshold, weight, or eligibility decision
    # is made from it; it is evaluated once using the frozen development
    # protocol and the full development history as training data.
    equal_factory = lambda: SoftVotingEnsemble(learn_weights=False)
    adaptive_factory = lambda: SoftVotingEnsemble(learn_weights=True)
    eq = _predict_block(equal_factory, development, holdout)
    ad = _predict_block(adaptive_factory, development, holdout)
    if eq is None or ad is None:
        final_holdout = {
            "status": "DEFERRED",
            "reason": "holdout_prediction_failed",
        }
    else:
        y_hold = [r["y"] for r in holdout]
        em = metrics(y_hold, eq)
        am = metrics(y_hold, ad)
        final_holdout = {
            "equal": em,
            "adaptive": am,
            "delta": {
                "accuracy": am["accuracy"] - em["accuracy"],
                "logloss": am["logloss"] - em["logloss"],
                "brier": am["brier"] - em["brier"],
            },
        }

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "final_holdout_protected": True,
        "data_source": data_source,
        "promotion_evidence_eligible": data_source == "live_binance_primary",
        "final_holdout_used_for_selection": False,
        "policy": "adaptive_ensemble_weight_learning_on_pre_test_training_window_only",
        "eligible_pending_frozen_holdout_confirmation": bool(eligible),
        "summary": summary,
        "development_n": len(development),
        "final_holdout_n": len(holdout),
        "final_holdout": final_holdout,
        "blocks": blocks,
    }


def main():
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
