"""Research-only causal class-prior recalibration for BTC 5m/10m probabilities.

The live Champion is intentionally untouched. The experiment tests whether the
large observed under-frequency of FLAT is partly a probability-prior mismatch.
For every unseen OOS block, the adjustment is fitted only from labels that had
already settled before that block began. The final holdout uses a policy frozen
on development data and is descriptive only.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from model_compare import (
        CLASSES,
        load_primary_production_strict_rows,
        metrics,
    )
except ModuleNotFoundError:
    from src.model_compare import (
        CLASSES,
        load_primary_production_strict_rows,
        metrics,
    )

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "class_prior_recalibration_oos.json"

HORIZONS = ("5m", "10m")
FINAL_HOLDOUT_FRAC = 0.20
MIN_ROWS = 60
MIN_HISTORY = 30
TEST_BLOCK = 10
MIN_TUNING_ROWS = 10
SMOOTHING = 3.0
GAMMA_GRID = tuple(float(x) for x in np.arange(0.0, 1.51, 0.25))


def _norm(p):
    arr = np.asarray(p, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("probabilities must have shape (n, 3)")
    if not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError("probabilities contain invalid values")
    totals = arr.sum(axis=1, keepdims=True)
    if np.any(totals <= 0) or not np.isfinite(totals).all():
        raise ValueError("probability rows must have positive finite sums")
    out = np.clip(arr, 1e-8, 1.0)
    out /= out.sum(axis=1, keepdims=True)
    return out[0] if squeeze else out


def empirical_prior(history):
    """Estimate the settled class prior from one strictly causal history."""
    counts = np.array(
        [sum(str(r["y"]) == c for r in history) for c in CLASSES],
        dtype=float,
    )
    if len(history) <= 0:
        raise ValueError("empty_history")
    return (counts + SMOOTHING) / (float(len(history)) + SMOOTHING * len(CLASSES))


def predicted_prior(history):
    """Estimate the model's emitted probability prior from the same history."""
    p = _norm([r["production"] for r in history])
    mean = p.mean(axis=0)
    mean = np.clip(mean, 1e-8, 1.0)
    return mean / mean.sum()


def adjust_probs(probs, target_prior, model_prior, gamma):
    """Apply a bounded class-logit offset; gamma=0 is exactly the raw model."""
    p = _norm(probs)
    target = np.clip(np.asarray(target_prior, dtype=float), 1e-8, 1.0)
    reference = np.clip(np.asarray(model_prior, dtype=float), 1e-8, 1.0)
    offset = np.log(target / reference)
    out = p * np.exp(float(gamma) * offset)
    return _norm(out)


def _history_ready(rows, block_start):
    start = datetime.fromisoformat(str(block_start).replace("Z", "+00:00"))
    eligible = []
    for row in rows:
        created = datetime.fromisoformat(str(row["created"]).replace("Z", "+00:00"))
        target = datetime.fromisoformat(str(row["target"]).replace("Z", "+00:00"))
        if created >= target:
            continue
        if target < start:
            eligible.append(row)
    return eligible


def choose_gamma(history):
    """Tune gamma using only an inner, already-settled tail of history."""
    if len(history) < MIN_HISTORY:
        return None
    split = max(int(len(history) * 0.75), 1)
    fit = history[:split]
    tuning = history[split:]
    if len(tuning) < MIN_TUNING_ROWS or len(fit) < 10:
        return None

    target_prior = empirical_prior(fit)
    model_prior = predicted_prior(fit)
    ys = [r["y"] for r in tuning]
    raw = np.asarray([r["production"] for r in tuning], dtype=float)
    candidates = []
    for gamma in GAMMA_GRID:
        adjusted = adjust_probs(raw, target_prior, model_prior, gamma)
        score = metrics(ys, adjusted)
        candidates.append((score["logloss"], score["brier"], -score["accuracy"], gamma, score))
    _, _, _, gamma, score = min(candidates)
    return {
        "gamma": float(gamma),
        "target_prior": target_prior.tolist(),
        "model_prior": model_prior.tolist(),
        "tuning_n": len(tuning),
        "tuning_metrics": score,
    }


def _policy_from_full_history(history, gamma):
    return {
        "gamma": float(gamma),
        "target_prior": empirical_prior(history).tolist(),
        "model_prior": predicted_prior(history).tolist(),
        "history_n": len(history),
    }


def _apply_policy(rows, policy):
    p = np.asarray([r["production"] for r in rows], dtype=float)
    return adjust_probs(p, policy["target_prior"], policy["model_prior"], policy["gamma"])


def _argmax_rates(probs):
    p = _norm(probs)
    idx = p.argmax(axis=1)
    return {c: float(np.mean(idx == i)) for i, c in enumerate(CLASSES)}


def _prior_summary(rows):
    target_counts = {c: sum(str(r["y"]) == c for r in rows) for c in CLASSES}
    model_mean = predicted_prior(rows).tolist() if rows else [None] * 3
    total = max(len(rows), 1)
    return {
        "n": len(rows),
        "target_share": {c: float(target_counts[c] / total) for c in CLASSES},
        "model_mean_probability": {c: model_mean[i] for i, c in enumerate(CLASSES)},
    }


def _evaluate_blocks(rows):
    blocks = []
    for end in range(MIN_HISTORY, len(rows), TEST_BLOCK):
        test = rows[end:min(end + TEST_BLOCK, len(rows))]
        if len(test) < TEST_BLOCK:
            break
        history = _history_ready(rows[:end], test[0]["created"])
        selection = choose_gamma(history)
        if selection is None:
            continue
        policy = _policy_from_full_history(history, selection["gamma"])
        raw = np.asarray([r["production"] for r in test], dtype=float)
        adjusted = _apply_policy(test, policy)
        ys = [r["y"] for r in test]
        raw_m = metrics(ys, raw)
        adj_m = metrics(ys, adjusted)
        blocks.append({
            "test_start": test[0]["created"],
            "test_end": test[-1]["created"],
            "history_n": len(history),
            "n": len(test),
            "selection": selection,
            "policy": policy,
            "raw": raw_m,
            "adjusted": adj_m,
            "delta": {
                "accuracy": adj_m["accuracy"] - raw_m["accuracy"],
                "logloss": adj_m["logloss"] - raw_m["logloss"],
                "brier": adj_m["brier"] - raw_m["brier"],
                "calibration_error": adj_m["calibration_error"] - raw_m["calibration_error"],
            },
            "raw_argmax_rate": _argmax_rates(raw),
            "adjusted_argmax_rate": _argmax_rates(adjusted),
            "actual_class_share": {
                c: float(sum(y == c for y in ys) / len(ys)) for c in CLASSES
            },
        })
    return blocks


def evaluate(horizon):
    rows = load_primary_production_strict_rows(horizon)
    if len(rows) < MIN_ROWS:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_strict_binance_primary_rows",
            "n": len(rows),
            "minimum_rows": MIN_ROWS,
            "promotion_evidence_eligible": False,
        }

    rows = rows[-12000:]
    dev_end = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    development = rows[:dev_end]
    holdout = rows[dev_end:]
    blocks = _evaluate_blocks(development)

    if not blocks:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_causal_oos_blocks",
            "n": len(rows),
            "development_n": len(development),
            "holdout_n": len(holdout),
            "promotion_evidence_eligible": False,
        }

    raw_devs = []
    adj_devs = []
    for b in blocks:
        raw_devs.append(b["raw"])
        adj_devs.append(b["adjusted"])
    raw_acc = float(np.mean([m["accuracy"] for m in raw_devs]))
    adj_acc = float(np.mean([m["accuracy"] for m in adj_devs]))
    raw_ll = float(np.mean([m["logloss"] for m in raw_devs]))
    adj_ll = float(np.mean([m["logloss"] for m in adj_devs]))
    raw_br = float(np.mean([m["brier"] for m in raw_devs]))
    adj_br = float(np.mean([m["brier"] for m in adj_devs]))
    raw_ece = float(np.mean([m["calibration_error"] for m in raw_devs]))
    adj_ece = float(np.mean([m["calibration_error"] for m in adj_devs]))

    holdout_policy = None
    holdout_result = {"status": "DEFERRED", "n": len(holdout)}
    history = _history_ready(development, holdout[0]["created"]) if holdout else []
    selection = choose_gamma(history)
    if selection is not None and holdout:
        holdout_policy = _policy_from_full_history(history, selection["gamma"])
        raw = np.asarray([r["production"] for r in holdout], dtype=float)
        adjusted = _apply_policy(holdout, holdout_policy)
        ys = [r["y"] for r in holdout]
        raw_m = metrics(ys, raw)
        adj_m = metrics(ys, adjusted)
        holdout_result = {
            "status": "OK",
            "n": len(holdout),
            "raw": raw_m,
            "adjusted": adj_m,
            "delta": {
                "accuracy": adj_m["accuracy"] - raw_m["accuracy"],
                "logloss": adj_m["logloss"] - raw_m["logloss"],
                "brier": adj_m["brier"] - raw_m["brier"],
                "calibration_error": adj_m["calibration_error"] - raw_m["calibration_error"],
            },
            "raw_argmax_rate": _argmax_rates(raw),
            "adjusted_argmax_rate": _argmax_rates(adjusted),
            "actual_class_share": {c: float(sum(y == c for y in ys) / len(ys)) for c in CLASSES},
            "frozen_policy": holdout_policy,
            "used_for_selection": False,
            "used_for_gate": False,
            "descriptive_only": True,
        }

    improvement = {
        "mean_accuracy_delta": adj_acc - raw_acc,
        "mean_logloss_delta": adj_ll - raw_ll,
        "mean_brier_delta": adj_br - raw_br,
        "mean_calibration_error_delta": adj_ece - raw_ece,
        "improved_logloss_ratio": float(np.mean([b["delta"]["logloss"] < 0 for b in blocks])),
        "improved_brier_ratio": float(np.mean([b["delta"]["brier"] < 0 for b in blocks])),
        "non_worse_accuracy_ratio": float(np.mean([b["delta"]["accuracy"] >= -0.005 for b in blocks])),
    }

    # This experiment is deliberately evidence-generating only. Even a positive
    # result cannot promote the Champion because class-prior correction changes
    # probability semantics and needs an independent, longer OOS gate.
    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "policy": "causal_class_prior_logit_offset_from_settled_history",
        "smoothing": SMOOTHING,
        "gamma_grid": list(GAMMA_GRID),
        "data_source": "live_binance_primary_strict_pit",
        "n": len(rows),
        "development_n": len(development),
        "final_holdout_protected": True,
        "final_holdout_used_for_selection": False,
        "development_baseline_prior_summary": _prior_summary(development),
        "nested_oos": {
            "blocks": len(blocks),
            "raw_mean": {
                "accuracy": raw_acc,
                "logloss": raw_ll,
                "brier": raw_br,
                "calibration_error": raw_ece,
            },
            "adjusted_mean": {
                "accuracy": adj_acc,
                "logloss": adj_ll,
                "brier": adj_br,
                "calibration_error": adj_ece,
            },
            "delta": improvement,
        },
        "blocks": blocks,
        "final_holdout": holdout_result,
    }


def main():
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "horizons": {h: evaluate(h) for h in HORIZONS},
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
