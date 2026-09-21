"""Chronological OOS evaluator for the research-only BTC context router.

This module deliberately never changes production artifacts. It evaluates whether
context-aware dynamic routing is more reliable than its pre-test global model
under rolling chronological splits. All routing decisions are made from rows
strictly before each test block.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model_compare import HORIZONS, load_rows
from context_model_oos import dynamic_route_predictions, evaluate_routing


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "context_router_oos.json"
MIN_TRAIN = 1000
TEST_BLOCK = 25


def factories():
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logreg_c0.1": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.1, max_iter=3000)),
        ]),
        "extra_trees_500": lambda: ExtraTreesClassifier(
            n_estimators=500, max_depth=7, min_samples_leaf=8,
            max_features="sqrt", random_state=42, n_jobs=-1,
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=250, max_leaf_nodes=15, learning_rate=0.04,
            l2_regularization=1.0, random_state=42,
        ),
    }


def _metric_delta(global_m, routed_m):
    return {
        "accuracy_delta": routed_m["accuracy"] - global_m["accuracy"],
        "logloss_delta": routed_m["logloss"] - global_m["logloss"],
        "brier_delta": routed_m["brier"] - global_m["brier"],
    }


def evaluate_horizon(horizon):
    rows = load_rows(horizon)
    if len(rows) < MIN_TRAIN + TEST_BLOCK:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_chronological_rows",
            "n": len(rows),
        }

    blocks = []
    for end in range(MIN_TRAIN, len(rows), TEST_BLOCK):
        train = rows[:end]
        test = rows[end:min(end + TEST_BLOCK, len(rows))]
        if len(test) < max(10, TEST_BLOCK // 2):
            continue
        routed = dynamic_route_predictions(train, test, factories())
        if routed is None:
            continue

        y = [r["y"] for r in test]
        metrics = evaluate_routing(y, routed["routed_probs"], routed["global_probs"])
        delta = _metric_delta(metrics["global"], metrics["routed"])
        blocks.append({
            "n": len(test),
            "metrics": metrics,
            "delta": delta,
            "global_model": routed.get("global_model"),
            "global_weights": routed.get("global_weights", {}),
            "context_weights": routed.get("context_weights", {}),
        })

    if not blocks:
        return {"status": "DEFERRED", "reason": "no_valid_router_blocks", "n": len(rows)}

    gl = [b["delta"]["logloss_delta"] for b in blocks]
    gb = [b["delta"]["brier_delta"] for b in blocks]
    ga = [b["delta"]["accuracy_delta"] for b in blocks]
    summary = {
        "blocks": len(blocks),
        "samples": int(sum(b["n"] for b in blocks)),
        "mean_logloss_delta": float(np.mean(gl)),
        "mean_brier_delta": float(np.mean(gb)),
        "mean_accuracy_delta": float(np.mean(ga)),
        "improved_logloss_ratio": float(np.mean(np.asarray(gl) < 0)),
        "improved_brier_ratio": float(np.mean(np.asarray(gb) < 0)),
        "non_worse_accuracy_ratio": float(np.mean(np.asarray(ga) >= -0.01)),
    }

    # Conservative research-only promotion candidate gate. This is evidence,
    # not permission to alter production.
    eligible = (
        summary["blocks"] >= 12
        and summary["improved_logloss_ratio"] >= 0.60
        and summary["improved_brier_ratio"] >= 0.60
        and summary["mean_logloss_delta"] <= -0.005
        and summary["mean_brier_delta"] <= -0.002
        and summary["non_worse_accuracy_ratio"] >= 0.80
    )
    return {
        "status": "OK",
        "research_only": True,
        "final_holdout_protected": True,
        "production_changed": False,
        "eligible_pending_frozen_holdout_confirmation": bool(eligible),
        "summary": summary,
        "blocks": blocks,
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "research_only": True,
        "policy": "diagnostic_only_no_model_input_no_promotion_effect",
        "final_holdout_protected": True,
        "production_changed": False,
        "horizons": {h: evaluate_horizon(h) for h in HORIZONS},
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
