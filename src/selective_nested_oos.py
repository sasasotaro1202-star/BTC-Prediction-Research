"""Research-only nested selective OOS gate for BTC short-horizon direction.

A selective policy may abstain, but its thresholds are chosen only from completed
prior OOS observations. Each future block is scored with a policy frozen before
that block. The final holdout is evaluated once for description only.
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from feature_schema import FEATURES
from model_compare import (
    HORIZONS,
    MIN_OOS,
    load_primary_production_strict_rows,
    load_archive_research_rows,
    aligned,
)

OUT = ROOT / "data" / "historical_research" / "selective_nested_oos.json"
MODEL_DIR = ROOT / "models"
CLASSES = ("DOWN", "FLAT", "UP")

FINAL_HOLDOUT_FRAC = 0.20
MIN_TRAIN = 2000
TEST_BLOCK = 150
MAX_ROWS = 12000
MIN_SELECTION_HISTORY = 600
MIN_POLICY_SELECTED = 100
MIN_BLOCK_SELECTED = 20
MIN_COVERAGE = 0.05
TARGET_ACCURACY = 0.70
Z95 = 1.959963984540054

CONFIDENCE_GRID = tuple(float(x) for x in np.arange(0.55, 0.951, 0.025))
MARGIN_GRID = tuple(float(x) for x in np.arange(0.05, 0.801, 0.025))


def _norm(p):
    arr = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0)
    if arr.ndim == 1:
        total = float(arr.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("invalid probability vector")
        return arr / total
    if arr.ndim != 2 or arr.shape[1] != len(CLASSES):
        raise ValueError("probabilities must be shape (n, 3)")
    totals = arr.sum(axis=1, keepdims=True)
    if not np.isfinite(totals).all() or np.any(totals <= 0):
        raise ValueError("invalid probability matrix")
    return arr / totals


def _wilson_lower(hits: int, n: int, z: float = Z95) -> float:
    if n <= 0:
        return 0.0
    p = float(hits) / float(n)
    den = 1.0 + (z * z) / n
    center = p + (z * z) / (2.0 * n)
    spread = z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)
    return float(max(0.0, (center - spread) / den))


def _score_rows(rows):
    p = _norm(np.asarray([r["production"] for r in rows], dtype=float))
    confidence = p.max(axis=1)
    sorted_p = np.sort(p, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    pred = p.argmax(axis=1)
    y = np.asarray([CLASSES.index(str(r["y"])) for r in rows], dtype=int)
    return p, confidence, margin, pred, y


def evaluate_policy(rows, min_confidence, min_margin):
    if not rows:
        return {"coverage": 0.0, "n": 0, "accuracy": None, "lcb95": 0.0, "hits": 0}
    _, confidence, margin, pred, y = _score_rows(rows)
    mask = (confidence >= float(min_confidence)) & (margin >= float(min_margin))
    n = int(mask.sum())
    hits = int(np.sum(pred[mask] == y[mask])) if n else 0
    coverage = float(n / len(rows))
    accuracy = float(hits / n) if n else None
    return {
        "coverage": coverage,
        "n": n,
        "accuracy": accuracy,
        "lcb95": _wilson_lower(hits, n),
        "hits": hits,
    }


def _candidate_key(result):
    target_hit = 1 if result["lcb95"] >= TARGET_ACCURACY else 0
    accuracy = result["accuracy"] if result["accuracy"] is not None else -1.0
    return (
        target_hit,
        float(result["lcb95"]),
        float(accuracy),
        float(result["coverage"]),
        int(result["n"]),
    )


def _parse_dt(value):
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def causal_history(rows, test_start):
    """Keep only labels that were settled before the next test block began."""
    start = _parse_dt(test_start)
    if start is None:
        return []
    out = []
    for row in rows:
        target = _parse_dt(row.get("target"))
        created = _parse_dt(row.get("created"))
        if target is None or created is None:
            continue
        if created >= target:
            continue
        if target < start:
            out.append(row)
    return out


def choose_policy(history):
    """Choose a gate only from observations whose labels were already known."""

    if len(history) < MIN_SELECTION_HISTORY:
        return None

    # Cap selection history to the recent causal window to avoid silently
    # treating years-old calibration as equally representative of the current
    # short-horizon regime.
    sample = history[-min(len(history), MAX_ROWS):]
    candidates = []
    for confidence in CONFIDENCE_GRID:
        for margin in MARGIN_GRID:
            r = evaluate_policy(sample, confidence, margin)
            if r["n"] < MIN_POLICY_SELECTED:
                continue
            if r["coverage"] < MIN_COVERAGE:
                continue
            candidates.append((
                _candidate_key(r),
                float(confidence),
                float(margin),
                r,
            ))
    if not candidates:
        return None

    # Primary objective is conservative lower-bound accuracy, not raw hit rate.
    # Coverage is only a secondary tie-breaker so a tiny sample cannot dominate.
    _, confidence, margin, evidence = max(candidates, key=lambda item: item[0])
    return {
        "min_confidence": confidence,
        "min_margin": margin,
        "selection_evidence": evidence,
        "selection_n": len(sample),
        "selection_target_accuracy": TARGET_ACCURACY,
    }


def _load_rows(horizon):
    rows = load_primary_production_strict_rows(horizon)
    source = "live_binance_primary"
    if len(rows) < MIN_TRAIN + TEST_BLOCK + MIN_SELECTION_HISTORY:
        archive = load_archive_research_rows(horizon, MAX_ROWS)
        if len(archive) > len(rows):
            try:
                model = joblib.load(MODEL_DIR / f"{horizon}.joblib")
                for row in archive:
                    x = np.asarray([row["x"]], dtype=float)
                    row["production"] = aligned(model, x)[0].tolist()
                rows = archive
                source = "binance_vision_archive_frozen_champion"
            except Exception as exc:
                return [], source, f"archive_model_scoring_failed:{type(exc).__name__}"
    rows = rows[-MAX_ROWS:]
    valid = []
    for row in rows:
        p = row.get("production")
        y = row.get("y")
        x = row.get("x")
        if y not in CLASSES or p is None or x is None:
            continue
        arr = np.asarray(p, dtype=float)
        if arr.shape != (3,) or not np.isfinite(arr).all() or float(arr.sum()) <= 0:
            continue
        valid.append({**row, "production": _norm(arr).tolist()})
    return valid, source, None


def _block_endpoints(n):
    endpoints = list(range(MIN_TRAIN, n, TEST_BLOCK))
    if len(endpoints) <= 24:
        return endpoints
    return sorted(set(int(v) for v in np.linspace(endpoints[0], endpoints[-1], 24)))


def _nested_development(rows):
    blocks = []
    policies = []
    for end in _block_endpoints(len(rows)):
        test = rows[end:min(end + TEST_BLOCK, len(rows))]
        history = causal_history(rows[:end], test[0].get("created"))
        if len(test) < MIN_BLOCK_SELECTED:
            continue
        policy = choose_policy(history)
        if policy is None:
            continue
        result = evaluate_policy(test, policy["min_confidence"], policy["min_margin"])
        if result["n"] < MIN_BLOCK_SELECTED:
            continue
        blocks.append({
            "n_total": len(test),
            "n_selected": result["n"],
            "coverage": result["coverage"],
            "accuracy": result["accuracy"],
            "lcb95": result["lcb95"],
            "min_confidence": policy["min_confidence"],
            "min_margin": policy["min_margin"],
            "selection_n": policy["selection_n"],
            "selection_accuracy": policy["selection_evidence"]["accuracy"],
            "selection_lcb95": policy["selection_evidence"]["lcb95"],
        })
        policies.append(policy)

    if not blocks:
        return None

    selected = sum(int(b["n_selected"]) for b in blocks)
    hits = sum(int(round(b["accuracy"] * b["n_selected"])) for b in blocks)
    overall_accuracy = float(hits / selected) if selected else None
    coverages = np.asarray([b["coverage"] for b in blocks], dtype=float)
    accuracies = np.asarray([b["accuracy"] for b in blocks], dtype=float)
    lcbs = np.asarray([b["lcb95"] for b in blocks], dtype=float)
    summary = {
        "blocks": len(blocks),
        "selected_n": selected,
        "overall_accuracy": overall_accuracy,
        "overall_lcb95": _wilson_lower(hits, selected),
        "mean_block_accuracy": float(accuracies.mean()),
        "mean_block_lcb95": float(lcbs.mean()),
        "block_accuracy_ge_70_ratio": float(np.mean(accuracies >= TARGET_ACCURACY)),
        "block_lcb95_ge_70_ratio": float(np.mean(lcbs >= TARGET_ACCURACY)),
        "mean_coverage": float(coverages.mean()),
        "min_block_coverage": float(coverages.min()),
        "max_block_coverage": float(coverages.max()),
    }
    return {"summary": summary, "blocks": blocks, "policies": policies}


def _final_holdout(rows, dev_end):
    development = rows[:dev_end]
    holdout = rows[dev_end:]
    holdout_start = holdout[0].get("created") if holdout else None
    policy = choose_policy(causal_history(development, holdout_start))
    if policy is None:
        return {
            "status": "DEFERRED",
            "reason": "no_frozen_policy_from_development",
            "n": len(holdout),
        }
    result = evaluate_policy(holdout, policy["min_confidence"], policy["min_margin"])
    return {
        "status": "OK",
        "n": len(holdout),
        "selected_n": result["n"],
        "coverage": result["coverage"],
        "accuracy": result["accuracy"],
        "lcb95": result["lcb95"],
        "hits": result["hits"],
        "frozen_policy": {
            "min_confidence": policy["min_confidence"],
            "min_margin": policy["min_margin"],
            "selection_n": policy["selection_n"],
        },
        "used_for_selection": False,
        "used_for_gate": False,
        "descriptive_only": True,
    }


def evaluate(horizon):
    rows, source, load_error = _load_rows(horizon)
    if load_error:
        return {
            "status": "DEFERRED",
            "reason": load_error,
            "data_source": source,
            "promotion_evidence_eligible": source == "live_binance_primary",
        }
    if len(rows) < MIN_TRAIN + TEST_BLOCK + MIN_SELECTION_HISTORY + 100:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_rows_for_nested_selective_oos",
            "n": len(rows),
            "data_source": source,
            "promotion_evidence_eligible": source == "live_binance_primary",
        }

    dev_end = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    dev = _nested_development(rows[:dev_end])
    if dev is None:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_nested_development_blocks",
            "n": len(rows),
            "data_source": source,
            "promotion_evidence_eligible": source == "live_binance_primary",
        }

    holdout = _final_holdout(rows, dev_end)
    s = dev["summary"]
    nested_eligible = bool(
        s["blocks"] >= 8
        and s["selected_n"] >= 1000
        and s["overall_accuracy"] is not None
        and s["overall_accuracy"] >= TARGET_ACCURACY
        and s["overall_lcb95"] >= TARGET_ACCURACY
        and s["block_lcb95_ge_70_ratio"] >= 0.60
        and s["mean_coverage"] >= MIN_COVERAGE
        and s["min_block_coverage"] >= MIN_COVERAGE / 2.0
    )

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "data_source": source,
        "promotion_evidence_eligible": source == "live_binance_primary",
        "final_holdout_protected": True,
        "final_holdout_used_for_selection": False,
        "policy": "nested_causal_selective_prediction_wilson_lower_bound",
        "target_accuracy": TARGET_ACCURACY,
        "minimum_selection_coverage": MIN_COVERAGE,
        "minimum_policy_selected": MIN_POLICY_SELECTED,
        "nested_oos": {
            **dev["summary"],
            "promotion_candidate": nested_eligible,
        },
        "blocks": dev["blocks"],
        "final_holdout": holdout,
    }


def main():
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "feature_schema": list(FEATURES),
        "horizons": {h: evaluate(h) for h in HORIZONS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
