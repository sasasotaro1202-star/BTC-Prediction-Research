"""Research-only disagreement-gated prequential routing for BTC.

The production Champion remains unchanged. Alternative experts may replace the
Champion only when they disagree on the current class and their historical
accuracy on *previous disagreement cases* is sufficiently better. The routing
state is updated only after each current outcome is observed.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_compare import (
    CLASSES,
    EMBARGO_BARS,
    MIN_OOS,
    MIN_TRAIN,
    PURGE_BARS,
    _temperature,
    aligned,
    apply_temperature,
    load_archive_research_rows,
    metrics,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUT = ROOT / "data" / "historical_research" / "disagreement_gated_routing_oos.json"

HORIZONS = ("5m", "10m")
EXPERTS = ("production", "logreg", "extra_trees", "hgb")
ALTERNATIVES = EXPERTS[1:]
MAX_ROWS = 12000
FINAL_HOLDOUT_FRAC = 0.20
TEST_BLOCK = 500
CAL_BLOCK = 500
CONTEXT_SHRINK_K = 40.0
WARMUP_ROWS = 100
MIN_DISAGREEMENT = 120
MIN_CONTEXT_DISAGREEMENT = 40
MIN_ADVANTAGE = 0.015
MIN_POSTERIOR_LOWER = 0.50
EPS = 1e-7
BOOTSTRAPS = 1000


def _prob(v: Any) -> np.ndarray:
    p = np.asarray(v, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or np.any(p < 0) or float(p.sum()) <= 0:
        raise ValueError("invalid_probability_vector")
    p = np.clip(p, EPS, 1.0)
    return p / p.sum()


def _context(x: list[float]) -> str:
    r5, r10, v5, v10, rp = float(x[2]), float(x[3]), float(x[5]), float(x[6]), float(x[7])
    trend = "UP" if r5 > 0 and r10 > 0 else "DOWN" if r5 < 0 and r10 < 0 else "MIXED"
    vol = "HIGH" if v10 >= max(v5, EPS) else "LOW"
    location = "LOW" if rp < 1 / 3 else "HIGH" if rp > 2 / 3 else "MID"
    return f"{trend}|{vol}|{location}"


def _factories() -> dict[str, Any]:
    return {
        "logreg": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.3, max_iter=2500)),
        ]),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=220,
            max_depth=9,
            min_samples_leaf=12,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=200,
            max_leaf_nodes=15,
            learning_rate=0.04,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


def _load_production(horizon: str):
    path = MODEL_DIR / f"{horizon}.joblib"
    if not path.is_file():
        raise ValueError(f"production_artifact_missing:{horizon}")
    return joblib.load(path)


def _attach_production(rows: list[dict[str, Any]], horizon: str) -> list[dict[str, Any]]:
    model = _load_production(horizon)
    X = np.asarray([r["x"] for r in rows], dtype=float)
    p = aligned(model, X)
    out = []
    for row, probs in zip(rows, p):
        item = dict(row)
        item["production"] = _prob(probs)
        out.append(item)
    return out


def _fit_alternatives(train: list[dict[str, Any]], cal: list[dict[str, Any]]) -> dict[str, Any]:
    if len(train) < MIN_TRAIN or len(cal) < 100:
        raise ValueError("insufficient_train_cal")
    X_train = np.asarray([r["x"] for r in train], dtype=float)
    y_train = np.asarray([r["y"] for r in train], dtype=str)
    X_cal = np.asarray([r["x"] for r in cal], dtype=float)
    y_cal = np.asarray([r["y"] for r in cal], dtype=str)
    if len(set(y_train)) < 3 or len(set(y_cal)) < 3:
        raise ValueError("single_class_train_cal")

    out = {}
    for name, factory in _factories().items():
        cal_model = factory()
        cal_model.fit(X_train, y_train)
        cal_probs = aligned(cal_model, X_cal)
        temperature = _temperature(cal_probs, y_cal)

        model = factory()
        model.fit(X_train, y_train)
        out[name] = (model, float(temperature))
    return out


def _predict_experts(models: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    X = np.asarray([r["x"] for r in rows], dtype=float)
    out = {}
    for name, (model, temperature) in models.items():
        out[name] = apply_temperature(aligned(model, X), temperature)
    return out


def _wilson_lower(hits: int, n: int, z: float = 1.959963984540054) -> float:
    if n <= 0:
        return 0.0
    p = hits / n
    z2 = z * z
    den = 1 + z2 / n
    center = p + z2 / (2 * n)
    spread = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)
    return max(0.0, (center - spread) / den)


def _new_state() -> dict[str, Any]:
    return {
        "global": {e: [0, 0] for e in EXPERTS},
        "context": defaultdict(lambda: {e: [0, 0] for e in EXPERTS}),
        "seen": 0,
    }


def _posterior_rates(state: dict[str, Any], context: str) -> dict[str, tuple[float, float, int]]:
    result = {}
    for expert in EXPERTS:
        g_hits, g_n = state["global"][expert][0], state["global"][expert][1]
        c_hits, c_n = state["context"][context][expert][0], state["context"][context][expert][1]

        # Only disagreement outcomes are stored. Context evidence shrinks to
        # global evidence, preventing tiny-context overreaction.
        if c_n >= MIN_CONTEXT_DISAGREEMENT:
            weight = c_n / (c_n + CONTEXT_SHRINK_K)
            hits = weight * c_hits + (1 - weight) * g_hits
            n = weight * c_n + (1 - weight) * g_n
        else:
            hits, n = float(g_hits), float(g_n)

        # Jeffreys smoothing keeps early rates finite and deterministic.
        mean = (hits + 0.5) / (n + 1.0)
        # Conservative lower bound uses the rounded effective counts.
        lower = _wilson_lower(int(round(hits)), int(round(n))) if n >= 1 else 0.0
        result[expert] = (mean, lower, int(round(n)))
    return result


def _route_one(
    state: dict[str, Any],
    probs: dict[str, np.ndarray],
    row: dict[str, Any],
) -> tuple[np.ndarray, str, dict[str, Any]]:
    production = probs["production"]
    prod_class = int(np.argmax(production))
    context = _context(row["x"])

    if int(state["seen"]) < WARMUP_ROWS:
        return production, "production:warmup", {"routed": False}

    rates = _posterior_rates(state, context)
    prod_mean, _, prod_n = rates["production"]

    candidates = []
    for expert in ALTERNATIVES:
        alt = probs[expert]
        if int(np.argmax(alt)) == prod_class:
            continue
        mean, lower, n = rates[expert]
        if n < MIN_DISAGREEMENT:
            continue
        advantage = mean - prod_mean
        if advantage < MIN_ADVANTAGE or lower < MIN_POSTERIOR_LOWER:
            continue
        candidates.append((advantage, lower, mean, expert))

    if not candidates:
        return production, "production:no_clear_alternative", {"routed": False}

    _, lower, mean, expert = max(candidates, key=lambda x: (x[0], x[1], x[2], x[3]))
    return probs[expert], f"route:{expert}", {
        "routed": True,
        "expert": expert,
        "expert_mean_accuracy": mean,
        "expert_lower_accuracy": lower,
        "production_mean_accuracy": prod_mean,
        "production_disagreement_n": prod_n,
    }


def _update_state(
    state: dict[str, Any],
    probs: dict[str, np.ndarray],
    row: dict[str, Any],
) -> None:
    y = CLASSES.index(row["y"])
    prod_class = int(np.argmax(probs["production"]))
    context = _context(row["x"])

    # Only outcomes on the cases that actually create a routing decision are
    # useful for learning the decision boundary. The production expert is also
    # evaluated on those same disagreement cases.
    disagreement = any(int(np.argmax(probs[e])) != prod_class for e in ALTERNATIVES)
    if disagreement:
        for expert in EXPERTS:
            pred = int(np.argmax(probs[expert]))
            state["global"][expert][0] += int(pred == y)
            state["global"][expert][1] += 1
            state["context"][context][expert][0] += int(pred == y)
            state["context"][context][expert][1] += 1

    state["seen"] = int(state["seen"]) + 1


def _metrics(rows: list[dict[str, Any]], p: np.ndarray) -> dict[str, float | int]:
    y = [r["y"] for r in rows]
    m = metrics(y, p)
    return {
        "n": int(m["accuracy"] * len(y)) if False else len(y),
        "accuracy": float(m["accuracy"]),
        "logloss": float(m["logloss"]),
        "brier": float(m["brier"]),
        "ece": float(m["calibration_error"]),
    }


def _bootstrap_ci(values: np.ndarray, seed: int = 42) -> dict[str, float]:
    if len(values) < 2:
        v = float(values[0]) if len(values) else float("nan")
        return {"lower": v, "mean": v, "upper": v}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"lower": float(lo), "mean": float(values.mean()), "upper": float(hi)}


def run(rows: list[dict[str, Any]], horizon: str) -> tuple[dict[str, Any], dict[str, Any]]:
    dev_end = int(len(rows) * (1 - FINAL_HOLDOUT_FRAC))
    development, holdout = rows[:dev_end], rows[dev_end:]
    state = _new_state()
    block_results = []
    all_candidate = []
    all_base = []
    for test_start in range(MIN_TRAIN + CAL_BLOCK, len(development), TEST_BLOCK):
        gap = PURGE_BARS[horizon] + EMBARGO_BARS[horizon]
        fit_end = test_start - gap
        cal_start = fit_end - CAL_BLOCK
        if cal_start < MIN_TRAIN:
            continue
        test = development[test_start:min(test_start + TEST_BLOCK, len(development))]
        train = development[:cal_start]
        cal = development[cal_start:fit_end]
        if len(test) < TEST_BLOCK // 2:
            continue

        models = _fit_alternatives(train, cal)
        alt = _predict_experts(models, test)
        probs = {e: _prob(row["production"]) for e, row in zip([*([e for e in []])], [])} if False else {"production": None}
        candidate_p = []
        base_p = []
        routed_n = 0
        routed_correct = 0
        for i, row in enumerate(test):
            current = {"production": _prob(row["production"]), **{e: alt[e][i] for e in ALTERNATIVES}}
            cp, reason, trace = _route_one(state, current, row)
            candidate_p.append(cp)
            base_p.append(current["production"])
            routed_n += int(trace.get("routed", False))
            if trace.get("routed", False):
                routed_correct += int(np.argmax(cp) == CLASSES.index(row["y"]))
            _update_state(state, current, row)

        cand = _metrics(test, np.asarray(candidate_p))
        base = _metrics(test, np.asarray(base_p))
        block_results.append({
            "n": len(test),
            "routed_n": routed_n,
            "route_rate": routed_n / len(test),
            "routed_accuracy": routed_correct / routed_n if routed_n else None,
            "baseline": base,
            "candidate": cand,
            "delta": {k: float(cand[k] - base[k]) for k in ("accuracy", "logloss", "brier", "ece")},
        })
        all_candidate.extend(candidate_p)
        all_base.extend(base_p)

    if len(all_candidate) < MIN_OOS or not block_results:
        return {"status": "DEFERRED", "reason": "insufficient_oos"}, {}

    y = [r["y"] for r in development[-len(all_candidate):]]
    cand = _metrics(rows[-len(all_candidate)-len(holdout): -len(holdout)] if False else development[-len(all_candidate):], np.asarray(all_candidate))
    base = _metrics(development[-len(all_base):], np.asarray(all_base))
    deltas = {k: np.asarray([b["delta"][k] for b in block_results], dtype=float) for k in ("accuracy", "logloss", "brier", "ece")}
    development_result = {
        "status": "OK",
        "baseline": base,
        "candidate": cand,
        "delta": {k: float(cand[k] - base[k]) for k in ("accuracy", "logloss", "brier", "ece")},
        "stability": {
            "blocks": len(block_results),
            "improved_accuracy_ratio": float(np.mean(deltas["accuracy"] > 0)),
            "improved_logloss_ratio": float(np.mean(deltas["logloss"] < 0)),
            "improved_brier_ratio": float(np.mean(deltas["brier"] < 0)),
            "non_worse_accuracy_ratio": float(np.mean(deltas["accuracy"] >= -0.005)),
            "ci95_block_accuracy_delta": _bootstrap_ci(deltas["accuracy"]),
            "ci95_block_logloss_delta": _bootstrap_ci(deltas["logloss"]),
            "ci95_block_brier_delta": _bootstrap_ci(deltas["brier"]),
        },
        "route_stats": {
            "total_routed": int(sum(b["routed_n"] for b in block_results)),
            "mean_route_rate": float(np.mean([b["route_rate"] for b in block_results])),
            "mean_routed_accuracy": float(np.mean([b["routed_accuracy"] for b in block_results if b["routed_accuracy"] is not None])) if any(b["routed_accuracy"] is not None for b in block_results) else None,
        },
        "blocks": block_results,
    }

    # Carry development state into the frozen holdout; outcomes are only
    # applied after each holdout prediction.
    hold_candidate = []
    hold_base = []
    routed_n = 0
    routed_correct = 0
    for row in holdout:
        # Holdout has no newly fit alternatives: use frozen Champion plus
        # alternatives fit on development and calibration-frozen at the boundary.
        current_models = _fit_alternatives(
            development[:max(MIN_TRAIN, len(development) - (CAL_BLOCK + int(PURGE_BARS[horizon] + EMBARGO_BARS[horizon])))],
            development[-(CAL_BLOCK + int(PURGE_BARS[horizon] + EMBARGO_BARS[horizon])):-int(PURGE_BARS[horizon] + EMBARGO_BARS[horizon])] if len(development) > CAL_BLOCK + int(PURGE_BARS[horizon] + EMBARGO_BARS[horizon]) else development[-CAL_BLOCK:],
        )
        break
    # Single frozen fit for the holdout, deliberately outside the routing loop.
    gap = PURGE_BARS[horizon] + EMBARGO_BARS[horizon]
    fit_end = len(development) - gap
    cal_start = fit_end - CAL_BLOCK
    hold_train = development[:cal_start]
    hold_cal = development[cal_start:fit_end]
    if len(hold_train) < MIN_TRAIN or len(hold_cal) < 100:
        return development_result, {"status": "DEFERRED", "reason": "insufficient_holdout_fit"}
    frozen_models = _fit_alternatives(hold_train, hold_cal)
    alt_hold = _predict_experts(frozen_models, holdout)

    for i, row in enumerate(holdout):
        current = {"production": _prob(row["production"]), **{e: alt_hold[e][i] for e in ALTERNATIVES}}
        cp, reason, trace = _route_one(state, current, row)
        hold_candidate.append(cp)
        hold_base.append(current["production"])
        routed_n += int(trace.get("routed", False))
        if trace.get("routed", False):
            routed_correct += int(np.argmax(cp) == CLASSES.index(row["y"]))
        _update_state(state, current, row)

    hold_result = {
        "status": "OK",
        "baseline": _metrics(holdout, np.asarray(hold_base)),
        "candidate": _metrics(holdout, np.asarray(hold_candidate)),
        "routed_n": routed_n,
        "route_rate": routed_n / len(holdout) if holdout else 0.0,
        "routed_accuracy": routed_correct / routed_n if routed_n else None,
        "delta": {
            k: float(_metrics(holdout, np.asarray(hold_candidate))[k] - _metrics(holdout, np.asarray(hold_base))[k])
            for k in ("accuracy", "logloss", "brier", "ece")
        },
        "used_for_selection": False,
        "used_for_gate": False,
    }
    return development_result, hold_result


def evaluate(horizon: str) -> dict[str, Any]:
    raw = load_archive_research_rows(horizon, MAX_ROWS)
    if len(raw) < MIN_TRAIN + MIN_OOS + CAL_BLOCK:
        return {"status": "DEFERRED", "reason": "insufficient_archive_rows", "n": len(raw)}
    rows = _attach_production(raw, horizon)
    dev, hold = run(rows, horizon)
    return {
        "status": "OK" if dev.get("status") == "OK" else "DEFERRED",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
        "final_holdout_protected": True,
        "final_holdout_used_for_selection": False,
        "horizon": horizon,
        "n": len(rows),
        "development": dev,
        "final_holdout": hold,
        "config": {
            "test_block": TEST_BLOCK,
            "calibration_block": CAL_BLOCK,
            "context_shrink_k": CONTEXT_SHRINK_K,
            "warmup_rows": WARMUP_ROWS,
            "min_disagreement": MIN_DISAGREEMENT,
            "min_context_disagreement": MIN_CONTEXT_DISAGREEMENT,
            "min_advantage": MIN_ADVANTAGE,
            "min_posterior_lower": MIN_POSTERIOR_LOWER,
            "gap_bars": int(PURGE_BARS[horizon] + EMBARGO_BARS[horizon]),
            "experts": list(EXPERTS),
        },
        "eligibility": False,
        "policy": "production_only_unless_alternative_disagrees_and_has_causal_historical_advantage",
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "horizons": {h: evaluate(h) for h in HORIZONS},
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
