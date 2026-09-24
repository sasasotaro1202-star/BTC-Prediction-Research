"""Research-only online expert aggregation for BTC short-horizon predictions.

The candidate never uses the current row's realized outcome to set its weight.
Weights depend only on previously settled, strict-PIT observations. This is
an adaptive ensemble diagnostic, not a production promotion path.
"""
from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from model_compare import (
    CLASSES,
    DB,
    dedupe_exact_prediction_events,
    prediction_precedes_target,
    strict_pit_provenance_reason,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "online_expert_oos.json"

HORIZONS = ("5m", "10m")
EXPERTS = ("model_raw", "structural", "fused_raw", "calibrated")
WARMUP = 40
TEST_BLOCK = 25
HOLDOUT_FRACTION = 0.10
ALPHA = 0.15
ETA = 4.0
SHRINK_K = 20.0
EPS = 1e-8


def _safe_json(value: Any) -> dict[str, Any]:
    try:
        obj = json.loads(value) if isinstance(value, str) else value
        return obj if isinstance(obj, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _utc(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _prob_vector(value: Any) -> np.ndarray | None:
    if not isinstance(value, dict):
        return None
    p = np.asarray([value.get(c, np.nan) for c in CLASSES], dtype=float)
    if not np.isfinite(p).all() or np.any(p < 0) or float(p.sum()) <= 0:
        return None
    p = np.clip(p, EPS, 1.0)
    return p / p.sum()


def _context_key(situation: dict[str, Any]) -> str:
    return "|".join(
        str(situation.get(key, "UNKNOWN"))
        for key in ("trend_state", "volatility_state", "orderflow_state", "signal_quality")
    )


def _load_rows(horizon: str) -> list[dict[str, Any]]:
    actual = f"actual_direction_{horizon}"
    target = f"target_{horizon}"
    with sqlite3.connect(DB) as con:
        raw = con.execute(
            f"""SELECT prediction_id, created_at_utc, {target}, {actual},
                       model_version, p_up_{horizon}, p_down_{horizon},
                       p_flat_{horizon}, scenario_json
                FROM predictions
                WHERE {actual} IS NOT NULL
                ORDER BY created_at_utc, prediction_id"""
        ).fetchall()

    rows: list[dict[str, Any]] = []
    for r in raw:
        created, target_at, y = r[1], r[2], r[3]
        if not prediction_precedes_target(created, target_at):
            continue
        if y not in CLASSES:
            continue
        scenario = _safe_json(r[8])
        if scenario.get("production_mode") != "binance_primary":
            continue
        if strict_pit_provenance_reason(scenario, created) is not None:
            continue
        components = scenario.get("components")
        situation = scenario.get("situation")
        if not isinstance(components, dict) or not isinstance(situation, dict):
            continue

        experts: dict[str, np.ndarray] = {}
        for expert in EXPERTS:
            experts[expert] = _prob_vector(components.get(f"{expert}_{horizon}"))
            if experts[expert] is None:
                break
        if len(experts) != len(EXPERTS) or any(value is None for value in experts.values()):
            continue

        row = {
            "id": r[0],
            "created": created,
            "target": target_at,
            "y": y,
            "model_version": r[4],
            "experts": experts,
            "context": _context_key(situation),
        }
        rows.append(row)

    # Use the same canonical exact-event deduplication as the existing OOS stack.
    canonical = []
    for row in rows:
        canonical.append(
            {
                "id": row["id"],
                "created": row["created"],
                "target": row["target"],
                "model_version": row["model_version"],
                "x": [],
                "production": row["experts"]["calibrated"].tolist(),
                "_online": row,
            }
        )
    deduped = dedupe_exact_prediction_events(canonical)
    return [item["_online"] for item in deduped]


def _loss(p: np.ndarray, y: str) -> float:
    index = CLASSES.index(y)
    return float(-math.log(max(float(p[index]), EPS)))



def _metrics(rows: list[dict[str, Any]], probs: np.ndarray) -> dict[str, float]:
    if not rows or probs.shape != (len(rows), 3):
        raise ValueError("metric_shape_invalid")
    labels = np.asarray([CLASSES.index(row["y"]) for row in rows], dtype=int)
    picked = np.clip(probs[np.arange(len(rows)), labels], EPS, 1.0)
    confidence = probs.max(axis=1)
    accuracy = float(np.mean(np.argmax(probs, axis=1) == labels))
    logloss = float(-np.mean(np.log(picked)))
    brier = float(np.mean(np.sum((probs - np.eye(3)[labels]) ** 2, axis=1)))

    ece = 0.0
    for lo in np.linspace(0.0, 0.9, 10):
        hi = lo + 0.1
        mask = (confidence >= lo) & (confidence < hi if hi < 1.0 else confidence <= hi)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(
            float(np.mean(np.argmax(probs[mask], axis=1) == labels[mask]))
            - float(np.mean(confidence[mask]))
        )
    return {"n": int(len(rows)), "accuracy": accuracy, "logloss": logloss, "brier": brier, "ece": ece}


def _weights(global_ema: dict[str, float], state_ema: dict[str, float], state_count: int) -> np.ndarray:
    estimates = []
    for expert in EXPERTS:
        g = global_ema[expert]
        if state_count <= 0:
            local = g
        else:
            local = (state_count / (state_count + SHRINK_K)) * state_ema[expert] + (
                SHRINK_K / (state_count + SHRINK_K)
            ) * g
        estimates.append(local)
    scores = -ETA * np.asarray(estimates, dtype=float)
    scores -= scores.max()
    w = np.exp(scores)
    w /= w.sum()
    return w


def _new_state() -> dict[str, Any]:
    return {
        "global_ema": {expert: 0.0 for expert in EXPERTS},
        "state_ema": defaultdict(lambda: {expert: 0.0 for expert in EXPERTS}),
        "state_counts": defaultdict(int),
        "seen": 0,
    }


def _apply_outcome(state: dict[str, Any], row: dict[str, Any]) -> None:
    context = row["context"]
    state["state_ema"][context]  # initialize lazily
    for expert in EXPERTS:
        loss = _loss(row["experts"][expert], row["y"])
        state["global_ema"][expert] = (
            (1.0 - ALPHA) * state["global_ema"][expert] + ALPHA * loss
        )
        state["state_ema"][context][expert] = (
            (1.0 - ALPHA) * state["state_ema"][context][expert] + ALPHA * loss
        )
    state["state_counts"][context] += 1


def run_strategy(
    rows: list[dict[str, Any]],
    strategy: str,
    state: dict[str, Any] | None = None,
    pending: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    state = _new_state() if state is None else state
    pending = [] if pending is None else list(pending)
    predictions: list[np.ndarray] = []
    trace: list[dict[str, Any]] = []

    for row in rows:
        created = _utc(row["created"])
        if created is None:
            raise ValueError("row_created_timestamp_invalid")

        # Release only outcomes whose targets were strictly before the
        # current prediction timestamp. This prevents overlap leakage, e.g.
        # a 10m prediction made at t+5m cannot learn from the t prediction's
        # target at t+10m because that result is not settled yet.
        still_pending = []
        for previous in pending:
            target = _utc(previous["target"])
            if target is None:
                continue
            if target < created:
                _apply_outcome(state, previous)
            else:
                still_pending.append(previous)
        pending = still_pending

        seen = int(state["seen"])
        global_ema = state["global_ema"]
        state_ema = state["state_ema"]
        state_counts = state["state_counts"]

        if seen < WARMUP:
            weights = np.full(len(EXPERTS), 1.0 / len(EXPERTS))
        elif strategy == "static_equal":
            weights = np.full(len(EXPERTS), 1.0 / len(EXPERTS))
        elif strategy == "online_ewma":
            weights = _weights(global_ema, global_ema, 0)
        elif strategy == "online_context":
            context = row["context"]
            weights = _weights(global_ema, state_ema[context], state_counts[context])
        else:
            raise ValueError(f"unknown_strategy:{strategy}")

        matrix = np.stack([row["experts"][expert] for expert in EXPERTS], axis=0)
        p = np.sum(weights[:, None] * matrix, axis=0)
        p = np.clip(p, EPS, 1.0)
        p /= p.sum()
        predictions.append(p)
        trace.append(
            {
                "created": row["created"],
                "weights": {expert: float(weight) for expert, weight in zip(EXPERTS, weights)},
            }
        )

        # Current outcome is not incorporated now. It becomes eligible only
        # when its target timestamp is strictly earlier than a later prediction.
        pending.append(row)
        state["seen"] = seen + 1

    return np.stack(predictions), trace, state, pending


def _split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    holdout = max(100, int(round(len(rows) * HOLDOUT_FRACTION)))
    return rows[:-holdout], rows[-holdout:]


def evaluate(horizon: str) -> dict[str, Any]:
    rows = _load_rows(horizon)
    if len(rows) < WARMUP + 100:
        return {
            "status": "DEFERRED",
            "research_only": True,
            "production_changed": False,
            "n": len(rows),
            "reason": "insufficient_strict_pit_primary_rows",
        }

    development, holdout = _split(rows)
    strategies = ("static_equal", "online_ewma", "online_context")
    dev_results = {}
    holdout_results = {}
    traces = {}

    for strategy in strategies:
        dev_probs, dev_trace, state, pending = run_strategy(development, strategy)
        hold_probs, _, _, _ = run_strategy(
            holdout, strategy, state=state, pending=pending
        )
        dev_results[strategy] = _metrics(development, dev_probs)
        holdout_results[strategy] = _metrics(holdout, hold_probs)
        traces[strategy] = dev_trace[-1]

    baseline = dev_results["online_ewma"]
    candidate = dev_results["online_context"]
    eligible = bool(
        candidate["logloss"] <= baseline["logloss"] - 0.003
        and candidate["brier"] <= baseline["brier"] - 0.001
        and candidate["accuracy"] >= baseline["accuracy"] - 0.005
    )

    # Frozen holdout is descriptive only. It cannot change eligibility.
    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "final_holdout_protected": True,
        "final_holdout_used_for_selection": False,
        "promotion_evidence_eligible": False,
        "config": {
            "warmup": WARMUP,
            "ewma_alpha": ALPHA,
            "eta": ETA,
            "context_shrink_k": SHRINK_K,
            "experts": list(EXPERTS),
        },
        "n": len(rows),
        "development_n": len(development),
        "final_holdout_n": len(holdout),
        "chronological": True,
        "strict_pit_primary_only": True,
        "strategies": dev_results,
        "final_holdout": holdout_results,
        "delta_context_vs_ewma": {
            key: float(candidate[key] - baseline[key])
            for key in ("accuracy", "logloss", "brier", "ece")
        },
        "eligibility": eligible,
        "policy": (
            "weights_use_only_previously_settled_rows; "
            "context_weights_shrink_to_global; final_holdout_descriptive_only"
        ),
        "trace_last_dev": traces,
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
