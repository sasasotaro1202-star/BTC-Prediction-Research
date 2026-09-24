"""Research-only OOS evaluator for Binance aggressive-flow/liquidation signals.

The candidate augments frozen production probabilities with only flow bins whose
end_time, last_event_time, and available_at are all at/before the prediction
cutoff. Outcome labels are used only after their target timestamps in
chronological training. Final holdout is protected and never used for selection.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from binance_flow_features import derive_flow_features
from model_compare import (
    CLASSES,
    DB,
    EMBARGO_BARS,
    metrics,
    strict_pit_provenance_reason,
    prediction_precedes_target,
)

ROOT = Path(__file__).resolve().parents[1]
FLOW_CACHE = ROOT / "data" / "binance_flow_5s.json"
OUT = ROOT / "data" / "historical_research" / "flow_meta_oos.json"

HORIZONS = ("5m", "10m")
MIN_TRAIN = 60
TEST_BLOCK = 20
FINAL_HOLDOUT_FRAC = 0.20
MIN_BLOCKS = 8
MIN_HOLDOUT = 20
EPS = 1e-7

FLOW_KEYS = (
    "flow_15s_signed_qty",
    "flow_15s_signed_notional",
    "flow_15s_buy_share",
    "flow_15s_trade_count",
    "flow_15s_avg_trade_notional",
    "flow_15s_max_trade_notional",
    "flow_15s_liquidation_count",
    "flow_15s_liquidation_signed_notional",
    "flow_15s_liquidation_notional",
    "flow_15s_liquidation_to_trade_notional",
    "flow_15s_missing",
    "flow_30s_signed_qty",
    "flow_30s_signed_notional",
    "flow_30s_buy_share",
    "flow_30s_trade_count",
    "flow_30s_avg_trade_notional",
    "flow_30s_max_trade_notional",
    "flow_30s_liquidation_count",
    "flow_30s_liquidation_signed_notional",
    "flow_30s_liquidation_notional",
    "flow_30s_liquidation_to_trade_notional",
    "flow_30s_missing",
    "flow_60s_signed_qty",
    "flow_60s_signed_notional",
    "flow_60s_buy_share",
    "flow_60s_trade_count",
    "flow_60s_avg_trade_notional",
    "flow_60s_max_trade_notional",
    "flow_60s_liquidation_count",
    "flow_60s_liquidation_signed_notional",
    "flow_60s_liquidation_notional",
    "flow_60s_liquidation_to_trade_notional",
    "flow_60s_missing",
    "flow_300s_signed_qty",
    "flow_300s_signed_notional",
    "flow_300s_buy_share",
    "flow_300s_trade_count",
    "flow_300s_avg_trade_notional",
    "flow_300s_max_trade_notional",
    "flow_300s_liquidation_count",
    "flow_300s_liquidation_signed_notional",
    "flow_300s_liquidation_notional",
    "flow_300s_liquidation_to_trade_notional",
    "flow_300s_missing",
    "flow_30s_vs_5m_signed_notional",
    "flow_60s_liquidation_shock",
    "flow_any_available",
)


def _utc(value: Any) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt.astimezone(timezone.utc) if dt.tzinfo else None


def _safe_json(value: Any) -> dict[str, Any]:
    try:
        obj = json.loads(value) if isinstance(value, str) else value
        return obj if isinstance(obj, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _norm(p: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(p, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.shape != (3,) or not np.isfinite(arr).all() or np.any(arr < 0) or float(arr.sum()) <= 0:
        return None
    arr = np.clip(arr, EPS, 1.0)
    return arr / arr.sum()


def _entropy(p: np.ndarray) -> float:
    return float(-np.sum(p * np.log(np.clip(p, EPS, 1.0))) / math.log(3.0))


def _margin(p: np.ndarray) -> float:
    ordered = np.sort(p)[::-1]
    return float(ordered[0] - ordered[1])


def _load_flow_rows() -> list[dict[str, Any]]:
    if not FLOW_CACHE.is_file() or FLOW_CACHE.stat().st_size <= 0:
        return []
    try:
        obj = json.loads(FLOW_CACHE.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return []
    if (
        obj.get("schema_version") != 1
        or obj.get("source") != "Binance USD-M Futures WebSocket"
        or obj.get("streams") != ["btcusdt@aggTrade", "btcusdt@forceOrder"]
        or obj.get("bin_ms") != 5000
    ):
        return []
    rows = obj.get("rows")
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            start = int(row["start_time_ms"])
            end = int(row["end_time_ms"])
            available = int(row["available_at_ms"])
            event = int(row["last_event_time_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        if end != start + 5000 or available < event or end <= 0:
            continue
        numeric = (
            "buy_qty", "sell_qty", "buy_notional", "sell_notional",
            "max_trade_notional", "liquidation_buy_notional", "liquidation_sell_notional",
        )
        if not all(math.isfinite(float(row.get(k, 0.0))) for k in numeric):
            continue
        out.append(dict(row))
    return sorted(out, key=lambda r: int(r["end_time_ms"]))


def _load_predictions(horizon: str, flow_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actual = f"actual_direction_{horizon}"
    target = f"target_{horizon}"
    pu = f"p_up_{horizon}"
    pd = f"p_down_{horizon}"
    pf = f"p_flat_{horizon}"
    with sqlite3.connect(DB) as con:
        raw = con.execute(
            f"""SELECT prediction_id, created_at_utc, {target}, {actual},
                       {pu}, {pd}, {pf}, scenario_json, model_version
                FROM predictions
                WHERE {actual} IS NOT NULL
                ORDER BY created_at_utc, prediction_id"""
        ).fetchall()

    out: list[dict[str, Any]] = []
    for rid, created, target_at, y, up, down, flat, scenario_json, model_version in raw:
        if y not in CLASSES or not prediction_precedes_target(created, target_at):
            continue
        scenario = _safe_json(scenario_json)
        if scenario.get("production_mode") != "binance_primary":
            continue
        if strict_pit_provenance_reason(scenario, created) is not None:
            continue
        cutoff = _utc(created)
        target_dt = _utc(target_at)
        if cutoff is None or target_dt is None:
            continue
        p = _norm([down, flat, up])
        if p is None:
            continue
        flow = derive_flow_features(flow_rows, int(cutoff.timestamp() * 1000))
        if float(flow.get("flow_any_available", 0.0)) <= 0.0:
            continue

        # Aggregate only the causal flow features; no current/future label is used.
        flow_values = []
        missing = False
        for key in FLOW_KEYS:
            try:
                value = float(flow.get(key, 0.0))
            except (TypeError, ValueError):
                missing = True
                break
            if not math.isfinite(value):
                missing = True
                break
            flow_values.append(value)
        if missing:
            continue

        out.append(
            {
                "id": rid,
                "created": cutoff.isoformat(),
                "target": target_dt.isoformat(),
                "y": y,
                "model_version": str(model_version or ""),
                "base": p.tolist(),
                "flow": flow_values,
                "context": {
                    "entropy": _entropy(p),
                    "margin": _margin(p),
                    "direction": CLASSES[int(np.argmax(p))],
                },
            }
        )
    return out


def _vector(row: dict[str, Any]) -> np.ndarray:
    p = np.asarray(row["base"], dtype=float)
    flow = np.asarray(row["flow"], dtype=float)
    extra = np.asarray(
        [row["context"]["entropy"], row["context"]["margin"]],
        dtype=float,
    )
    x = np.concatenate([p, extra, flow])
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("flow_vector_nonfinite")
    return x


def _factories():
    return {
        "logreg_flow_meta": lambda: Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.3, max_iter=2500)),
            ]
        ),
        "extra_trees_flow_meta": lambda: ExtraTreesClassifier(
            n_estimators=260,
            max_depth=8,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "hgb_flow_meta": lambda: HistGradientBoostingClassifier(
            max_iter=180,
            max_leaf_nodes=15,
            learning_rate=0.04,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


def _causal_train(rows: list[dict[str, Any]], test_start: str, horizon: str) -> list[dict[str, Any]]:
    cutoff = _utc(test_start)
    if cutoff is None:
        return []
    # Target settlement must precede the test period by the same explicit
    # embargo used elsewhere in the BTC OOS stack.
    embargo_cutoff = cutoff - timedelta(minutes=int(EMBARGO_BARS[horizon]))
    return [
        row
        for row in rows
        if (created := _utc(row["created"])) is not None
        and (target := _utc(row["target"])) is not None
        and created < target
        and target < embargo_cutoff
    ]


def _metrics_safe(rows: list[dict[str, Any]], probs: np.ndarray) -> dict[str, Any]:
    if len(rows) == 0:
        return {"n": 0}
    m = metrics([r["y"] for r in rows], probs)
    return {
        "n": len(rows),
        "accuracy": float(m["accuracy"]),
        "logloss": float(m["logloss"]),
        "brier": float(m["brier"]),
        "ece": float(m["calibration_error"]),
    }


def _evaluate_model(rows: list[dict[str, Any]], factory, horizon: str) -> dict[str, Any]:
    if len(rows) < MIN_TRAIN + TEST_BLOCK:
        return {"status": "DEFERRED", "reason": "insufficient_flow_covered_rows", "n": len(rows)}
    split = int(len(rows) * (1.0 - FINAL_HOLDOUT_FRAC))
    development = rows[:split]
    holdout = rows[split:]
    blocks = []
    for end in range(MIN_TRAIN, len(development), TEST_BLOCK):
        test = development[end : min(end + TEST_BLOCK, len(development))]
        if len(test) < max(10, TEST_BLOCK // 2):
            continue
        train = _causal_train(development[:end], test[0]["created"], horizon)
        if len(train) < MIN_TRAIN:
            continue
        model = factory()
        model.fit(
            np.stack([_vector(r) for r in train]),
            np.asarray([r["y"] for r in train]),
        )
        probs = model.predict_proba(np.stack([_vector(r) for r in test]))
        aligned = np.full((len(test), 3), EPS, dtype=float)
        for j, cls in enumerate(model.classes_):
            if str(cls) in CLASSES:
                aligned[:, CLASSES.index(str(cls))] = probs[:, j]
        aligned = aligned / aligned.sum(axis=1, keepdims=True)
        blocks.append(
            {
                "n": len(test),
                "candidate": _metrics_safe(test, aligned),
                "baseline": _metrics_safe(
                    test,
                    np.asarray([r["base"] for r in test], dtype=float),
                ),
            }
        )
    if len(blocks) < MIN_BLOCKS or len(holdout) < MIN_HOLDOUT:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_oos_blocks_or_holdout",
            "blocks": len(blocks),
            "holdout_n": len(holdout),
            "n": len(rows),
        }

    cand_ll = np.asarray([b["candidate"]["logloss"] - b["baseline"]["logloss"] for b in blocks])
    cand_br = np.asarray([b["candidate"]["brier"] - b["baseline"]["brier"] for b in blocks])
    cand_ac = np.asarray([b["candidate"]["accuracy"] - b["baseline"]["accuracy"] for b in blocks])
    dev_candidate = {
        "accuracy": float(np.mean([b["candidate"]["accuracy"] for b in blocks])),
        "logloss": float(np.mean([b["candidate"]["logloss"] for b in blocks])),
        "brier": float(np.mean([b["candidate"]["brier"] for b in blocks])),
        "ece": float(np.mean([b["candidate"]["ece"] for b in blocks])),
    }
    dev_baseline = {
        "accuracy": float(np.mean([b["baseline"]["accuracy"] for b in blocks])),
        "logloss": float(np.mean([b["baseline"]["logloss"] for b in blocks])),
        "brier": float(np.mean([b["baseline"]["brier"] for b in blocks])),
        "ece": float(np.mean([b["baseline"]["ece"] for b in blocks])),
    }

    model = factory()
    model.fit(
        np.stack([_vector(r) for r in development]),
        np.asarray([r["y"] for r in development]),
    )
    hp = model.predict_proba(np.stack([_vector(r) for r in holdout]))
    aligned_hp = np.full((len(holdout), 3), EPS, dtype=float)
    for j, cls in enumerate(model.classes_):
        if str(cls) in CLASSES:
            aligned_hp[:, CLASSES.index(str(cls))] = hp[:, j]
    aligned_hp /= aligned_hp.sum(axis=1, keepdims=True)

    base_holdout = np.asarray([r["base"] for r in holdout], dtype=float)
    holdout_candidate = _metrics_safe(holdout, aligned_hp)
    holdout_baseline = _metrics_safe(holdout, base_holdout)

    # Candidate gate is descriptive only. It cannot change production.
    eligible_dev = bool(
        dev_candidate["logloss"] <= dev_baseline["logloss"] - 0.003
        and dev_candidate["brier"] <= dev_baseline["brier"] - 0.0015
        and dev_candidate["accuracy"] >= dev_baseline["accuracy"] - 0.005
        and float(np.mean(cand_ll < 0.0)) >= 0.70
        and float(np.mean(cand_br < 0.0)) >= 0.70
        and float(np.mean(cand_ac >= -0.005)) >= 0.70
    )
    holdout_ok = bool(
        holdout_candidate["logloss"] <= holdout_baseline["logloss"]
        and holdout_candidate["brier"] <= holdout_baseline["brier"]
        and holdout_candidate["accuracy"] >= holdout_baseline["accuracy"] - 0.005
        and holdout_candidate["ece"] <= holdout_baseline["ece"] + 0.01
    )

    # High-uncertainty subgroup diagnostics: entropy/margin are prediction-time only.
    uncertain = [r for r in holdout if r["context"]["entropy"] >= 0.90 or r["context"]["margin"] <= 0.05]
    high_uncertainty = {}
    if len(uncertain) >= 20:
        idx = {r["id"]: i for i, r in enumerate(holdout)}
        sel = np.asarray([idx[r["id"]] for r in uncertain], dtype=int)
        high_uncertainty = {
            "n": len(uncertain),
            "candidate": _metrics_safe(uncertain, aligned_hp[sel]),
            "baseline": _metrics_safe(uncertain, base_holdout[sel]),
        }

    return {
        "status": "OK",
        "n": len(rows),
        "development_n": len(development),
        "holdout_n": len(holdout),
        "blocks": blocks,
        "development_summary": {
            "candidate": dev_candidate,
            "baseline": dev_baseline,
            "improved_logloss_ratio": float(np.mean(cand_ll < 0.0)),
            "improved_brier_ratio": float(np.mean(cand_br < 0.0)),
            "non_worse_accuracy_ratio": float(np.mean(cand_ac >= -0.005)),
        },
        "final_holdout": {
            "protected": True,
            "candidate": holdout_candidate,
            "baseline": holdout_baseline,
            "meets_holdout_guard": holdout_ok,
        },
        "high_uncertainty_holdout": high_uncertainty,
        "eligible_pending_policy": bool(eligible_dev and holdout_ok),
        "promotion_allowed": False,
        "research_only": True,
        "production_changed": False,
        "pit_policy": "flow_end_event_available_at_all_le_prediction_cutoff; target_settlement_purged_and_embargoed",
    }


def evaluate(horizon: str, flow_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = _load_predictions(horizon, flow_rows)
    if not rows:
        return {
            "status": "DEFERRED",
            "reason": "no_strict_pit_predictions_with_flow_coverage",
            "n": 0,
        }
    results = {}
    for name, factory in _factories().items():
        results[name] = _evaluate_model(rows, factory, horizon)
    usable = [v for v in results.values() if v.get("status") == "OK"]
    return {
        "status": "OK" if usable else "DEFERRED",
        "research_only": True,
        "production_changed": False,
        "strict_pit": True,
        "flow_rows": len(flow_rows),
        "flow_cache_end_time_ms": max(int(r["end_time_ms"]) for r in flow_rows) if flow_rows else None,
        "horizon": horizon,
        "models": results,
    }


def main() -> None:
    flow_rows = _load_flow_rows()
    if not DB.exists():
        raise SystemExit("prediction database missing")
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "flow_cache_present": bool(flow_rows),
        "horizons": {h: evaluate(h, flow_rows) for h in HORIZONS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
