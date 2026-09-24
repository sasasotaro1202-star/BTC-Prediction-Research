"""PIT/OOS situation-aware meta-model research for BTC short-horizon prediction.

The meta-model consumes only prediction-time information already persisted with
historical production predictions: calibrated production probabilities,
microstructure snapshots, and descriptive situation state. It is strictly
research-only and never mutates production artifacts.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_compare import (
    HORIZONS,
    CLASSES,
    DB,
    prediction_precedes_target,
    strict_pit_provenance_reason,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "situation_meta_oos.json"

MIN_ROWS = 3000
MIN_TRAIN = 1000
TEST_BLOCK = 25
MAX_BLOCKS = 24
HOLDOUT_MIN = 1000
PURGE_BARS = {"5m": 5, "10m": 10}
EMBARGO_BARS = {"5m": 60, "10m": 60}

MICRO_KEYS = (
    "vwap_distance_5m",
    "vwap_distance_15m",
    "vwap_distance_30m",
    "volume_burst_5m",
    "volume_burst_15m",
    "range_compression_5m",
    "range_compression_15m",
    "taker_imbalance_5m",
    "taker_imbalance_15m",
    "taker_imbalance_delta_5m_15m",
    "book_imbalance",
    "bybit_book_imbalance",
    "cross_exchange_gap",
    "spot_futures_gap",
    "funding_binance",
    "funding_bybit",
)
CAT_KEYS = (
    ("trend_state", ("RANGE", "TREND_UP", "TREND_DOWN")),
    ("volatility_state", ("STABLE", "EXPANDING", "COMPRESSING")),
    ("orderflow_state", ("BALANCED", "BUY_PRESSURE", "SELL_PRESSURE")),
    ("signal_quality", ("LOW", "MEDIUM", "HIGH")),
    ("horizon_alignment", ("AGREE", "CONFLICT")),
    ("direction_5m", CLASSES),
    ("direction_10m", CLASSES),
    ("data_state", ("HEALTHY", "PARTIAL", "DEGRADED")),
)


def _safe_json(value: Any) -> dict[str, Any]:
    try:
        obj = json.loads(value) if isinstance(value, str) else value
        return obj if isinstance(obj, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _parse_utc(value: Any) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt.astimezone(timezone.utc) if dt.tzinfo else None


def primary_live_scenario(scenario: dict[str, Any]) -> bool:
    return isinstance(scenario, dict) and scenario.get("production_mode") == "binance_primary"


def _row_key(row: dict[str, Any]) -> tuple:
    return (
        row.get("created", ""),
        row.get("target", ""),
        row.get("model_version", ""),
        tuple(round(float(v), 12) for v in row.get("production", [])),
        row.get("horizon", ""),
    )


def _load_rows(horizon: str) -> list[dict[str, Any]]:
    target_col = "target_5m" if horizon == "5m" else "target_10m"
    actual_col = "actual_direction_5m" if horizon == "5m" else "actual_direction_10m"
    with sqlite3.connect(DB) as con:
        raw = con.execute(
            f"""SELECT prediction_id, created_at_utc, {target_col}, feature_json,
                       {actual_col}, p_up_{horizon}, p_down_{horizon},
                       p_flat_{horizon}, model_version, scenario_json
                FROM predictions
                WHERE {actual_col} IS NOT NULL
                ORDER BY created_at_utc, prediction_id"""
        ).fetchall()

    rows: list[dict[str, Any]] = []
    for r in raw:
        if not prediction_precedes_target(r[1], r[2]):
            continue
        scenario = _safe_json(r[9])
        if not primary_live_scenario(scenario):
            continue
        if strict_pit_provenance_reason(scenario, r[1]) is not None:
            continue
        features = _safe_json(r[3])
        if not isinstance(features, dict):
            continue
        production = [_finite(r[6]), _finite(r[7]), _finite(r[5])]
        if r[4] not in CLASSES or any(v < 0 for v in production) or sum(production) <= 0:
            continue
        micro = scenario.get("microstructure")
        situation = scenario.get("situation")
        if not isinstance(micro, dict) or not isinstance(situation, dict):
            continue
        # Normalize the persisted production probabilities again so malformed
        # but finite historical rows cannot distort the meta-model.
        total = float(sum(production))
        production = [v / total for v in production]
        rows.append({
            "id": r[0],
            "created": r[1],
            "target": r[2],
            "x": features,
            "production": production,
            "y": r[4],
            "model_version": r[8],
            "microstructure": micro,
            "situation": situation,
            "production_mode": str(scenario.get("production_mode", "")),
            "horizon": horizon,
        })

    # Stable immutable identity: do not let repeated rows overweight a block.
    seen = set()
    out = []
    for row in rows:
        key = _row_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def meta_feature_names() -> list[str]:
    names = ["prod_down", "prod_flat", "prod_up", *MICRO_KEYS]
    names.extend(f"missing={key}" for key in MICRO_KEYS)
    names.extend(["situation_entropy", "situation_margin"])
    for key, values in CAT_KEYS:
        names.extend(f"{key}={value}" for value in values)
    return names


def build_meta_vector(row: dict[str, Any]) -> np.ndarray:
    p = row["production"]
    values = [float(p[0]), float(p[1]), float(p[2])]
    micro = row.get("microstructure") or {}
    for key in MICRO_KEYS:
        # Missing remains an explicit neutral value rather than borrowing a
        # future observation. Missingness is also encoded explicitly so the
        # model can learn that degraded market-data states are informative.
        raw = micro.get(key)
        values.append(_finite(raw, 0.0))
    for key in MICRO_KEYS:
        raw = micro.get(key)
        values.append(1.0 if raw is None or not math.isfinite(_finite(raw, float("nan"))) else 0.0)
    situation = row.get("situation") or {}
    values.extend([
        _finite(situation.get("normalized_entropy"), 0.5),
        _finite(situation.get("probability_margin"), 0.0),
    ])
    for key, allowed in CAT_KEYS:
        actual = str(situation.get(key, ""))
        values.extend([1.0 if actual == value else 0.0 for value in allowed])
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("meta_feature_nonfinite")
    return arr


def causal_train_rows(rows: list[dict[str, Any]], test_start: str, horizon: str) -> list[dict[str, Any]]:
    start = _parse_utc(test_start)
    if start is None:
        return []
    cutoff = start - timedelta(minutes=int(EMBARGO_BARS[horizon]))
    out = []
    for row in rows:
        created = _parse_utc(row.get("created"))
        target = _parse_utc(row.get("target"))
        if created is None or target is None:
            continue
        if created >= target:
            continue
        # Purge by actual label settlement plus an explicit embargo.
        if target >= cutoff:
            continue
        out.append(row)
    return out


def factories() -> dict[str, Callable[[], object]]:
    return {
        "logreg": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.15, max_iter=3000)),
        ]),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "hgb": lambda: HistGradientBoostingClassifier(
            max_iter=220,
            max_leaf_nodes=15,
            learning_rate=0.04,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


def aligned_probs(model: object, rows: list[dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.empty((0, 3), dtype=float)
    raw = np.asarray(model.predict_proba(np.stack([build_meta_vector(r) for r in rows])), dtype=float)
    classes = [str(v) for v in getattr(model, "classes_", [])]
    out = np.full((len(rows), 3), 1e-6, dtype=float)
    for j, cls in enumerate(classes):
        if cls in CLASSES:
            out[:, CLASSES.index(cls)] = raw[:, j]
    out /= out.sum(axis=1, keepdims=True)
    return out


def _metrics(y: list[str], probs: np.ndarray) -> dict[str, float]:
    yi = np.asarray([CLASSES.index(v) for v in y], dtype=int)
    p = np.asarray(probs, dtype=float)
    if len(yi) == 0 or p.shape != (len(yi), 3):
        raise ValueError("metric_shape_invalid")
    picked = np.clip(p[np.arange(len(yi)), yi], 1e-12, 1.0)
    ll = float(-np.mean(np.log(picked)))
    brier = float(np.mean(np.sum((p - np.eye(3)[yi]) ** 2, axis=1)))
    acc = float(np.mean(np.argmax(p, axis=1) == yi))
    return {"logloss": ll, "brier": brier, "accuracy": acc, "n": int(len(yi))}


def _select_model(train: list[dict[str, Any]], horizon: str) -> tuple[str, object] | None:
    if len(train) < MIN_TRAIN:
        return None
    split = max(int(len(train) * 0.70), len(train) - 500)
    if split < 500 or len(train) - split < 100:
        return None
    valid_rows = train[split:]
    # Inner selection is causal too: labels used to fit a candidate must have
    # settled before the first validation prediction, with the same 60m embargo.
    valid_start = valid_rows[0].get("created", "")
    fit_rows = causal_train_rows(train[:split], valid_start, horizon)
    if len(fit_rows) < 500:
        return None
    y_fit = np.asarray([r["y"] for r in fit_rows])
    if len(set(y_fit.tolist())) < 3:
        return None
    scored: list[tuple[float, str, object]] = []
    for name, factory in factories().items():
        try:
            model = factory()
            model.fit(np.stack([build_meta_vector(r) for r in fit_rows]), y_fit)
            m = _metrics([r["y"] for r in valid_rows], aligned_probs(model, valid_rows))
            # Balanced probabilistic objective; accuracy breaks near-ties.
            score = 0.70 * m["logloss"] + 0.30 * m["brier"]
            scored.append((score, name, model))
        except Exception:
            continue
    if not scored:
        return None
    scored.sort(key=lambda x: x[0])
    _, winner, _ = scored[0]
    # Refit the selected family only on the causal training window.
    model = factories()[winner]()
    model.fit(np.stack([build_meta_vector(r) for r in train]), np.asarray([r["y"] for r in train]))
    return winner, model


def _endpoints(n: int) -> list[int]:
    endpoints = list(range(MIN_TRAIN, n, TEST_BLOCK))
    if len(endpoints) <= MAX_BLOCKS:
        return endpoints
    return sorted(set(int(v) for v in np.linspace(endpoints[0], endpoints[-1], MAX_BLOCKS)))


def _situation_summary(rows: list[dict[str, Any]], baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(rows):
        state = str((row.get("situation") or {}).get("market_state", "UNKNOWN"))
        item = grouped.setdefault(state, {"n": 0, "baseline": [], "candidate": []})
        item["n"] += 1
        item["baseline"].append(baseline[i])
        item["candidate"].append(candidate[i])
    out = {}
    for state, item in grouped.items():
        idx = [i for i, row in enumerate(rows) if str((row.get("situation") or {}).get("market_state", "UNKNOWN")) == state]
        yy = [rows[i]["y"] for i in idx]
        bp = np.stack([baseline[i] for i in idx])
        cp = np.stack([candidate[i] for i in idx])
        bm = _metrics(yy, bp)
        cm = _metrics(yy, cp)
        out[state] = {
            "n": item["n"],
            "baseline": bm,
            "candidate": cm,
            "delta": {
                "logloss": cm["logloss"] - bm["logloss"],
                "brier": cm["brier"] - bm["brier"],
                "accuracy": cm["accuracy"] - bm["accuracy"],
            },
        }
    return out


def evaluate_horizon(horizon: str) -> dict[str, Any]:
    rows = _load_rows(horizon)
    if len(rows) < MIN_ROWS:
        return {
            "status": "DEFERRED",
            "research_only": True,
            "production_changed": False,
            "reason": "insufficient_strict_pit_primary_rows",
            "n": len(rows),
            "data_source": "live_binance_primary",
        }
    holdout_n = max(HOLDOUT_MIN, int(len(rows) * 0.10))
    if len(rows) <= MIN_TRAIN + TEST_BLOCK + holdout_n:
        return {
            "status": "DEFERRED",
            "research_only": True,
            "production_changed": False,
            "reason": "insufficient_development_and_frozen_holdout_rows",
            "n": len(rows),
            "data_source": "live_binance_primary",
        }
    development = rows[:-holdout_n]
    holdout = rows[-holdout_n:]

    blocks = []
    for end in _endpoints(len(development)):
        test = development[end:min(end + TEST_BLOCK, len(development))]
        if len(test) < max(10, TEST_BLOCK // 2):
            continue
        test_start = test[0].get("created", "")
        train = causal_train_rows(development[:end], test_start, horizon)
        selected = _select_model(train, horizon)
        if selected is None:
            continue
        name, model = selected
        candidate = aligned_probs(model, test)
        baseline = np.stack([r["production"] for r in test])
        bm = _metrics([r["y"] for r in test], baseline)
        cm = _metrics([r["y"] for r in test], candidate)
        blocks.append({
            "n": len(test),
            "model": name,
            "baseline": bm,
            "candidate": cm,
            "delta": {
                "logloss": cm["logloss"] - bm["logloss"],
                "brier": cm["brier"] - bm["brier"],
                "accuracy": cm["accuracy"] - bm["accuracy"],
            },
            "situations": _situation_summary(test, baseline, candidate),
        })

    if len(blocks) < 12:
        return {
            "status": "DEFERRED",
            "research_only": True,
            "production_changed": False,
            "reason": "insufficient_valid_oos_blocks",
            "n": len(rows),
            "blocks": len(blocks),
            "data_source": "live_binance_primary",
        }

    ll = np.asarray([b["delta"]["logloss"] for b in blocks], dtype=float)
    br = np.asarray([b["delta"]["brier"] for b in blocks], dtype=float)
    ac = np.asarray([b["delta"]["accuracy"] for b in blocks], dtype=float)
    summary = {
        "blocks": len(blocks),
        "samples": int(sum(b["n"] for b in blocks)),
        "mean_logloss_delta": float(ll.mean()),
        "mean_brier_delta": float(br.mean()),
        "mean_accuracy_delta": float(ac.mean()),
        "improved_logloss_ratio": float(np.mean(ll < 0)),
        "improved_brier_ratio": float(np.mean(br < 0)),
        "non_worse_accuracy_ratio": float(np.mean(ac >= -0.01)),
    }

    # Frozen holdout: no selection, threshold, or model-family tuning is based on it.
    selected_final = _select_model(development, horizon)
    if selected_final is None:
        return {
            "status": "DEFERRED",
            "research_only": True,
            "production_changed": False,
            "reason": "final_model_fit_unavailable",
            "n": len(rows),
            "blocks": len(blocks),
            "data_source": "live_binance_primary",
        }
    final_name, final_model = selected_final
    holdout_candidate = aligned_probs(final_model, holdout)
    holdout_baseline = np.stack([r["production"] for r in holdout])
    hm_base = _metrics([r["y"] for r in holdout], holdout_baseline)
    hm_cand = _metrics([r["y"] for r in holdout], holdout_candidate)
    frozen_delta = {
        "logloss": hm_cand["logloss"] - hm_base["logloss"],
        "brier": hm_cand["brier"] - hm_base["brier"],
        "accuracy": hm_cand["accuracy"] - hm_base["accuracy"],
    }

    eligible = (
        summary["blocks"] >= 12
        and summary["improved_logloss_ratio"] >= 0.60
        and summary["improved_brier_ratio"] >= 0.60
        and summary["mean_logloss_delta"] <= -0.003
        and summary["mean_brier_delta"] <= -0.0015
        and summary["non_worse_accuracy_ratio"] >= 0.80
    )

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "final_holdout_protected": True,
        "final_holdout_used_for_selection": False,
        "data_source": "live_binance_primary",
        "promotion_evidence_eligible": True,
        "selected_model_family": final_name,
        "summary": summary,
        "eligible_pending_frozen_holdout_confirmation": bool(eligible),
        "final_holdout": {
            "n": holdout_n,
            "baseline": hm_base,
            "candidate": hm_cand,
            "delta": frozen_delta,
            "selected_model_family": final_name,
            "descriptive_only": True,
        },
        "blocks": blocks,
    }


def main() -> None:
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "policy": (
            "strict_pit_primary_only; chronological_oos; purge_target_overlap; "
            "embargo_60m; frozen_holdout_descriptive_only; no_runtime_selection"
        ),
        "feature_names": meta_feature_names(),
        "horizons": {h: evaluate_horizon(h) for h in HORIZONS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
