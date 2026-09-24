"""PIT/OOS situation-aware meta-model research for BTC short-horizon prediction.

The meta-model consumes only prediction-time information persisted with
historical production predictions: calibrated production probabilities,
microstructure snapshots, and descriptive situation state. It is research-only.
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
    CLASSES,
    DB,
    HORIZONS,
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
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _parse_utc(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else None


def primary_live_scenario(scenario: dict[str, Any]) -> bool:
    return isinstance(scenario, dict) and scenario.get("production_mode") == "binance_primary"


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
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

    out: list[dict[str, Any]] = []
    for row in raw:
        if not prediction_precedes_target(row[1], row[2]):
            continue
        scenario = _safe_json(row[9])
        if not primary_live_scenario(scenario):
            continue
        if strict_pit_provenance_reason(scenario, row[1]) is not None:
            continue
        features = _safe_json(row[3])
        micro = scenario.get("microstructure")
        situation = scenario.get("situation")
        production = [_finite(row[6]), _finite(row[7]), _finite(row[5])]
        if not isinstance(features, dict) or not isinstance(micro, dict) or not isinstance(situation, dict):
            continue
        if row[4] not in CLASSES or any(value < 0 for value in production) or sum(production) <= 0:
            continue
        total = sum(production)
        production = [value / total for value in production]
        out.append(
            {
                "id": row[0],
                "created": row[1],
                "target": row[2],
                "x": features,
                "production": production,
                "y": row[4],
                "model_version": row[8],
                "microstructure": micro,
                "situation": situation,
                "production_mode": str(scenario.get("production_mode", "")),
                "horizon": horizon,
            }
        )

    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for row in out:
        key = _row_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def meta_feature_names() -> list[str]:
    names = ["prod_down", "prod_flat", "prod_up", *MICRO_KEYS]
    names.extend(f"missing={key}" for key in MICRO_KEYS)
    names.extend(["situation_entropy", "situation_margin"])
    for key, values in CAT_KEYS:
        names.extend(f"{key}={value}" for value in values)
    return names


def build_meta_vector(row: dict[str, Any]) -> np.ndarray:
    production = row["production"]
    micro = row.get("microstructure") or {}
    situation = row.get("situation") or {}

    values = [float(production[0]), float(production[1]), float(production[2])]
    for key in MICRO_KEYS:
        raw = micro.get(key)
        values.append(_finite(raw, 0.0))
    for key in MICRO_KEYS:
        raw = micro.get(key)
        try:
            number = float(raw)
        except (TypeError, ValueError):
            number = float("nan")
        values.append(1.0 if not math.isfinite(number) else 0.0)

    values.extend(
        [
            _finite(situation.get("normalized_entropy"), 0.5),
            _finite(situation.get("probability_margin"), 0.0),
        ]
    )
    for key, allowed in CAT_KEYS:
        actual = str(situation.get(key, ""))
        values.extend(1.0 if actual == value else 0.0 for value in allowed)

    vector = np.asarray(values, dtype=float)
    if not np.isfinite(vector).all():
        raise ValueError("meta_feature_nonfinite")
    return vector


def causal_train_rows(
    rows: list[dict[str, Any]],
    test_start: str,
    horizon: str,
) -> list[dict[str, Any]]:
    start = _parse_utc(test_start)
    if start is None:
        return []
    cutoff = start - timedelta(minutes=int(EMBARGO_BARS[horizon]))
    out: list[dict[str, Any]] = []
    for row in rows:
        created = _parse_utc(row.get("created"))
        target = _parse_utc(row.get("target"))
        if created is None or target is None or created >= target:
            continue
        if target >= cutoff:
            continue
        out.append(row)
    return out


def factories() -> dict[str, Callable[[], object]]:
    return {
        "logreg": lambda: Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.15, max_iter=3000)),
            ]
        ),
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
    raw = np.asarray(
        model.predict_proba(np.stack([build_meta_vector(row) for row in rows])),
        dtype=float,
    )
    classes = [str(value) for value in getattr(model, "classes_", [])]
    out = np.full((len(rows), 3), 1e-6, dtype=float)
    for index, cls in enumerate(classes):
        if cls in CLASSES:
            out[:, CLASSES.index(cls)] = raw[:, index]
    out /= out.sum(axis=1, keepdims=True)
    return out


def _metrics(y: list[str], probs: np.ndarray) -> dict[str, float]:
    labels = np.asarray([CLASSES.index(value) for value in y], dtype=int)
    p = np.asarray(probs, dtype=float)
    if len(labels) == 0 or p.shape != (len(labels), 3):
        raise ValueError("metric_shape_invalid")
    picked = np.clip(p[np.arange(len(labels)), labels], 1e-12, 1.0)
    return {
        "logloss": float(-np.mean(np.log(picked))),
        "brier": float(np.mean(np.sum((p - np.eye(3)[labels]) ** 2, axis=1))),
        "accuracy": float(np.mean(np.argmax(p, axis=1) == labels)),
        "n": int(len(labels)),
    }


def _select_model(
    train: list[dict[str, Any]],
    horizon: str,
) -> tuple[str, object] | None:
    if len(train) < MIN_TRAIN:
        return None
    split = max(int(len(train) * 0.70), len(train) - 500)
    if split < 500 or len(train) - split < 100:
        return None

    valid_rows = train[split:]
    fit_rows = causal_train_rows(
        train[:split],
        valid_rows[0].get("created", ""),
        horizon,
    )
    if len(fit_rows) < 500:
        return None

    y_fit = np.asarray([row["y"] for row in fit_rows])
    if len(set(y_fit.tolist())) < 3:
        return None

    scored: list[tuple[float, str]] = []
    for name, factory in factories().items():
        try:
            model = factory()
            model.fit(
                np.stack([build_meta_vector(row) for row in fit_rows]),
                y_fit,
            )
            metrics = _metrics(
                [row["y"] for row in valid_rows],
                aligned_probs(model, valid_rows),
            )
            score = 0.70 * metrics["logloss"] + 0.30 * metrics["brier"]
            scored.append((score, name))
        except Exception:
            continue

    if not scored:
        return None

    scored.sort()
    winner = scored[0][1]
    model = factories()[winner]()
    model.fit(
        np.stack([build_meta_vector(row) for row in train]),
        np.asarray([row["y"] for row in train]),
    )
    return winner, model


def _endpoints(n: int) -> list[int]:
    endpoints = list(range(MIN_TRAIN, n, TEST_BLOCK))
    if len(endpoints) <= MAX_BLOCKS:
        return endpoints
    return sorted(
        set(int(value) for value in np.linspace(
            endpoints[0], endpoints[-1], MAX_BLOCKS
        ))
    )


def _situation_summary(
    rows: list[dict[str, Any]],
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        state = str((row.get("situation") or {}).get("market_state", "UNKNOWN"))
        groups.setdefault(state, []).append(index)

    out: dict[str, Any] = {}
    for state, indices in groups.items():
        y = [rows[index]["y"] for index in indices]
        base = np.stack([baseline[index] for index in indices])
        candidate_probs = np.stack([candidate[index] for index in indices])
        baseline_metrics = _metrics(y, base)
        candidate_metrics = _metrics(y, candidate_probs)
        out[state] = {
            "n": len(indices),
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "delta": {
                "logloss": candidate_metrics["logloss"] - baseline_metrics["logloss"],
                "brier": candidate_metrics["brier"] - baseline_metrics["brier"],
                "accuracy": candidate_metrics["accuracy"] - baseline_metrics["accuracy"],
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
    blocks: list[dict[str, Any]] = []

    for end in _endpoints(len(development)):
        test = development[end:min(end + TEST_BLOCK, len(development))]
        if len(test) < max(10, TEST_BLOCK // 2):
            continue
        train = causal_train_rows(
            development[:end],
            test[0].get("created", ""),
            horizon,
        )
        selected = _select_model(train, horizon)
        if selected is None:
            continue
        name, model = selected

        candidate = aligned_probs(model, test)
        baseline = np.stack([row["production"] for row in test])
        baseline_metrics = _metrics([row["y"] for row in test], baseline)
        candidate_metrics = _metrics([row["y"] for row in test], candidate)

        blocks.append(
            {
                "n": len(test),
                "model": name,
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "delta": {
                    "logloss": candidate_metrics["logloss"] - baseline_metrics["logloss"],
                    "brier": candidate_metrics["brier"] - baseline_metrics["brier"],
                    "accuracy": candidate_metrics["accuracy"] - baseline_metrics["accuracy"],
                },
                "situations": _situation_summary(test, baseline, candidate),
            }
        )

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

    logloss_delta = np.asarray([block["delta"]["logloss"] for block in blocks], dtype=float)
    brier_delta = np.asarray([block["delta"]["brier"] for block in blocks], dtype=float)
    accuracy_delta = np.asarray([block["delta"]["accuracy"] for block in blocks], dtype=float)
    summary = {
        "blocks": len(blocks),
        "samples": int(sum(block["n"] for block in blocks)),
        "mean_logloss_delta": float(logloss_delta.mean()),
        "mean_brier_delta": float(brier_delta.mean()),
        "mean_accuracy_delta": float(accuracy_delta.mean()),
        "improved_logloss_ratio": float(np.mean(logloss_delta < 0)),
        "improved_brier_ratio": float(np.mean(brier_delta < 0)),
        "non_worse_accuracy_ratio": float(np.mean(accuracy_delta >= -0.01)),
    }

    # Frozen holdout is touched exactly once for final descriptive evaluation.
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
    holdout_baseline = np.stack([row["production"] for row in holdout])
    holdout_candidate = aligned_probs(final_model, holdout)
    holdout_base_metrics = _metrics([row["y"] for row in holdout], holdout_baseline)
    holdout_candidate_metrics = _metrics([row["y"] for row in holdout], holdout_candidate)
    frozen_delta = {
        "logloss": holdout_candidate_metrics["logloss"] - holdout_base_metrics["logloss"],
        "brier": holdout_candidate_metrics["brier"] - holdout_base_metrics["brier"],
        "accuracy": holdout_candidate_metrics["accuracy"] - holdout_base_metrics["accuracy"],
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
            "baseline": holdout_base_metrics,
            "candidate": holdout_candidate_metrics,
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
        "horizons": {horizon: evaluate_horizon(horizon) for horizon in HORIZONS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
