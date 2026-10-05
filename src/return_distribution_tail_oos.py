"""Research-only conditional BTC return-distribution and tail-risk OOS research.

This lane does not change Production. It learns conditional return quantiles
(q10/q50/q90) from already-settled strict-PIT Binance-primary predictions and
evaluates them in chronological walk-forward blocks against an unconditional
training-window quantile baseline.

The output is deliberately descriptive/research-only. It is not a trading
instruction and it does not manufacture drawdown paths that are not stored in
the prediction ledger.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from db import DB
    from feature_schema import FEATURES
    from model_compare import load_rows
except ModuleNotFoundError:
    from src.db import DB
    from src.feature_schema import FEATURES
    from src.model_compare import load_rows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "return_distribution_tail_oos.json"

HORIZONS = ("5m", "10m")
PURGE_BARS = {"5m": 5, "10m": 10}
EMBARGO_BARS = {"5m": 60, "10m": 60}
MIN_TRAIN = 250
TEST_BLOCK = 50
MIN_BLOCK = 25
MIN_EVAL_BLOCKS = 2
MAX_BLOCKS = 20
EPS = 1e-12


def _parse_iso(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _finite_return(base: Any, actual: Any) -> float | None:
    try:
        b = float(base)
        a = float(actual)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(b) and math.isfinite(a) and b > 0 and a > 0):
        return None
    r = a / b - 1.0
    return r if math.isfinite(r) else None


def _load_primary_return_rows(horizon: str) -> list[dict[str, Any]]:
    """Join strict-PIT prediction rows with the immutable matured endpoint price."""
    strict = load_rows(horizon, strict_pit=True)
    strict = [
        row for row in strict
        if str(row.get("production_mode", "")) == "binance_primary"
    ]
    if not strict:
        return []

    actual_col = f"actual_price_{horizon}"
    target_col = f"target_{horizon}"

    ids = sorted({int(row["id"]) for row in strict})
    placeholders = ",".join("?" for _ in ids)
    by_id: dict[int, tuple[Any, ...]] = {}
    with sqlite3.connect(DB) as con:
        cursor = con.execute(
            f"""
            SELECT prediction_id, created_at_utc, {target_col}, base_price,
                   {actual_col}
            FROM predictions
            WHERE prediction_id IN ({placeholders})
            """,
            ids,
        )
        for record in cursor.fetchall():
            by_id[int(record[0])] = record

    out: list[dict[str, Any]] = []
    for row in strict:
        record = by_id.get(int(row["id"]))
        if record is None:
            continue
        _, created_at, target_at, base_price, actual = record
        created = _parse_iso(created_at)
        target = _parse_iso(target_at)
        if created is None or target is None or not (created < target):
            continue
        future_return = _finite_return(base_price, actual)
        if future_return is None:
            continue
        x = row.get("x")
        if not isinstance(x, list) or len(x) != len(FEATURES):
            continue
        try:
            values = [float(v) for v in x]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in values):
            continue
        out.append(
            {
                "id": int(row["id"]),
                "created": created.isoformat(),
                "target": target.isoformat(),
                "x": values,
                "return": future_return,
            }
        )

    out.sort(key=lambda item: (item["created"], item["id"]))
    seen: set[int] = set()
    deduped: list[dict[str, Any]] = []
    for row in out:
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        deduped.append(row)
    return deduped


def pinball_loss(y: np.ndarray, q: np.ndarray, quantile: float) -> float:
    err = y - q
    return float(np.mean(np.maximum(quantile * err, (quantile - 1.0) * err)))


def interval_coverage(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    return float(np.mean((y >= low) & (y <= high)))


def _model(quantile: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=float(quantile),
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=12,
        l2_regularization=1.5,
        random_state=42,
    )


def _block_endpoints(n: int) -> list[int]:
    raw = list(range(MIN_TRAIN, max(MIN_TRAIN, n - MIN_BLOCK + 1), TEST_BLOCK))
    if len(raw) <= MAX_BLOCKS:
        return raw
    return [int(value) for value in np.linspace(raw[0], raw[-1], MAX_BLOCKS)]


def _fit_quantiles(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    X_train = np.asarray([row["x"] for row in train], dtype=float)
    y_train = np.asarray([row["return"] for row in train], dtype=float)
    X_test = np.asarray([row["x"] for row in test], dtype=float)

    result: dict[str, np.ndarray] = {}
    for name, quantile in (("q10", 0.10), ("q50", 0.50), ("q90", 0.90)):
        model = _model(quantile)
        model.fit(X_train, y_train)
        result[name] = np.asarray(model.predict(X_test), dtype=float)

    result["q50"] = np.maximum(result["q50"], result["q10"])
    result["q90"] = np.maximum(result["q90"], result["q50"])
    return result


def _baseline_quantiles(train_returns: np.ndarray, n: int) -> dict[str, np.ndarray]:
    return {
        "q10": np.full(n, float(np.quantile(train_returns, 0.10)), dtype=float),
        "q50": np.full(n, float(np.quantile(train_returns, 0.50)), dtype=float),
        "q90": np.full(n, float(np.quantile(train_returns, 0.90)), dtype=float),
    }


def _evaluate_block(y: np.ndarray, candidate: dict[str, np.ndarray], baseline: dict[str, np.ndarray]) -> dict[str, Any]:
    candidate_width = candidate["q90"] - candidate["q10"]
    baseline_width = baseline["q90"] - baseline["q10"]

    candidate_pinball = float(
        (
            pinball_loss(y, candidate["q10"], 0.10)
            + pinball_loss(y, candidate["q50"], 0.50)
            + pinball_loss(y, candidate["q90"], 0.90)
        )
        / 3.0
    )
    baseline_pinball = float(
        (
            pinball_loss(y, baseline["q10"], 0.10)
            + pinball_loss(y, baseline["q50"], 0.50)
            + pinball_loss(y, baseline["q90"], 0.90)
        )
        / 3.0
    )

    lower_breach = float(np.mean(y < candidate["q10"]))
    upper_breach = float(np.mean(y > candidate["q90"]))
    median_sign_accuracy = float(np.mean(np.sign(candidate["q50"]) == np.sign(y)))

    lower_mask = y < candidate["q10"]
    upper_mask = y > candidate["q90"]
    lower_cvar = float(np.mean(y[lower_mask])) if lower_mask.any() else None
    upper_tail_mean = float(np.mean(y[upper_mask])) if upper_mask.any() else None

    return {
        "n": int(len(y)),
        "candidate": {
            "pinball_mean": candidate_pinball,
            "q10_pinball": pinball_loss(y, candidate["q10"], 0.10),
            "q50_pinball": pinball_loss(y, candidate["q50"], 0.50),
            "q90_pinball": pinball_loss(y, candidate["q90"], 0.90),
            "interval_coverage": interval_coverage(y, candidate["q10"], candidate["q90"]),
            "interval_width_mean": float(np.mean(candidate_width)),
            "lower_tail_breach_rate": lower_breach,
            "upper_tail_breach_rate": upper_breach,
            "median_sign_accuracy": median_sign_accuracy,
            "predicted_q10_mean": float(np.mean(candidate["q10"])),
            "predicted_q50_mean": float(np.mean(candidate["q50"])),
            "predicted_q90_mean": float(np.mean(candidate["q90"])),
            "realized_lower_tail_mean": lower_cvar,
            "realized_upper_tail_mean": upper_tail_mean,
        },
        "baseline": {
            "pinball_mean": baseline_pinball,
            "q10_pinball": pinball_loss(y, baseline["q10"], 0.10),
            "q50_pinball": pinball_loss(y, baseline["q50"], 0.50),
            "q90_pinball": pinball_loss(y, baseline["q90"], 0.90),
            "interval_coverage": interval_coverage(y, baseline["q10"], baseline["q90"]),
            "interval_width_mean": float(np.mean(baseline_width)),
        },
        "delta": {
            "pinball_mean": candidate_pinball - baseline_pinball,
            "coverage": interval_coverage(y, candidate["q10"], candidate["q90"]) - interval_coverage(y, baseline["q10"], baseline["q90"]),
            "width": float(np.mean(candidate_width) - np.mean(baseline_width)),
        },
    }


def evaluate_horizon(horizon: str) -> dict[str, Any]:
    rows = _load_primary_return_rows(horizon)
    if len(rows) < MIN_TRAIN + TEST_BLOCK:
        return {
            "status": "DEFERRED",
            "research_only": True,
            "production_changed": False,
            "n": len(rows),
            "minimum_rows": MIN_TRAIN + TEST_BLOCK,
            "reason": "insufficient_strict_primary_matured_return_rows",
        }

    blocks: list[dict[str, Any]] = []
    for end in _block_endpoints(len(rows)):
        train_end = max(0, end - PURGE_BARS[horizon] - EMBARGO_BARS[horizon])
        train = rows[:train_end]
        test = rows[end:min(end + TEST_BLOCK, len(rows))]
        if len(train) < MIN_TRAIN or len(test) < MIN_BLOCK:
            continue

        train_returns = np.asarray([row["return"] for row in train], dtype=float)
        y = np.asarray([row["return"] for row in test], dtype=float)
        candidate = _fit_quantiles(train, test)
        baseline = _baseline_quantiles(train_returns, len(test))
        evaluated = _evaluate_block(y, candidate, baseline)
        evaluated.update(
            {
                "test_start": test[0]["created"],
                "test_end": test[-1]["target"],
                "train_end": train[-1]["created"],
            }
        )
        blocks.append(evaluated)

    if len(blocks) < MIN_EVAL_BLOCKS:
        return {
            "status": "DEFERRED",
            "research_only": True,
            "production_changed": False,
            "n": len(rows),
            "minimum_rows": MIN_TRAIN + TEST_BLOCK,
            "blocks": len(blocks),
            "reason": "insufficient_valid_walk_forward_blocks",
        }

    pinball_delta = np.asarray([block["delta"]["pinball_mean"] for block in blocks], dtype=float)
    coverage = np.asarray([block["candidate"]["interval_coverage"] for block in blocks], dtype=float)
    width = np.asarray([block["candidate"]["interval_width_mean"] for block in blocks], dtype=float)
    baseline_pinball = np.asarray([block["baseline"]["pinball_mean"] for block in blocks], dtype=float)
    candidate_pinball = np.asarray([block["candidate"]["pinball_mean"] for block in blocks], dtype=float)
    newest = blocks[-1]

    baseline_mean = float(np.mean(baseline_pinball))
    candidate_mean = float(np.mean(candidate_pinball))
    relative_gain = (baseline_mean - candidate_mean) / max(EPS, abs(baseline_mean))

    return {
        "status": "OK",
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "target_definition": "settled_endpoint_return=(actual_price/base_price)-1",
        "pit_scope": "strict_binance_primary_prediction_provenance",
        "horizon": horizon,
        "n": len(rows),
        "feature_count": len(FEATURES),
        "features": list(FEATURES),
        "walk_forward": {
            "blocks": len(blocks),
            "samples": int(sum(block["n"] for block in blocks)),
            "min_train": MIN_TRAIN,
            "test_block": TEST_BLOCK,
            "purge_bars": PURGE_BARS[horizon],
            "embargo_bars": EMBARGO_BARS[horizon],
        },
        "aggregate": {
            "candidate_pinball_mean": candidate_mean,
            "baseline_pinball_mean": baseline_mean,
            "relative_pinball_improvement": relative_gain,
            "improved_block_ratio": float(np.mean(pinball_delta < 0.0)),
            "non_worse_block_ratio": float(np.mean(pinball_delta <= 0.0)),
            "mean_interval_coverage": float(np.mean(coverage)),
            "coverage_target": 0.80,
            "coverage_abs_error": abs(float(np.mean(coverage)) - 0.80),
            "mean_interval_width": float(np.mean(width)),
            "newest_block_pinball_delta": float(newest["delta"]["pinball_mean"]),
            "newest_block_coverage": float(newest["candidate"]["interval_coverage"]),
            "newest_block_n": int(newest["n"]),
        },
        "tail_summary": {
            "target_lower_tail_breach": 0.10,
            "target_upper_tail_breach": 0.10,
            "mean_lower_tail_breach": float(np.mean([block["candidate"]["lower_tail_breach_rate"] for block in blocks])),
            "mean_upper_tail_breach": float(np.mean([block["candidate"]["upper_tail_breach_rate"] for block in blocks])),
            "newest_lower_tail_breach": float(newest["candidate"]["lower_tail_breach_rate"]),
            "newest_upper_tail_breach": float(newest["candidate"]["upper_tail_breach_rate"]),
            "note": "This endpoint-return lane does not claim intrahorizon maximum drawdown because path data is not part of this research contract.",
        },
        "first_block": blocks[0],
        "newest_block": newest,
        "blocks": blocks,
        "promotion_evidence_eligible": False,
        "decision": "RESEARCH_ONLY_NO_PRODUCTION_EFFECT",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    payload = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "source_snapshot": "canonical_prediction_ledger",
        "horizons": {h: evaluate_horizon(h) for h in HORIZONS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
