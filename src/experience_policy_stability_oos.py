"""Chronological stability evaluation for experience-derived policies.

Each block is evaluated against a baseline using only experiences settled before
that block. The report is research-only and designed to feed the promotion gate.
"""
from __future__ import annotations

try:
    from experience_pit_scope import load_strict_primary_rows
except ModuleNotFoundError:
    from src.experience_pit_scope import load_strict_primary_rows
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from experience_policy_oos import _baseline, _hierarchical_memory_predict, _metrics, _parse_ts
except ModuleNotFoundError:
    from src.experience_policy_oos import _baseline, _hierarchical_memory_predict, _metrics, _parse_ts

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "predictions.db"
OUT = ROOT / "data" / "historical_research" / "experience_policy_stability_oos.json"
CONFIG = ROOT / "config" / "EXPERIENCE_POLICY_STABILITY_OOS.json"

DEFAULT_CONFIG = {
    "schema_version": 1,
    "min_train_rows": 100,
    "test_block_rows": 50,
    "min_blocks": 5,
    "required_non_worse_fraction": 0.70,
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config(path: Path = CONFIG) -> dict[str, Any]:
    if not path.is_file():
        return dict(DEFAULT_CONFIG)
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return dict(DEFAULT_CONFIG)
        out = dict(DEFAULT_CONFIG)
        out.update(obj)
        return out
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return dict(DEFAULT_CONFIG)


def evaluate_rows(
    rows: list[Any],
    *,
    min_train_rows: int = 100,
    test_block_rows: int = 50,
    min_blocks: int = 5,
) -> dict[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (_parse_ts(row["settled_at_utc"]), int(row["experience_id"])),
    )
    if len(ordered) <= min_train_rows:
        return {
            "status": "DEFERRED",
            "reason": "insufficient_settled_experience",
            "rows": len(ordered),
            "min_train_rows": int(min_train_rows),
        }

    block_size = max(1, int(test_block_rows))
    blocks: list[dict[str, Any]] = []
    start = int(min_train_rows)
    while start > 0 and start < len(ordered) and str(ordered[start]["settled_at_utc"]) == str(ordered[start - 1]["settled_at_utc"]):
        start += 1
    block_id = 0
    while start < len(ordered):
        train_rows = ordered[:start]
        stop = min(len(ordered), start + block_size)
        # Keep an entire settlement timestamp in the same OOS block.
        while (
            stop < len(ordered)
            and str(ordered[stop]["settled_at_utc"]) == str(ordered[stop - 1]["settled_at_utc"])
        ):
            stop += 1
        test_rows = ordered[start:stop]
        y_true = [1 - int(row["correct"]) for row in test_rows]
        baseline = [_baseline(train_rows)] * len(test_rows)
        memory = _hierarchical_memory_predict(train_rows, test_rows)
        baseline_metrics = _metrics(y_true, baseline)
        memory_metrics = _metrics(y_true, memory)
        ll_gain = (
            (baseline_metrics["logloss"] - memory_metrics["logloss"])
            / max(abs(baseline_metrics["logloss"]), 1e-12)
        )
        br_gain = (
            (baseline_metrics["brier"] - memory_metrics["brier"])
            / max(abs(baseline_metrics["brier"]), 1e-12)
        )
        non_worse = (
            memory_metrics["logloss"] <= baseline_metrics["logloss"]
            and memory_metrics["brier"] <= baseline_metrics["brier"]
        )
        block_id += 1
        blocks.append({
            "block_id": block_id,
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "train_end_settled_at_utc": str(train_rows[-1]["settled_at_utc"]),
            "test_start_settled_at_utc": str(test_rows[0]["settled_at_utc"]),
            "test_end_settled_at_utc": str(test_rows[-1]["settled_at_utc"]),
            "baseline": baseline_metrics,
            "candidate": memory_metrics,
            "relative_logloss_gain": float(ll_gain),
            "relative_brier_gain": float(br_gain),
            "non_worse": bool(non_worse),
        })
        start = stop

    usable = [b for b in blocks if int(b["test_rows"]) > 0]
    if not usable:
        return {
            "status": "DEFERRED",
            "reason": "no_oos_blocks",
            "rows": len(ordered),
            "min_train_rows": int(min_train_rows),
        }

    non_worse_count = sum(bool(b["non_worse"]) for b in usable)
    non_worse_fraction = float(non_worse_count / len(usable))
    latest = usable[-1]
    return {
        "status": "OK",
        "rows": len(ordered),
        "min_train_rows": int(min_train_rows),
        "test_block_rows": int(block_size),
        "block_count": len(usable),
        "min_blocks": int(min_blocks),
        "usable_block_count_pass": len(usable) >= int(min_blocks),
        "non_worse_block_count": int(non_worse_count),
        "non_worse_fraction": non_worse_fraction,
        "latest_block": latest,
        "blocks": usable,
        "research_only": True,
        "production_changed": False,
        "promotion_evidence_eligible": False,
    }


def build(
    db_path: Path = DB,
    *,
    config_path: Path = CONFIG,
    output_path: Path = OUT,
) -> dict[str, Any]:
    cfg = _config(config_path)
    rows, pit_scope = load_strict_primary_rows(db_path)
    rows_by_horizon: dict[str, list[Any]] = {"5m": [], "10m": []}
    for row in rows:
        rows_by_horizon[str(row["horizon"])].append(row)


    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "research_only": True,
        "production_changed": False,
        "strict_pit_scope": True,
        "pit_scope": pit_scope,
        "promotion_evidence_eligible": False,
        "evaluation_boundary": "each block uses only earlier settled experiences",
        "config": cfg,
        "horizons": {},
    }
    for horizon, rows in rows_by_horizon.items():
        payload["horizons"][horizon] = evaluate_rows(
            rows,
            min_train_rows=int(cfg["min_train_rows"]),
            test_block_rows=int(cfg["test_block_rows"]),
            min_blocks=int(cfg["min_blocks"]),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False))
