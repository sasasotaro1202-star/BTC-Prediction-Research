"""Research-only OOS challenger for unweighted experience-risk logistic regression.

This candidate keeps the real historical error rate as the class prior instead
of forcing balanced class weights. It uses the same chronological settlement
boundary as the main experience learner and never changes production state.
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
    from experience_policy_oos import _config, prequential_evaluate
    from db import DB
except ModuleNotFoundError:
    from src.experience_policy_oos import _config, prequential_evaluate
    from src.db import DB

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_research" / "experience_policy_unweighted_logistic_oos.json"
CONFIG = ROOT / "config" / "EXPERIENCE_POLICY_UNWEIGHTED_LOGISTIC_OOS.json"

DEFAULT_CONFIG = {
    "schema_version": 1,
    "min_train_rows": 100,
    "thresholds": [0.55, 0.60, 0.65, 0.70],
    "max_report_cases": 100,
    "model_c": 0.5,
    "class_weight": None,
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    if not path.is_file():
        return dict(DEFAULT_CONFIG)
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return dict(DEFAULT_CONFIG)
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(obj)
        return cfg
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return dict(DEFAULT_CONFIG)


def build(
    db_path: Path = DB,
    *,
    config_path: Path = CONFIG,
    output_path: Path = OUT,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    horizons: dict[str, list[Any]] = {"5m": [], "10m": []}
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
        "candidate": "unweighted_logistic_regression",
        "learning_boundary": (
            "each test case is scored using only prior_settled_experiences "
            "(earlier settlement timestamps); same-settlement cases are evaluated "
            "as one batch and cannot train one another"
        ),
        "hypothesis": (
            "removing balanced class weighting may improve probability calibration "
            "when prediction-error prevalence differs materially from 50%"
        ),
        "config": cfg,
        "horizons": {},
    }
    thresholds = tuple(float(x) for x in cfg["thresholds"])
    for horizon, rows_for_horizon in horizons.items():
        result = prequential_evaluate(
            rows_for_horizon,
            min_train_rows=int(cfg["min_train_rows"]),
            thresholds=thresholds,
            block_size=1,
            model_c=float(cfg["model_c"]),
            max_report_cases=int(cfg["max_report_cases"]),
            class_weight=None,
        )
        payload["horizons"][horizon] = result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False))
