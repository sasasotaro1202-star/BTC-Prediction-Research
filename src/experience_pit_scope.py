"""Shared strict-PIT population filter for experience research.

Only prediction experiences backed by a valid primary Binance provenance envelope
are admitted. Legacy/unverifiable rows remain in the ledger for audit/history,
but are excluded from adaptive learning.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

try:
    from pit_oos_audit import AVAILABLE_STATUSES, validate_provenance_envelope
except ModuleNotFoundError:
    from src.pit_oos_audit import AVAILABLE_STATUSES, validate_provenance_envelope

try:
    from db import DB, init_db
except ModuleNotFoundError:
    from src.db import DB, init_db

MAX_TIME_SKEW_SECONDS = 60
PRIMARY_SOURCE = "binance_futures"


def _parse_utc(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


def _strict_prediction_ok(
    row: sqlite3.Row,
    horizon: str,
) -> tuple[bool, str]:
    actual_key = f"actual_direction_{horizon}"
    target_key = f"target_{horizon}"
    actual = row[actual_key]
    target_raw = row[target_key]
    if actual not in {"DOWN", "FLAT", "UP"} or not target_raw:
        return False, "missing_settled_outcome"

    try:
        created = _parse_utc(row["created_at_utc"])
        target = _parse_utc(target_raw)
    except Exception:
        return False, "invalid_prediction_timestamp"

    try:
        scenario = json.loads(row["scenario_json"] or "{}")
    except Exception:
        return False, "invalid_scenario_json"
    if not isinstance(scenario, dict):
        return False, "scenario_not_object"

    provenance = scenario.get("provenance")
    if not isinstance(provenance, dict):
        return False, "missing_top_level_provenance"

    top_errors = validate_provenance_envelope(provenance, "experience")
    if top_errors:
        return False, top_errors[0]

    decision_raw = scenario.get("decision_time_utc") or provenance.get("prediction_cutoff")
    if not decision_raw:
        return False, "missing_decision_time"
    try:
        decision = _parse_utc(decision_raw)
    except Exception:
        return False, "invalid_decision_time"

    if abs((decision - created).total_seconds()) > MAX_TIME_SKEW_SECONDS:
        return False, "decision_time_mismatch"
    if target <= decision:
        return False, "target_not_after_decision"

    if horizon == "10m":
        target5_raw = row["target_5m"]
        if not target5_raw:
            return False, "missing_5m_target"
        try:
            if target <= _parse_utc(target5_raw):
                return False, "10m_target_not_after_5m_target"
        except Exception:
            return False, "invalid_5m_target"

    sources = provenance.get("sources")
    if not isinstance(sources, dict):
        return False, "missing_source_provenance"
    primary = sources.get(PRIMARY_SOURCE)
    if not isinstance(primary, dict):
        return False, "missing_binance_provenance"
    if str(primary.get("status", "")) not in AVAILABLE_STATUSES:
        return False, "binance_provenance_not_available"
    source_errors = validate_provenance_envelope(primary, "experience:binance_futures")
    if source_errors:
        return False, source_errors[0]
    return True, "ok"


def load_strict_primary_rows(
    db_path: Any = DB,
    horizon: str | None = None,
) -> tuple[list[sqlite3.Row], dict[str, Any]]:
    init_db()
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        params: list[Any] = []
        where = [
            "e.actual_direction IN ('DOWN','FLAT','UP')",
            "e.settled_at_utc IS NOT NULL",
        ]
        if horizon is not None:
            where.append("e.horizon = ?")
            params.append(horizon)
        rows = list(
            con.execute(
                f"""SELECT e.*, p.scenario_json,
                           p.target_5m, p.target_10m,
                           p.actual_direction_5m, p.actual_direction_10m
                    FROM experience_ledger e
                    JOIN predictions p ON p.prediction_id = e.prediction_id
                    WHERE {' AND '.join(where)}
                    ORDER BY e.settled_at_utc, e.experience_id""",
                params,
            ).fetchall()
        )

    accepted: list[sqlite3.Row] = []
    reasons: dict[str, int] = {}
    for row in rows:
        ok, reason = _strict_prediction_ok(row, str(row["horizon"]))
        if ok:
            accepted.append(row)
        else:
            reasons[reason] = reasons.get(reason, 0) + 1

    diagnostics = {
        "input_rows": int(len(rows)),
        "accepted_rows": int(len(accepted)),
        "excluded_rows": int(len(rows) - len(accepted)),
        "primary_source": PRIMARY_SOURCE,
        "horizon": horizon,
        "exclusion_reasons": dict(sorted(reasons.items())),
        "pit_policy": "strict_primary_binance_provenance_and_target_after_decision",
    }
    return accepted, diagnostics
