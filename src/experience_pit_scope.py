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
    required_source: str | None = PRIMARY_SOURCE,
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
    if not isinstance(provenance, dict) or not provenance:
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
    if not isinstance(sources, dict) or not sources:
        return False, "missing_source_provenance"
    if required_source is None:
        valid_sources = _valid_source_names(row)
        if not valid_sources:
            return False, "no_valid_production_source"
        return True, "ok"
    primary = sources.get(required_source)
    if not isinstance(primary, dict):
        return False, f"missing_{required_source}_provenance"
    if str(primary.get("status", "")) not in AVAILABLE_STATUSES:
        return False, f"{required_source}_provenance_not_available"
    source_errors = validate_provenance_envelope(
        primary,
        f"experience:{required_source}",
    )
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


def _valid_source_names(
    row: sqlite3.Row,
) -> set[str]:
    """Return PIT-valid source records without treating failed/unused sources as violations."""
    try:
        scenario = json.loads(row["scenario_json"] or "{}")
    except Exception:
        return set()
    if not isinstance(scenario, dict):
        return set()
    provenance = scenario.get("provenance")
    if not isinstance(provenance, dict):
        return set()
    if validate_provenance_envelope(provenance, "experience"):
        return set()
    sources = provenance.get("sources")
    if not isinstance(sources, dict) or not sources:
        return set()
    valid: set[str] = set()
    for source_name, source_record in sources.items():
        if not isinstance(source_record, dict):
            continue
        if str(source_record.get("status", "")) not in AVAILABLE_STATUSES:
            continue
        if not validate_provenance_envelope(source_record, f"experience:{source_name}"):
            valid.add(str(source_name))
    return valid


def load_strict_verified_rows(
    db_path: Any = DB,
    horizon: str | None = None,
) -> tuple[list[sqlite3.Row], dict[str, Any]]:
    """Load any production-source row with a fully valid PIT provenance envelope.

    This is a research population only. Primary Binance rows remain explicitly
    counted so promotion gates can continue to require the primary benchmark
    population; verified fallback rows are never silently upgraded to primary.
    """
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
    source_counts: dict[str, int] = {}
    primary_count = 0
    fallback_count = 0
    for row in rows:
        ok, reason = _strict_prediction_ok(
            row,
            str(row["horizon"]),
            required_source=None,
        )
        if not ok:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        sources = _valid_source_names(row)
        if not sources:
            reasons["no_valid_production_source"] = reasons.get(
                "no_valid_production_source", 0
            ) + 1
            continue
        accepted.append(row)
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        if PRIMARY_SOURCE in sources:
            primary_count += 1
        elif sources.intersection({"bybit_futures", "coinbase_futures", "kraken_futures"}):
            fallback_count += 1

    diagnostics = {
        "input_rows": int(len(rows)),
        "accepted_rows": int(len(accepted)),
        "excluded_rows": int(len(rows) - len(accepted)),
        "primary_source": PRIMARY_SOURCE,
        "horizon": horizon,
        "population": "all_verified_production_sources",
        "promotion_scope": "binance_futures_primary_only",
        "verified_primary_rows": int(primary_count),
        "verified_fallback_rows": int(fallback_count),
        "verified_source_counts": dict(sorted(source_counts.items())),
        "exclusion_reasons": dict(sorted(reasons.items())),
        "pit_policy": "strict_pit_any_verified_production_source_with_explicit_primary_benchmark_separation",
    }
    return accepted, diagnostics
