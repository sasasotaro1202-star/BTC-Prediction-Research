"""Canonical immutable BTC prediction-event identity and compaction digests.

The same identity rules are used by state compaction, research-input auditing,
and model-comparison diagnostics. Settlement fields are mutable and therefore
excluded from the immutable event identity.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

SETTLEMENT_COLUMNS = (
    "actual_price_5m",
    "actual_direction_5m",
    "correct_5m",
    "settled_5m_at_utc",
    "actual_price_10m",
    "actual_direction_10m",
    "correct_10m",
    "settled_10m_at_utc",
    "settlement_source_5m",
    "settlement_source_10m",
)

# Settlement timestamps describe when a known outcome was recorded, not the
# outcome itself. Multiple state replicas may therefore contain different
# observation timestamps for the same immutable event. They are reconciled
# deterministically instead of being treated as contradictory outcomes.
SETTLEMENT_TIMESTAMP_COLUMNS = (
    "settled_5m_at_utc",
    "settled_10m_at_utc",
)

SETTLEMENT_STATE_COLUMNS = tuple(
    c for c in SETTLEMENT_COLUMNS if c not in SETTLEMENT_TIMESTAMP_COLUMNS
)


def _canonical_json(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def model_prediction_event_key(
    *,
    created: Any,
    target: Any,
    model_version: Any,
    x: Iterable[Any],
    production: Iterable[Any],
) -> str:
    payload = {
        "created": str(created),
        "target": str(target),
        "model_version": str(model_version),
        "x": [float(v) for v in x],
        "production": [float(v) for v in production],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def prediction_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    """Canonical immutable DB identity for one prediction event."""
    features = _canonical_json(row.get("feature_json") or "{}")
    return (
        str(row.get("created_at_utc")),
        str(row.get("target_5m")),
        str(row.get("target_10m")),
        str(row.get("model_version")),
        features,
        row.get("p_up_5m"),
        row.get("p_down_5m"),
        row.get("p_flat_5m"),
        row.get("p_up_10m"),
        row.get("p_down_10m"),
        row.get("p_flat_10m"),
    )


def _digest_tokens(tokens: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for token in sorted(tokens):
        raw = token.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _identity_token(identity: tuple[Any, ...]) -> str:
    return json.dumps(identity, sort_keys=False, separators=(",", ":"), allow_nan=False)


def compacted_identity_digest(con: sqlite3.Connection) -> tuple[int, str]:
    """Digest the unique immutable prediction-event identities retained after compaction."""
    cols = [r[1] for r in con.execute("PRAGMA table_info(predictions)").fetchall()]
    if not cols:
        return 0, hashlib.sha256(b"").hexdigest()
    quoted = ",".join('"' + c.replace('"', '""') + '"' for c in cols)
    rows = con.execute(f"SELECT {quoted} FROM predictions").fetchall()
    identities = {
        _identity_token(prediction_identity(dict(zip(cols, row))))
        for row in rows
    }
    return len(identities), _digest_tokens(identities)


def canonical_settlement_timestamp(values: Iterable[Any]) -> str:
    """Return the earliest valid timezone-aware settlement timestamp in UTC.

    The settlement timestamp is operational metadata. When replicated rows
    recorded the same outcome at different times, the earliest valid timestamp
    is the deterministic canonical representation. Invalid timestamps fail
    closed rather than being guessed.
    """
    parsed = []
    for value in values:
        if value is None:
            continue
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid_settlement_timestamp:{value!r}") from exc
        if dt.tzinfo is None:
            raise ValueError(f"settlement_timestamp_timezone_required:{value!r}")
        parsed.append(dt.astimezone(timezone.utc))
    if not parsed:
        raise ValueError("no_settlement_timestamp_values")
    return min(parsed).isoformat()


def settlement_state_digest(con: sqlite3.Connection) -> tuple[int, str, tuple[str, ...]]:
    """Digest the merged settlement state expected for each immutable event.

    Outcome fields must agree across duplicate replicas. Settlement timestamps
    are not outcome state; they are canonicalized to the earliest valid UTC
    timestamp so repeated settlement/recovery passes remain idempotent.
    """
    cols = [r[1] for r in con.execute("PRAGMA table_info(predictions)").fetchall()]
    if not cols:
        return 0, hashlib.sha256(b"").hexdigest(), ()
    selected = ["rowid"] + cols
    quoted = ",".join('"' + c.replace('"', '""') + '"' for c in selected)
    rows = con.execute(f"SELECT {quoted} FROM predictions").fetchall()
    groups: dict[tuple[Any, ...], dict[str, set[Any]]] = defaultdict(
        lambda: {c: set() for c in SETTLEMENT_COLUMNS}
    )
    for row in rows:
        item = dict(zip(selected, row))
        identity = prediction_identity(item)
        for col in SETTLEMENT_COLUMNS:
            value = item.get(col)
            if value is not None:
                groups[identity][col].add(value)

    conflicts: list[str] = []
    tokens: list[str] = []
    for identity, settlements in groups.items():
        clean: dict[str, Any] = {}
        for col, values in settlements.items():
            if not values:
                continue
            if col in SETTLEMENT_TIMESTAMP_COLUMNS:
                try:
                    clean[col] = canonical_settlement_timestamp(values)
                except ValueError as exc:
                    conflicts.append(f"{col}:invalid:{exc}")
                continue
            if len(values) > 1:
                conflicts.append(f"{col}:{sorted(map(str, values))}")
            else:
                clean[col] = next(iter(values))
        tokens.append(
            json.dumps(
                {"identity": identity, "settlement": clean},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    return len(groups), _digest_tokens(tokens), tuple(sorted(conflicts))


def canonical_compaction_snapshot(con: sqlite3.Connection) -> dict[str, Any]:
    unique_count, identity_digest = compacted_identity_digest(con)
    settlement_count, settlement_digest, conflicts = settlement_state_digest(con)
    return {
        "unique_event_count": unique_count,
        "immutable_identity_sha256": identity_digest,
        "settlement_group_count": settlement_count,
        "settlement_state_sha256": settlement_digest,
        "settlement_conflicts": list(conflicts),
    }
