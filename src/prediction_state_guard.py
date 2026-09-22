"""Deterministic fingerprinting for the persisted BTC prediction-event table.

Used only as a safety guard. The fingerprint includes row identity and every
persisted predictions-table column, so research/maintenance workflows can fail
closed if they unexpectedly mutate or truncate prediction state.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA_COLUMNS = "PRAGMA table_info(predictions)"


def _canonical_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, (str, int, float)) or value is None:
        return value
    return str(value)


def prediction_table_fingerprint(db: Path) -> dict[str, Any]:
    """Return a deterministic summary + SHA-256 of the complete predictions table."""
    if not db.is_file() or db.stat().st_size <= 0:
        raise ValueError(f"prediction database missing or empty: {db}")

    with sqlite3.connect(db) as con:
        info = con.execute(SCHEMA_COLUMNS).fetchall()
        if not info:
            raise ValueError("predictions table is missing")

        columns = [str(row[1]) for row in info]
        quoted = ",".join('"' + c.replace('"', '""') + '"' for c in columns)
        rows = con.execute(
            f'SELECT rowid,{quoted} FROM predictions ORDER BY rowid'
        ).fetchall()

    digest = hashlib.sha256()
    for row in rows:
        payload = {
            "rowid": int(row[0]),
            "values": {
                column: _canonical_value(value)
                for column, value in zip(columns, row[1:])
            },
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    min_rowid = int(rows[0][0]) if rows else None
    max_rowid = int(rows[-1][0]) if rows else None
    max_created = None
    if "created_at_utc" in columns and rows:
        created_idx = 1 + columns.index("created_at_utc")
        max_created = max(
            (str(row[created_idx]) for row in rows if row[created_idx] is not None),
            default=None,
        )

    return {
        "schema_version": 1,
        "count": len(rows),
        "sha256": digest.hexdigest(),
        "columns": columns,
        "min_rowid": min_rowid,
        "max_rowid": max_rowid,
        "max_created_at_utc": max_created,
    }
