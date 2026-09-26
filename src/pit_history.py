"""Bounded, immutable point-in-time audit history persistence.

The PIT audit itself remains authoritative. This helper only persists a compact
history of audit verdicts so drift and regressions can be inspected over time.
History is bucketed to avoid one file per 5-minute production cycle and is
bounded to a fixed retention window to protect repository size.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

HISTORY_INTERVAL_MINUTES = 15
HISTORY_RETENTION = 7 * 24 * (60 // HISTORY_INTERVAL_MINUTES)
FILE_PREFIX = "pit_"


def _parse_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamp_must_include_timezone")
    return dt.astimezone(timezone.utc)


def _bucket_start(value: datetime) -> datetime:
    minute = (value.minute // HISTORY_INTERVAL_MINUTES) * HISTORY_INTERVAL_MINUTES
    return value.replace(minute=minute, second=0, microsecond=0)


def _history_filename(bucket: datetime) -> str:
    return f"{FILE_PREFIX}{bucket.strftime('%Y%m%dT%H%MZ')}.json"


def _runtime_metadata() -> dict[str, Any]:
    return {
        "github_run_id": os.getenv("GITHUB_RUN_ID") or None,
        "github_run_number": os.getenv("GITHUB_RUN_NUMBER") or None,
        "github_workflow": os.getenv("GITHUB_WORKFLOW") or None,
        "github_job": os.getenv("GITHUB_JOB") or None,
        "github_sha": os.getenv("GITHUB_SHA") or None,
        "github_ref": os.getenv("GITHUB_REF") or None,
    }


def record_pit_history(result: dict[str, Any], history_dir: Path) -> dict[str, Any]:
    """Persist one immutable bucketed PIT snapshot and prune old snapshots.

    Existing buckets are never overwritten. A failed history write is surfaced
    to the caller rather than being silently treated as success.
    """
    generated_at = _parse_utc(result["generated_at_utc"])
    bucket = _bucket_start(generated_at)
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / _history_filename(bucket)

    if path.exists():
        removed: list[str] = []
        recorded = False
    else:
        snapshot = dict(result)
        snapshot["history"] = {
            "schema_version": 1,
            "bucket_start_utc": bucket.isoformat(),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": _runtime_metadata(),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
        removed = []
        recorded = True

    candidates = sorted(history_dir.glob(f"{FILE_PREFIX}*.json"))
    if len(candidates) > HISTORY_RETENTION:
        for old in candidates[: len(candidates) - HISTORY_RETENTION]:
            old.unlink()
            removed.append(old.name)

    return {
        "status": "RECORDED" if recorded else "SKIPPED_EXISTING_BUCKET",
        "bucket_start_utc": bucket.isoformat(),
        "path": str(path),
        "retention": HISTORY_RETENTION,
        "history_count": len(list(history_dir.glob(f"{FILE_PREFIX}*.json"))),
        "removed": removed,
    }
