#!/usr/bin/env python3
"""Safely merge a local Binance WS cache with a remote cache checkpoint.

Exit codes:
  0  merged/persisted successfully
  11 local merged history is gappy/too short; caller must preserve remote
  12 semantic no-op; remote already contains the same rows
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MIN_CONTIGUOUS = 40
MAX_ROWS = 720


def load(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path} is not an object")
    if obj.get("schema_version") != 1 or obj.get("source") != "Binance USD-M Futures WebSocket":
        raise ValueError(f"{path} provenance invalid")
    return obj


def contiguous_tail(rows: list[dict]) -> int:
    ordered = sorted(
        (row for row in rows if isinstance(row, dict) and "open_time_ms" in row),
        key=lambda row: int(row["open_time_ms"]),
    )
    if not ordered:
        return 0
    suffix = 1
    for i in range(len(ordered) - 1, 0, -1):
        if int(ordered[i]["open_time_ms"]) - int(ordered[i - 1]["open_time_ms"]) != 60_000:
            break
        suffix += 1
    return suffix


def semantic_rows(rows: list[dict]) -> list[dict]:
    return [
        {key: value for key, value in row.items() if key != "retrieved_at_ms"}
        for row in rows[-MAX_ROWS:]
        if isinstance(row, dict)
    ]


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: merge_binance_ws_cache.py LOCAL REMOTE")

    local_path = Path(sys.argv[1])
    remote_path = Path(sys.argv[2])
    local = load(local_path)
    remote = load(remote_path)

    local_rows = local.get("rows", [])
    remote_rows = remote.get("rows", [])
    if not isinstance(local_rows, list) or not isinstance(remote_rows, list):
        raise ValueError("cache rows are not lists")

    merged = {
        int(row["open_time_ms"]): row
        for row in remote_rows
        if isinstance(row, dict) and "open_time_ms" in row
    }
    for row in local_rows:
        if not isinstance(row, dict) or "open_time_ms" not in row:
            continue
        key = int(row["open_time_ms"])
        previous = merged.get(key)
        if previous is None or int(row.get("retrieved_at_ms", 0)) >= int(previous.get("retrieved_at_ms", 0)):
            merged[key] = row

    rows = [merged[key] for key in sorted(merged)[-MAX_ROWS:]]
    suffix = contiguous_tail(rows)
    print(f"merged_cache_rows={len(rows)} contiguous_tail={suffix}")
    if suffix < MIN_CONTIGUOUS:
        print("WARN: refusing to publish gappy Binance WS checkpoint; preserving remote cache.", file=sys.stderr)
        return 11

    if semantic_rows(rows) == semantic_rows(remote_rows):
        print("Binance WS cache semantic no-op; remote checkpoint already contains the same rows.")
        return 12

    local["rows"] = rows
    temp = local_path.with_name(".binance_ws_1m.merge.tmp")
    temp.write_text(json.dumps(local, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    temp.replace(local_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: Binance WS cache merge failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
