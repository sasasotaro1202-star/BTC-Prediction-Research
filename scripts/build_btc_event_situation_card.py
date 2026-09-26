#!/usr/bin/env python3
"""Build a research-only BTC Situation Card from persisted Binance WS caches.

No prediction or production artifact is written. The output is suitable for
debugging/replay and later chronological OOS experiments.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.btc_event_layer import (
    build_situation_card,
    event_from_binance_depth,
    event_from_binance_kline,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KLINE = ROOT / "data" / "binance_ws_1m.json"
DEFAULT_DEPTH = ROOT / "data" / "binance_ws_depth.json"


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_events(kline_path: Path, depth_path: Path) -> list[dict]:
    events: list[dict] = []

    if kline_path.is_file():
        obj = _load_json(kline_path)
        rows = obj.get("rows", []) if isinstance(obj, dict) else []
        if not isinstance(rows, list):
            raise ValueError("binance_kline_cache_rows_invalid")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("binance_kline_cache_row_invalid")
            events.append(event_from_binance_kline(row))

    if depth_path.is_file():
        obj = _load_json(depth_path)
        snapshot = obj.get("snapshot") if isinstance(obj, dict) else None
        if snapshot is not None:
            if not isinstance(snapshot, dict):
                raise ValueError("binance_depth_cache_snapshot_invalid")
            events.append(event_from_binance_depth(snapshot))

    return events


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-time-ms", type=int, default=None)
    parser.add_argument("--kline-cache", type=Path, default=DEFAULT_KLINE)
    parser.add_argument("--depth-cache", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "historical_research" / "btc_situation_card.json")
    args = parser.parse_args()

    prediction_time_ms = int(args.prediction_time_ms or time.time() * 1000)
    events = load_events(args.kline_cache, args.depth_cache)
    card = build_situation_card(
        events,
        prediction_time_ms=prediction_time_ms,
        strict_pit=True,
    )
    card["input_cache_files"] = [
        str(args.kline_cache.relative_to(ROOT)) if args.kline_cache.is_relative_to(ROOT) else str(args.kline_cache),
        str(args.depth_cache.relative_to(ROOT)) if args.depth_cache.is_relative_to(ROOT) else str(args.depth_cache),
    ]
    card["event_source_count"] = len(events)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(f".{args.output.name}.tmp")
    temp.write_text(
        json.dumps(card, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(args.output)
    print(json.dumps({
        "status": "OK",
        "prediction_time_ms": prediction_time_ms,
        "events_loaded": len(events),
        "events_used": card["event_count"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
