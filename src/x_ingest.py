"""Validate and ingest an externally fetched X/Twitter post snapshot.

The fetcher is deliberately separate from this module. This prevents a flaky or
opaque scraper from silently becoming a production model input. Accepted input
is JSON containing either a list of post objects or {"posts": [...]}.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from x_data_contract import deduplicate, normalize_post


def ingest(input_path: Path, output_path: Path, fetched_at_utc: str | None = None) -> dict:
    fetched = fetched_at_utc or datetime.now(timezone.utc).isoformat()
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    posts = raw.get("posts") if isinstance(raw, dict) else raw
    if not isinstance(posts, list):
        raise ValueError("input must be a JSON list or an object containing posts[]")

    normalized = [normalize_post(p, fetched_at_utc=fetched) for p in posts]
    normalized = deduplicate(normalized)
    normalized.sort(key=lambda p: (p["created_at_utc"], p["post_id"]))
    result = {
        "schema_version": "x_dataset_v1",
        "fetched_at_utc": fetched,
        "source": "external_x_fetcher",
        "post_count": len(normalized),
        "posts": normalized,
        "policy": "raw_source_preserved; PIT_filter_required; no_production_model_input",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fetched-at-utc")
    args = parser.parse_args()
    result = ingest(args.input, args.output, args.fetched_at_utc)
    print(json.dumps({k: result[k] for k in ("schema_version", "fetched_at_utc", "post_count", "policy")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
