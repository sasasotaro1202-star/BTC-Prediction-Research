"""Point-in-time audit for the X research dataset.

This audit is intentionally fail-closed. A dataset is not considered usable for
research if identity, timestamps, source URLs, hashes, or PIT boundaries fail.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from x_data_contract import parse_utc

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data" / "x_research" / "posts.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "historical_research" / "x_pit_audit.json"


def audit(path: Path = DEFAULT_INPUT, out: Path = DEFAULT_OUTPUT) -> dict:
    violations: list[str] = []
    checked = 0
    if not path.exists() or path.stat().st_size == 0:
        violations.append("x_dataset_missing_or_empty")
    else:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if obj.get("schema_version") != "x_dataset_v1":
                violations.append("invalid_dataset_schema")
            posts = obj.get("posts")
            if not isinstance(posts, list):
                violations.append("posts_not_a_list")
                posts = []
            fetched = parse_utc(str(obj.get("fetched_at_utc")))
            ids: set[str] = set()
            for post in posts:
                checked += 1
                pid = str(post.get("post_id", "")).strip()
                if not pid:
                    violations.append(f"row_{checked}:missing_post_id")
                elif pid in ids:
                    violations.append(f"row_{checked}:duplicate_post_id:{pid}")
                ids.add(pid)
                try:
                    created = parse_utc(str(post.get("created_at_utc")))
                    fetched_row = parse_utc(str(post.get("fetched_at_utc")))
                    if created > fetched:
                        violations.append(f"row_{checked}:created_after_dataset_fetch")
                    if created > fetched_row:
                        violations.append(f"row_{checked}:created_after_row_fetch")
                    if fetched_row > fetched:
                        violations.append(f"row_{checked}:row_fetch_after_dataset_fetch")
                except Exception:
                    violations.append(f"row_{checked}:invalid_timestamp")
                if not str(post.get("source_url", "")).startswith("https://x.com/"):
                    violations.append(f"row_{checked}:invalid_source_url")
                if len(str(post.get("raw_sha256", ""))) != 64:
                    violations.append(f"row_{checked}:invalid_raw_sha256")
                if not str(post.get("author_handle", "")).strip():
                    violations.append(f"row_{checked}:missing_author_handle")
                if not isinstance(post.get("text"), str):
                    violations.append(f"row_{checked}:text_not_string")
        except Exception as exc:
            violations.append(f"dataset_parse_error:{type(exc).__name__}")

    result = {
        "ok": not violations,
        "checked_posts": checked,
        "violation_count": len(violations),
        "violations": violations[:200],
        "policy": "fail_closed; immutable_post_id; explicit_utc_timestamps; source_hash_required; PIT_filter_required; no_production_model_input",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    result = audit()
    print(json.dumps(result, ensure_ascii=False))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
