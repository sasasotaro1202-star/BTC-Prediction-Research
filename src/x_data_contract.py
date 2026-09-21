"""Strict contract and point-in-time helpers for external X/Twitter research data.

This module is intentionally model-agnostic: X data is stored and audited first.
No X-derived value is allowed into the production predictor from this module.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "x_v1"
SOURCE_KIND = "x_public_post"
REQUIRED = ("post_id", "author_handle", "created_at_utc", "text", "source_url")
MAX_TEXT_CHARS = 20_000


def parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp must be a non-empty string")
    dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamp must include an explicit timezone")
    return dt.astimezone(timezone.utc)


def normalize_post(raw: dict[str, Any], *, fetched_at_utc: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("X post must be an object")
    missing = [k for k in REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"missing required fields: {','.join(missing)}")

    created = parse_utc(str(raw["created_at_utc"]))
    fetched = parse_utc(fetched_at_utc)
    text = str(raw["text"])
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError("X post text exceeds maximum length")
    post_id = str(raw["post_id"]).strip()
    handle = str(raw["author_handle"]).strip().lstrip("@").lower()
    url = str(raw["source_url"]).strip()
    if not post_id or not handle or not url.startswith("https://x.com/"):
        raise ValueError("invalid post identity/source URL")
    if created > fetched:
        raise ValueError("post created_at_utc is after fetched_at_utc")

    # Preserve the exact source text; derived NLP fields must never overwrite it.
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "source_kind": SOURCE_KIND,
        "post_id": post_id,
        "author_handle": handle,
        "created_at_utc": created.isoformat(),
        "fetched_at_utc": fetched.isoformat(),
        "text": text,
        "source_url": url,
        "lang": str(raw.get("lang", "unknown")),
        "is_repost": bool(raw.get("is_repost", False)),
        "raw_sha256": str(raw.get("raw_sha256") or canonical_hash(raw)),
    }
    return normalized


def canonical_hash(raw: dict[str, Any]) -> str:
    payload = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def available_at(post: dict[str, Any], decision_time_utc: str) -> bool:
    """Return whether a post was published by the prediction decision time."""
    return parse_utc(str(post["created_at_utc"])) <= parse_utc(decision_time_utc)


def deduplicate(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by immutable X post id while preserving first occurrence."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for post in posts:
        key = str(post["post_id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(post)
    return out
