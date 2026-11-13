"""Decide whether a deferred Live Cycle should refresh the persisted PIT/OOS audit.

A safely deferred cycle does not create a new prediction, but it can still settle
existing predictions. Refreshing the PIT audit periodically keeps the immutable
ledger audit current without fabricating a market snapshot on every 5-minute
deferred cycle.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_MAX_AGE_SECONDS = 900
FUTURE_SKEW_SECONDS = 60


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def refresh_required(
    path: Path,
    *,
    now: datetime | None = None,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
) -> bool:
    """Return True when a deferred cycle must refresh its PIT/OOS audit."""
    if max_age_seconds < 0:
        raise ValueError("max_age_seconds_must_be_nonnegative")

    if not path.is_file() or path.stat().st_size <= 0:
        return True

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        generated_raw = payload.get("generated_at_utc")
        if not isinstance(payload, dict) or not generated_raw:
            return True
        generated_at = _parse_utc(str(generated_raw))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return True

    current = now or datetime.now(timezone.utc)
    if generated_at > current + timedelta(seconds=FUTURE_SKEW_SECONDS):
        return True

    age = (current - generated_at).total_seconds()
    return age > max_age_seconds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--max-age-seconds", type=int, default=DEFAULT_MAX_AGE_SECONDS)
    args = parser.parse_args()

    print("REFRESH" if refresh_required(args.path, max_age_seconds=args.max_age_seconds) else "SKIP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
