"""Bounded, retrying HTTP transport for free/public BTC data paths.

The goal is not to disable timeouts. Each individual socket operation remains
bounded, while transient network failures receive bounded retries within a
whole-request deadline. Non-retryable HTTP responses fail immediately.
"""
from __future__ import annotations

import json
import random
import socket
import time
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

RETRYABLE_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}
DEFAULT_ATTEMPTS = 5
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_TOTAL_TIMEOUT_SECONDS = 120.0
MAX_BACKOFF_SECONDS = 8.0


def _retry_after_seconds(exc: HTTPError) -> float | None:
    try:
        value = exc.headers.get("Retry-After")
    except Exception:
        return None
    if not value:
        return None
    try:
        return max(0.0, min(30.0, float(value)))
    except (TypeError, ValueError):
        pass
    try:
        target = parsedate_to_datetime(value)
        return max(0.0, min(30.0, target.timestamp() - time.time()))
    except (TypeError, ValueError, OverflowError):
        return None


def request_bytes(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    attempts: int = DEFAULT_ATTEMPTS,
    total_timeout: float = DEFAULT_TOTAL_TIMEOUT_SECONDS,
) -> bytes:
    """Fetch bytes with bounded per-attempt and whole-request timeouts."""
    attempts = max(1, int(attempts))
    timeout = max(0.25, float(timeout))
    total_timeout = max(timeout, float(total_timeout))
    started = time.monotonic()
    deadline = started + total_timeout
    last: Exception | None = None

    for attempt in range(attempts):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        req = Request(url, headers=headers or {})
        per_attempt = min(timeout, remaining)
        try:
            with urlopen(req, timeout=per_attempt) as response:
                return response.read()
        except HTTPError as exc:
            last = exc
            if exc.code not in RETRYABLE_HTTP_STATUS:
                raise
            retry_after = _retry_after_seconds(exc)
        except (URLError, TimeoutError, socket.timeout, OSError) as exc:
            last = exc
            retry_after = None

        if attempt + 1 >= attempts:
            break

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        backoff = min(MAX_BACKOFF_SECONDS, 0.75 * (2 ** attempt))
        if retry_after is not None:
            backoff = max(backoff, retry_after)
        # Small bounded jitter prevents synchronized retries across runners.
        backoff += random.uniform(0.0, min(0.25, backoff * 0.10))
        time.sleep(min(backoff, remaining))

    if last is not None:
        raise last
    raise TimeoutError(f"request deadline exceeded: {url}")


def request_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    attempts: int = DEFAULT_ATTEMPTS,
    total_timeout: float = DEFAULT_TOTAL_TIMEOUT_SECONDS,
):
    raw = request_bytes(
        url,
        headers=headers,
        timeout=timeout,
        attempts=attempts,
        total_timeout=total_timeout,
    )
    return json.loads(raw.decode("utf-8"))
