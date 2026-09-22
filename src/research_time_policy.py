"""Shared time-alignment contracts for research-only historical BTC labels."""
from __future__ import annotations

MINUTE_MS = 60_000


def exact_elapsed_pairs(rows, steps: int):
    """Yield (row, exact-future-row) pairs using elapsed wall-clock minutes."""
    minutes = int(steps)
    if minutes <= 0:
        raise ValueError("steps must be positive")
    index = {int(row[0]): row for row in rows}
    delta = minutes * MINUTE_MS
    for row in rows:
        future = index.get(int(row[0]) + delta)
        if future is not None:
            yield row, future


def consecutive_minute_window(window, interval_ms: int = MINUTE_MS) -> bool:
    """Return True only when every adjacent timestamp is exactly one interval apart."""
    if not window:
        return False
    return all(int(window[i][0]) - int(window[i - 1][0]) == interval_ms for i in range(1, len(window)))
