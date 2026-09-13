from __future__ import annotations
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from db import DB, init_db
from market_data import target_close_binance

THRESHOLD = 0.00020  # 2 bps
MAX_TARGET_WORKERS = 4


def direction(base: float, actual: float, threshold: float = THRESHOLD) -> str:
    r = actual / base - 1.0
    if r > threshold:
        return 'UP'
    if r < -threshold:
        return 'DOWN'
    return 'FLAT'


def resolve_targets(targets: list[str], max_workers: int = MAX_TARGET_WORKERS) -> dict[str, tuple[float | None, str]]:
    """Resolve unique target timestamps concurrently, without changing scoring semantics."""
    unique_targets = sorted(set(targets))
    if not unique_targets:
        return {}

    results: dict[str, tuple[float | None, str]] = {}
    workers = max(1, min(max_workers, len(unique_targets)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='btc-settle') as executor:
        futures = {executor.submit(target_close_binance, target): target for target in unique_targets}
        for future in as_completed(futures):
            target = futures[future]
            try:
                results[target] = future.result()
            except Exception:
                # A single public-source failure must not abort settlement of the other targets.
                results[target] = (None, 'unavailable')
    return results


def settle():
    init_db()
    now = datetime.now(timezone.utc)
    settled = unavailable = 0
    with sqlite3.connect(DB) as con:
        rows = con.execute('''
            SELECT prediction_id, target_5m, target_10m, base_price,
                   p_up_5m, p_down_5m, p_flat_5m,
                   p_up_10m, p_down_10m, p_flat_10m,
                   actual_price_5m, actual_price_10m
            FROM predictions
            WHERE (actual_price_5m IS NULL AND target_5m <= ?)
               OR (actual_price_10m IS NULL AND target_10m <= ?)
            ORDER BY prediction_id
        ''', (now.isoformat(), now.isoformat())).fetchall()

        targets: list[str] = []
        for r in rows:
            if r[10] is None and r[1] <= now.isoformat():
                targets.append(r[1])
            if r[11] is None and r[2] <= now.isoformat():
                targets.append(r[2])
        resolved = resolve_targets(targets)

        for r in rows:
            prediction_id, target5, target10, base, up5, down5, flat5, up10, down10, flat10, actual5, actual10 = r
            if actual5 is None and target5 <= now.isoformat():
                px, _ = resolved.get(target5, (None, 'unavailable'))
                if px is not None:
                    actual_dir = direction(base, px)
                    pred_dir = max((('UP', up5), ('DOWN', down5), ('FLAT', flat5)), key=lambda x: x[1])[0]
                    con.execute('''UPDATE predictions SET actual_price_5m=?, actual_direction_5m=?, correct_5m=?, settled_5m_at_utc=? WHERE prediction_id=?''', (px, actual_dir, int(actual_dir == pred_dir), now.isoformat(), prediction_id))
                    settled += 1
                else:
                    unavailable += 1
            if actual10 is None and target10 <= now.isoformat():
                px, _ = resolved.get(target10, (None, 'unavailable'))
                if px is not None:
                    actual_dir = direction(base, px)
                    pred_dir = max((('UP', up10), ('DOWN', down10), ('FLAT', flat10)), key=lambda x: x[1])[0]
                    con.execute('''UPDATE predictions SET actual_price_10m=?, actual_direction_10m=?, correct_10m=?, settled_10m_at_utc=? WHERE prediction_id=?''', (px, actual_dir, int(actual_dir == pred_dir), now.isoformat(), prediction_id))
                    settled += 1
                else:
                    unavailable += 1
    print('settled_fields', settled, 'unavailable_fields', unavailable, 'threshold_bps', THRESHOLD * 10000)


if __name__ == '__main__':
    settle()
