from __future__ import annotations
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from db import DB, init_db
from settlement_source import preferred_source_from_scenario, target_close_preferred
from label_policy import direction_from_prices, direction_from_return, NEUTRAL_BPS

MAX_TARGET_WORKERS = 4


def direction(base: float, actual: float) -> str:
    return direction_from_prices(base, actual)


def scenario_source(raw: str) -> str:
    try:
        return preferred_source_from_scenario(json.loads(raw or '{}'))
    except Exception:
        return 'binance_futures'


def resolve_targets(targets: list[tuple[str, str]], max_workers: int = MAX_TARGET_WORKERS) -> dict[tuple[str, str], tuple[float | None, str]]:
    unique_targets = sorted(set(targets))
    if not unique_targets:
        return {}
    results: dict[tuple[str, str], tuple[float | None, str]] = {}
    workers = max(1, min(max_workers, len(unique_targets)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='btc-settle') as executor:
        futures = {executor.submit(target_close_preferred, target, source): (target, source) for target, source in unique_targets}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception:
                results[key] = (None, 'unavailable')
    return results


def settle():
    init_db()
    now = datetime.now(timezone.utc)
    settled = unavailable = skipped_degraded = 0
    with sqlite3.connect(DB) as con:
        rows = con.execute('''
            SELECT prediction_id, target_5m, target_10m, base_price,
                   p_up_5m, p_down_5m, p_flat_5m,
                   p_up_10m, p_down_10m, p_flat_10m,
                   actual_price_5m, actual_price_10m, model_version, scenario_json
            FROM predictions
            WHERE (actual_price_5m IS NULL AND target_5m <= ?)
               OR (actual_price_10m IS NULL AND target_10m <= ?)
            ORDER BY prediction_id
        ''', (now.isoformat(), now.isoformat())).fetchall()

        targets: list[tuple[str, str]] = []
        for r in rows:
            if r[12] == 'DEGRADED_NO_FRESH_DATA':
                skipped_degraded += 1
                continue
            source = scenario_source(r[13])
            if r[10] is None and r[1] <= now.isoformat():
                targets.append((r[1], source))
            if r[11] is None and r[2] <= now.isoformat():
                targets.append((r[2], source))
        resolved = resolve_targets(targets)

        for r in rows:
            prediction_id, target5, target10, base, up5, down5, flat5, up10, down10, flat10, actual5, actual10, model_version, scenario_json = r
            if model_version == 'DEGRADED_NO_FRESH_DATA':
                continue
            if not isinstance(base, (int, float)) or base <= 0:
                unavailable += 1
                continue
            source = scenario_source(scenario_json)
            if actual5 is None and target5 <= now.isoformat():
                px, _ = resolved.get((target5, source), (None, 'unavailable'))
                if px is not None:
                    actual_dir = direction_from_prices(base, px)
                    pred_dir = max((('UP', up5), ('DOWN', down5), ('FLAT', flat5)), key=lambda x: x[1])[0]
                    con.execute('''UPDATE predictions SET actual_price_5m=?, actual_direction_5m=?, correct_5m=?, settled_5m_at_utc=? WHERE prediction_id=?''', (px, actual_dir, int(actual_dir == pred_dir), now.isoformat(), prediction_id))
                    settled += 1
                else:
                    unavailable += 1
            if actual10 is None and target10 <= now.isoformat():
                px, _ = resolved.get((target10, source), (None, 'unavailable'))
                if px is not None:
                    actual_dir = direction_from_prices(base, px)
                    pred_dir = max((('UP', up10), ('DOWN', down10), ('FLAT', flat10)), key=lambda x: x[1])[0]
                    con.execute('''UPDATE predictions SET actual_price_10m=?, actual_direction_10m=?, correct_10m=?, settled_10m_at_utc=? WHERE prediction_id=?''', (px, actual_dir, int(actual_dir == pred_dir), now.isoformat(), prediction_id))
                    settled += 1
                else:
                    unavailable += 1
    print('settled_fields', settled, 'unavailable_fields', unavailable, 'skipped_degraded', skipped_degraded, 'threshold_bps', NEUTRAL_BPS)


if __name__ == '__main__':
    settle()
