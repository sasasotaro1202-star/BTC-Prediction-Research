from __future__ import annotations
import sqlite3
from datetime import datetime, timezone
from db import DB, init_db
from market_data import target_close_binance

THRESHOLD = 0.00020  # 2 bps

def direction(base: float, actual: float, threshold: float = THRESHOLD) -> str:
    r = actual / base - 1.0
    if r > threshold:
        return 'UP'
    if r < -threshold:
        return 'DOWN'
    return 'FLAT'

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
        for r in rows:
            prediction_id, target5, target10, base, up5, down5, flat5, up10, down10, flat10, actual5, actual10 = r
            if actual5 is None and target5 <= now.isoformat():
                px, _ = target_close_binance(target5)
                if px is not None:
                    actual_dir = direction(base, px)
                    pred_dir = max((('UP', up5), ('DOWN', down5), ('FLAT', flat5)), key=lambda x: x[1])[0]
                    con.execute('''UPDATE predictions SET actual_price_5m=?, actual_direction_5m=?, correct_5m=?, settled_5m_at_utc=? WHERE prediction_id=?''', (px, actual_dir, int(actual_dir == pred_dir), now.isoformat(), prediction_id))
                    settled += 1
                else:
                    unavailable += 1
            if actual10 is None and target10 <= now.isoformat():
                px, _ = target_close_binance(target10)
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
