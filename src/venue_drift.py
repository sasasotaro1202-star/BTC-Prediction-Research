from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from pathlib import Path

from db import DB

OUT = Path(DB).parent / 'historical_research' / 'venue_drift.json'
WINDOW = 5000


def _finite(values):
    return [float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(float(x))]


def build_report(db_path: str | Path = DB, window: int = WINDOW) -> dict:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            'SELECT scenario_json FROM predictions ORDER BY prediction_id DESC LIMIT ?',
            (int(window),),
        ).fetchall()
    finally:
        con.close()

    sources = Counter()
    gaps = []
    coverage = Counter()
    parse_errors = 0
    for (raw,) in rows:
        try:
            obj = json.loads(raw or '{}')
            dq = obj.get('data_quality') or {}
            src = str(dq.get('price_feature_fallback') or 'unknown')
            sources[src] += 1
            if dq.get('bybit_series') is True:
                coverage['bybit_series_available'] += 1
            if dq.get('binance_futures_series') is True:
                coverage['binance_futures_series_available'] += 1
            if dq.get('binance_spot_series') is True:
                coverage['binance_spot_series_available'] += 1
            micro = obj.get('microstructure') or {}
            gap = micro.get('cross_exchange_gap')
            if isinstance(gap, (int, float)) and math.isfinite(float(gap)):
                gaps.append(float(gap))
        except Exception:
            parse_errors += 1

    gaps_sorted = sorted(gaps)
    def pct(q):
        if not gaps_sorted:
            return None
        idx = min(len(gaps_sorted) - 1, max(0, int(round((len(gaps_sorted) - 1) * q))))
        return gaps_sorted[idx]

    total = len(rows)
    source_share = {k: v / total for k, v in sorted(sources.items())} if total else {}
    report = {
        'ok': parse_errors == 0 and total > 0,
        'window': total,
        'source_counts': dict(sorted(sources.items())),
        'source_share': source_share,
        'coverage_counts': dict(sorted(coverage.items())),
        'cross_exchange_gap': {
            'n': len(gaps),
            'median': pct(0.50),
            'p95': pct(0.95),
            'max_abs': max((abs(x) for x in gaps), default=None),
        },
        'parse_errors': parse_errors,
        'policy': 'monitor_only_no_model_input_no_promotion_effect',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    return report


def main():
    report = build_report()
    print(json.dumps(report, ensure_ascii=False))
    if not report['ok']:
        raise SystemExit('venue drift audit could not validate prediction scenario history')


if __name__ == '__main__':
    main()
