from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from pathlib import Path

from db import DB

OUT = Path(DB).parent / 'historical_research' / 'flat_diagnostic.json'
WINDOW = 5000
CLASSES = ('DOWN', 'FLAT', 'UP')


def _argmax(values):
    vals = [float(x) for x in values]
    if any(not math.isfinite(x) for x in vals):
        return None
    return CLASSES[max(range(3), key=lambda i: vals[i])]


def _stage(obj: dict, key: str):
    value = obj.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    return _argmax(value)


def build_report(db_path: str | Path = DB, window: int = WINDOW) -> dict:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            '''SELECT p_down_5m,p_flat_5m,p_up_5m,p_down_10m,p_flat_10m,p_up_10m,scenario_json
               FROM predictions ORDER BY prediction_id DESC LIMIT ?''',
            (int(window),),
        ).fetchall()
    finally:
        con.close()

    counts = {h: {stage: Counter() for stage in ('model_raw', 'structural', 'fused_raw', 'calibrated', 'final')} for h in ('5m', '10m')}
    parse_errors = 0
    for p_down5, p_flat5, p_up5, p_down10, p_flat10, p_up10, raw in rows:
        try:
            obj = json.loads(raw or '{}')
            components = obj.get('components') or {}
            for h, final in (
                ('5m', (p_down5, p_flat5, p_up5)),
                ('10m', (p_down10, p_flat10, p_up10)),
            ):
                for stage, key in (
                    ('model_raw', f'model_raw_{h}'),
                    ('structural', f'structural_{h}'),
                    ('fused_raw', f'fused_raw_{h}'),
                    ('calibrated', f'calibrated_{h}'),
                ):
                    winner = _stage(components, key)
                    if winner is not None:
                        counts[h][stage][winner] += 1
                winner = _argmax(final)
                if winner is not None:
                    counts[h]['final'][winner] += 1
        except Exception:
            parse_errors += 1

    report = {
        'ok': parse_errors == 0 and len(rows) > 0,
        'window': len(rows),
        'parse_errors': parse_errors,
        'counts': {
            h: {stage: dict(sorted(counter.items())) for stage, counter in stages.items()}
            for h, stages in counts.items()
        },
        'policy': 'diagnostic_only_no_model_input_no_promotion_effect',
        'purpose': 'locate_the_pipeline_stage_where_FLAT_stops_being_argmax',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    return report


def main():
    report = build_report()
    print(json.dumps(report, ensure_ascii=False))
    if not report['ok']:
        raise SystemExit('flat diagnostic could not validate prediction scenario history')


if __name__ == '__main__':
    main()
