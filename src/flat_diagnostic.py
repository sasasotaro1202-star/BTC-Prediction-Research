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
    if len(vals) != 3 or any(not math.isfinite(x) for x in vals):
        return None
    return CLASSES[max(range(3), key=lambda i: vals[i])]


def _probabilities(value):
    if isinstance(value, dict):
        try:
            vals = [float(value[c]) for c in CLASSES]
        except (KeyError, TypeError, ValueError):
            return None
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            vals = [float(x) for x in value]
        except (TypeError, ValueError):
            return None
    else:
        return None
    if any(not math.isfinite(x) or x < 0.0 for x in vals):
        return None
    total = sum(vals)
    if not math.isfinite(total) or total <= 0.0:
        return None
    return [x / total for x in vals]


def _metrics(labels, probabilities):
    if not labels or len(labels) != len(probabilities):
        return None
    y_idx = [CLASSES.index(label) for label in labels]
    n = len(labels)
    correct = 0
    logloss_sum = 0.0
    brier_sum = 0.0
    confidences = []
    hits = []
    for y, raw in zip(y_idx, probabilities):
        p = _probabilities(raw)
        if p is None:
            return None
        pred = max(range(3), key=lambda i: p[i])
        correct += int(pred == y)
        logloss_sum -= math.log(max(p[y], 1e-15))
        brier_sum += sum((p[i] - (1.0 if i == y else 0.0)) ** 2 for i in range(3))
        confidences.append(max(p))
        hits.append(float(pred == y))
    ece = 0.0
    for bucket in range(10):
        lo = bucket / 10.0
        hi = (bucket + 1) / 10.0
        members = [
            i for i, conf in enumerate(confidences)
            if conf >= lo and (conf <= hi if bucket == 9 else conf < hi)
        ]
        if members:
            acc = sum(hits[i] for i in members) / len(members)
            conf = sum(confidences[i] for i in members) / len(members)
            ece += len(members) / n * abs(acc - conf)
    return {
        'n': n,
        'accuracy': correct / n,
        'logloss': logloss_sum / n,
        'brier': brier_sum / n,
        'ece': ece,
    }


def _settled_stage_diagnostics(db_path: str | Path, window: int):
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            '''SELECT actual_direction_5m,p_down_5m,p_flat_5m,p_up_5m,
                      actual_direction_10m,p_down_10m,p_flat_10m,p_up_10m,
                      scenario_json
               FROM predictions ORDER BY prediction_id DESC LIMIT ?''',
            (int(window),),
        ).fetchall()
    finally:
        con.close()

    labels = {h: {stage: [] for stage in ('model_raw', 'structural', 'fused_raw', 'calibrated', 'final')} for h in ('5m', '10m')}
    probabilities = {h: {stage: [] for stage in labels[h]} for h in ('5m', '10m')}
    transition_pairs = {h: {stage: [] for stage in ('structural', 'fused_raw', 'calibrated', 'final')} for h in ('5m', '10m')}

    for actual5, p_down5, p_flat5, p_up5, actual10, p_down10, p_flat10, p_up10, raw in rows:
        try:
            obj = json.loads(raw or '{}')
            components = obj.get('components') or {}
            for h, actual, final in (
                ('5m', actual5, (p_down5, p_flat5, p_up5)),
                ('10m', actual10, (p_down10, p_flat10, p_up10)),
            ):
                if actual not in CLASSES:
                    continue
                stage_values = {
                    'model_raw': components.get(f'model_raw_{h}'),
                    'structural': components.get(f'structural_{h}'),
                    'fused_raw': components.get(f'fused_raw_{h}'),
                    'calibrated': components.get(f'calibrated_{h}'),
                    'final': final,
                }
                stage_preds = {}
                for stage, value in stage_values.items():
                    p = _probabilities(value)
                    if p is None:
                        continue
                    labels[h][stage].append(actual)
                    probabilities[h][stage].append(p)
                    stage_preds[stage] = CLASSES[max(range(3), key=lambda i: p[i])]
                raw_pred = stage_preds.get('model_raw')
                if raw_pred is not None:
                    for target_stage in transition_pairs[h]:
                        target_pred = stage_preds.get(target_stage)
                        if target_pred is not None:
                            transition_pairs[h][target_stage].append((raw_pred, target_pred))
        except Exception:
            continue

    settled_metrics = {h: {} for h in ('5m', '10m')}
    argmax_transitions = {h: {} for h in ('5m', '10m')}
    for h in ('5m', '10m'):
        for stage in labels[h]:
            metric = _metrics(labels[h][stage], probabilities[h][stage])
            if metric is not None:
                settled_metrics[h][stage] = metric
        for target_stage, pairs in transition_pairs[h].items():
            matrix = {src: {dst: 0 for dst in CLASSES} for src in CLASSES}
            for src, dst in pairs:
                matrix[src][dst] += 1
            if pairs:
                argmax_transitions[h][f'model_raw_to_{target_stage}'] = matrix
    settled_metrics = {h: v for h, v in settled_metrics.items() if v}
    argmax_transitions = {h: v for h, v in argmax_transitions.items() if v}
    return {'settled_metrics': settled_metrics, 'argmax_transitions': argmax_transitions}


def _stage(obj: dict, key: str):
    value = obj.get(key)
    if isinstance(value, dict):
        try:
            return _argmax([value[c] for c in CLASSES])
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return _argmax(value)
    return None


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
    stage_observations = Counter()
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
                        stage_observations[(h, stage)] += 1
                winner = _argmax(final)
                if winner is not None:
                    counts[h]['final'][winner] += 1
                    stage_observations[(h, 'final')] += 1
        except Exception:
            parse_errors += 1

    report = {
        'ok': parse_errors == 0 and len(rows) > 0,
        'window': len(rows),
        'parse_errors': parse_errors,
        'stage_observations': {f'{h}_{stage}': n for (h, stage), n in sorted(stage_observations.items())},
        **_settled_stage_diagnostics(db_path, window),
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
