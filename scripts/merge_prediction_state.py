from __future__ import annotations

import json
import sqlite3
import sys

SETTLEMENT_COLUMNS = (
    'actual_price_5m', 'actual_direction_5m', 'correct_5m', 'settled_5m_at_utc',
    'actual_price_10m', 'actual_direction_10m', 'correct_10m', 'settled_10m_at_utc',
)


def prediction_identity(row):
    raw = row.get('feature_json')
    try:
        feature_key = json.dumps(json.loads(raw), sort_keys=True, separators=(',', ':'))
    except (TypeError, ValueError, json.JSONDecodeError):
        feature_key = str(raw)
    return (
        str(row.get('created_at_utc')),
        str(row.get('target_5m')),
        str(row.get('target_10m')),
        feature_key,
    )


def compact_predictions(con):
    """Collapse duplicate prediction events and preserve settlement state."""
    info = con.execute('PRAGMA table_info(predictions)').fetchall()
    if not info:
        return {}
    cols = [r[1] for r in info]
    names = ','.join('"' + c + '"' for c in cols)
    target_rows = con.execute(f'SELECT rowid, {names} FROM predictions').fetchall()
    groups = {}
    for item in target_rows:
        item_map = dict(zip(['rowid'] + cols, item))
        groups.setdefault(prediction_identity(item_map), []).append(item_map)

    index = {}
    compacted = 0
    for key, group in groups.items():
        survivor = max(group, key=settlement_score)
        duplicate_ids = []
        for candidate in group:
            if candidate['rowid'] == survivor['rowid']:
                continue
            duplicate_ids.append(candidate['rowid'])
            for c in SETTLEMENT_COLUMNS:
                if survivor.get(c) is None and candidate.get(c) is not None:
                    survivor[c] = candidate[c]
        settlement_updates = {c: survivor[c] for c in SETTLEMENT_COLUMNS if survivor.get(c) is not None}
        if settlement_updates:
            set_clause = ','.join(f'"{c}"=?' for c in settlement_updates)
            values = list(settlement_updates.values()) + [survivor['rowid']]
            con.execute(f'UPDATE predictions SET {set_clause} WHERE rowid=?', values)
        if duplicate_ids:
            placeholders = ','.join('?' for _ in duplicate_ids)
            con.execute(f'DELETE FROM predictions WHERE rowid IN ({placeholders})', duplicate_ids)
            compacted += len(duplicate_ids)
        index[key] = survivor
    print(f'predictions: compacted_duplicates={compacted} total_events={len(index)}')
    return index


def settlement_score(row):
    present = sum(row.get(c) is not None for c in SETTLEMENT_COLUMNS)
    latest = max(str(row.get(c) or '') for c in ('settled_5m_at_utc', 'settled_10m_at_utc'))
    return (present, latest, -int(row['rowid']))

if len(sys.argv) == 3 and sys.argv[1] == '--compact':
    target_path = sys.argv[2]
    con = sqlite3.connect(target_path)
    try:
        compact_predictions(con)
        con.commit()
    finally:
        con.close()
    print('prediction-state compaction: OK')
    raise SystemExit(0)

if len(sys.argv) != 3:
    raise SystemExit('usage: merge_prediction_state.py LOCAL_DB TARGET_DB | --compact TARGET_DB')

local_path, target_path = sys.argv[1:]

# Conflict recovery can run this merge repeatedly. Auto-increment IDs are not
# logical identity. Prediction settlement fields are mutable, so they must not
# participate in duplicate detection; otherwise a prediction that was settled
# on one side and unsettled on the other can be inserted twice.
con = sqlite3.connect(target_path)
con.execute('PRAGMA foreign_keys=ON')
con.execute('ATTACH DATABASE ? AS local', (local_path,))

try:
    for table in ('predictions', 'model_metrics', 'model_registry'):
        target_info = con.execute(f'PRAGMA table_info({table})').fetchall()
        local_info = con.execute(f'PRAGMA local.table_info({table})').fetchall()
        target_cols = [r[1] for r in target_info]
        local_cols = [r[1] for r in local_info]
        common = [c for c in target_cols if c in local_cols]
        if not common:
            continue

        names = ','.join('"' + c + '"' for c in common)
        placeholders = ','.join('?' for _ in common)

        if table == 'model_registry':
            # This table is a latest-state pointer. During a push conflict the
            # local checkout may contain an older registry snapshot than the
            # origin checkout. Never roll production back merely because the
            # local state won the database merge. Prefer the row with the newest
            # valid updated_at_utc; if timestamps are unavailable, preserve the
            # target row rather than guessing.
            target_rows = con.execute(f'SELECT {names} FROM {table}').fetchall()
            local_rows = con.execute(f'SELECT {names} FROM local.{table}').fetchall()
            index = {row[common.index('horizon')]: row for row in target_rows} if 'horizon' in common else {}
            updated_idx = common.index('updated_at_utc') if 'updated_at_utc' in common else None
            for row in local_rows:
                key = row[common.index('horizon')] if 'horizon' in common else None
                current = index.get(key)
                if current is None:
                    con.execute(
                        f'INSERT OR REPLACE INTO {table} ({names}) VALUES ({placeholders})',
                        row,
                    )
                    index[key] = row
                    continue
                if updated_idx is None:
                    continue
                local_ts = str(row[updated_idx] or '')
                target_ts = str(current[updated_idx] or '')
                if local_ts and (not target_ts or local_ts > target_ts):
                    con.execute(
                        f'INSERT OR REPLACE INTO {table} ({names}) VALUES ({placeholders})',
                        row,
                    )
                    index[key] = row
            print(f'{table}: target={len(target_rows)} local={len(local_rows)} latest-timestamp-wins')
            continue

        if table == 'predictions':
            # Compact the target first, then merge local events using the exact same
            # identity contract as the research duplicate audit.
            index = compact_predictions(con)
            local_rows = con.execute(f'SELECT {names} FROM local.{table}').fetchall()
            inserted = updated = skipped = 0
            for row in local_rows:
                row_map = dict(zip(common, row))
                key = prediction_identity(row_map)
                existing_map = index.get(key)
                if existing_map:
                    skipped += 1
                    settlement_cols = [
                        c for c in SETTLEMENT_COLUMNS
                        if c in common and existing_map.get(c) is None and row_map.get(c) is not None
                    ]
                    if settlement_cols:
                        set_clause = ','.join(f'\"{c}\"=?' for c in settlement_cols)
                        values = [row_map[c] for c in settlement_cols] + [existing_map['rowid']]
                        con.execute(f'UPDATE {table} SET {set_clause} WHERE rowid=?', values)
                        for c in settlement_cols:
                            existing_map[c] = row_map[c]
                        updated += 1
                    continue

                insert_cols = [c for c in common if c not in {'prediction_id', 'id', 'metric_id'}]
                insert_values = [row_map[c] for c in insert_cols]
                insert_names = ','.join('\"' + c + '\"' for c in insert_cols)
                insert_placeholders = ','.join('?' for _ in insert_cols)
                try:
                    cursor = con.execute(
                        f'INSERT INTO {table} ({insert_names}) VALUES ({insert_placeholders})',
                        insert_values,
                    )
                    row_map['rowid'] = cursor.lastrowid
                    index[key] = row_map
                    inserted += 1
                except sqlite3.IntegrityError as exc:
                    message = str(exc).lower()
                    if 'unique' in message or 'constraint' in message:
                        skipped += 1
                    else:
                        raise

            print(f'{table}: inserted={inserted} updated={updated} skipped_existing={skipped}')
            continue

        if not identity_cols:
            raise RuntimeError(f'No logical identity columns available for {table}')

        rows = con.execute(f'SELECT {names} FROM local.{table}').fetchall()
        inserted = updated = skipped = 0
        for row in rows:
            row_map = dict(zip(common, row))
            where = ' AND '.join(
                f'((\"{c}\" = ?) OR (\"{c}\" IS NULL AND ? IS NULL))'
                for c in identity_cols
            )
            params = []
            for c in identity_cols:
                params.extend((row_map[c], row_map[c]))
            existing = con.execute(f'SELECT rowid, {names} FROM {table} WHERE {where} LIMIT 1', params).fetchone()
            if existing:
                skipped += 1
                continue
            insert_cols = [c for c in common if c not in {'prediction_id', 'id', 'metric_id'}]
            insert_values = [row_map[c] for c in insert_cols]
            insert_names = ','.join('\"' + c + '\"' for c in insert_cols)
            insert_placeholders = ','.join('?' for _ in insert_cols)
            try:
                con.execute(f'INSERT INTO {table} ({insert_names}) VALUES ({insert_placeholders})', insert_values)
                inserted += 1
            except sqlite3.IntegrityError as exc:
                message = str(exc).lower()
                if 'unique' in message or 'constraint' in message:
                    skipped += 1
                else:
                    raise

        print(f'{table}: inserted={inserted} updated={updated} skipped_existing={skipped}')

    con.commit()
finally:
    try:
        con.execute('DETACH DATABASE local')
    finally:
        con.close()

print('prediction-state merge: OK')
