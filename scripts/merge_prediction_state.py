from __future__ import annotations

import sqlite3
import sys

if len(sys.argv) != 3:
    raise SystemExit('usage: merge_prediction_state.py LOCAL_DB TARGET_DB')

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
            # horizon is the declared primary key; this table is a latest-state
            # pointer. Preserve the historical behavior that the recovered local
            # snapshot wins for this table; production model artifacts are guarded
            # separately by provenance/integrity checks.
            rows = con.execute(f'SELECT {names} FROM local.{table}').fetchall()
            for row in rows:
                con.execute(
                    f'INSERT OR REPLACE INTO {table} ({names}) VALUES ({placeholders})',
                    row,
                )
            print(f'{table}: replaced={len(rows)}')
            continue

        if table == 'predictions':
            # These columns describe the prediction event at creation time and are
            # immutable. actual_* / correct_* / settled_* fields are explicitly
            # excluded because settlement mutates them later.
            mutable = {
                'actual_price_5m', 'actual_direction_5m', 'correct_5m', 'settled_5m_at_utc',
                'actual_price_10m', 'actual_direction_10m', 'correct_10m', 'settled_10m_at_utc',
            }
            identity_cols = [
                c for c in common
                if c not in {'prediction_id', 'id', 'metric_id'} and c not in mutable
            ]
        else:
            # model_metrics is append-only: every non-ID field belongs to the
            # metric event identity.
            identity_cols = [c for c in common if c not in {'prediction_id', 'id', 'metric_id'}]

        if not identity_cols:
            raise RuntimeError(f'No logical identity columns available for {table}')

        rows = con.execute(f'SELECT {names} FROM local.{table}').fetchall()
        inserted = updated = skipped = 0
        for row in rows:
            row_map = dict(zip(common, row))
            where = ' AND '.join(
                f'(("{c}" = ?) OR ("{c}" IS NULL AND ? IS NULL))'
                for c in identity_cols
            )
            params = []
            for c in identity_cols:
                params.extend((row_map[c], row_map[c]))

            existing = con.execute(
                f'SELECT rowid, {names} FROM {table} WHERE {where} LIMIT 1',
                params,
            ).fetchone()
            if existing:
                skipped += 1
                if table == 'predictions':
                    # Merge only previously-missing settlement fields. Never
                    # overwrite a non-null target settlement with a conflicting
                    # local value: target may contain the later/authoritative
                    # settlement observation. This makes recovery idempotent and
                    # preserves the most complete known state.
                    existing_map = dict(zip(['rowid'] + common, existing))
                    settlement_cols = [
                        c for c in mutable
                        if c in common and existing_map.get(c) is None and row_map.get(c) is not None
                    ]
                    if settlement_cols:
                        set_clause = ','.join(f'"{c}"=?' for c in settlement_cols)
                        values = [row_map[c] for c in settlement_cols] + [existing_map['rowid']]
                        con.execute(f'UPDATE {table} SET {set_clause} WHERE rowid=?', values)
                        updated += 1
                continue

            insert_cols = [c for c in common if c not in {'prediction_id', 'id', 'metric_id'}]
            insert_values = [row_map[c] for c in insert_cols]
            insert_names = ','.join('"' + c + '"' for c in insert_cols)
            insert_placeholders = ','.join('?' for _ in insert_cols)
            try:
                con.execute(
                    f'INSERT INTO {table} ({insert_names}) VALUES ({insert_placeholders})',
                    insert_values,
                )
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
