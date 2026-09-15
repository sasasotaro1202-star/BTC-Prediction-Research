from __future__ import annotations

import sqlite3
import sys

if len(sys.argv) != 3:
    raise SystemExit('usage: merge_prediction_state.py LOCAL_DB TARGET_DB')

local_path, target_path = sys.argv[1:]

# Conflict recovery can run this merge repeatedly. Auto-increment IDs are not
# logical identity, so prediction/metric rows are matched on all non-ID fields.
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
            # pointer, so the local snapshot intentionally wins on recovery.
            rows = con.execute(f'SELECT {names} FROM local.{table}').fetchall()
            for row in rows:
                con.execute(
                    f'INSERT OR REPLACE INTO {table} ({names}) VALUES ({placeholders})',
                    row,
                )
            print(f'{table}: replaced={len(rows)}')
            continue

        identity_cols = [
            c for c in common
            if c not in {'prediction_id', 'id', 'metric_id'}
        ]
        if not identity_cols:
            raise RuntimeError(f'No logical identity columns available for {table}')

        rows = con.execute(f'SELECT {names} FROM local.{table}').fetchall()
        inserted = 0
        skipped = 0
        for row in rows:
            row_map = dict(zip(common, row))
            # NULL-safe equality, evaluated against this specific local row.
            where = ' AND '.join(
                f'(("{c}" = ?) OR ("{c}" IS NULL AND ? IS NULL))'
                for c in identity_cols
            )
            params = []
            for c in identity_cols:
                params.extend((row_map[c], row_map[c]))

            exists = con.execute(
                f'SELECT 1 FROM {table} WHERE {where} LIMIT 1',
                params,
            ).fetchone()
            if exists:
                skipped += 1
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

        print(f'{table}: inserted={inserted} skipped_existing={skipped}')

    con.commit()
finally:
    try:
        con.execute('DETACH DATABASE local')
    finally:
        con.close()

print('prediction-state merge: OK')
