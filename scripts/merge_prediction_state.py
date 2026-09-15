from __future__ import annotations

import sqlite3
import sys

if len(sys.argv) != 3:
    raise SystemExit('usage: merge_prediction_state.py LOCAL_DB TARGET_DB')

local_path, target_path = sys.argv[1:]

# Merge is intentionally idempotent: conflict recovery may execute this script
# repeatedly, and the same local snapshot must never create duplicate logical
# prediction/metric rows. Auto-increment IDs are not used as identity because
# the two databases can allocate different IDs for the same logical record.
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
            # horizon is the declared primary key; this table represents the
            # latest production pointer, so the local state intentionally wins.
            rows = con.execute(f'SELECT {names} FROM local.{table}').fetchall()
            for row in rows:
                con.execute(
                    f'INSERT OR REPLACE INTO {table} ({names}) VALUES ({placeholders})',
                    row,
                )
            continue

        identity_cols = [
            c for c in common
            if c not in {'prediction_id', 'id', 'metric_id'}
        ]
        if not identity_cols:
            raise RuntimeError(f'No logical identity columns available for {table}')

        identity_names = ','.join('"' + c + '"' for c in identity_cols)
        # NULL-safe equality is important if optional outcome fields are part
        # of a future schema. Current prediction/metric identity fields are
        # non-NULL, but IS provides deterministic behavior across migrations.
        predicate = ' AND '.join(
            f'(t."{c}" IS l."{c}")' for c in identity_cols
        )

        rows = con.execute(f'SELECT {names} FROM local.{table}').fetchall()
        inserted = 0
        skipped = 0
        for row in rows:
            # Exclude the auto-increment ID from the inserted values. SQLite
            # allocates a target-local ID, avoiding collisions after branches
            # diverge and making repeated recovery merges safe.
            values = list(row)
            insert_cols = list(common)
            for id_col in ('prediction_id', 'id', 'metric_id'):
                if id_col in insert_cols:
                    idx = insert_cols.index(id_col)
                    insert_cols.pop(idx)
                    values.pop(idx)
                    break

            insert_names = ','.join('"' + c + '"' for c in insert_cols)
            insert_placeholders = ','.join('?' for _ in insert_cols)
            exists = con.execute(
                f'SELECT 1 FROM {table} AS t '
                f'WHERE EXISTS (SELECT 1 FROM local.{table} AS l '
                f'WHERE {predicate} LIMIT 1) LIMIT 1',
                row,
            ).fetchone()
            if exists:
                skipped += 1
                continue
            try:
                con.execute(
                    f'INSERT INTO {table} ({insert_names}) VALUES ({insert_placeholders})',
                    values,
                )
                inserted += 1
            except sqlite3.IntegrityError as exc:
                # A genuine schema-level uniqueness collision is safe to skip;
                # other errors must fail closed instead of being misreported as
                # a successful merge.
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
