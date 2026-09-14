from __future__ import annotations
import sqlite3
import sys

if len(sys.argv) != 3:
    raise SystemExit('usage: merge_prediction_state.py LOCAL_DB TARGET_DB')

local_path, target_path = sys.argv[1:]
con = sqlite3.connect(target_path)
con.execute('ATTACH DATABASE ? AS local', (local_path,))

for table in ('predictions', 'model_metrics', 'model_registry'):
    target_cols = [r[1] for r in con.execute(f'PRAGMA table_info({table})')]
    local_cols = [r[1] for r in con.execute(f'PRAGMA local.table_info({table})')]
    common = [c for c in target_cols if c in local_cols]
    if not common:
        continue
    names = ','.join('"' + c + '"' for c in common)
    placeholders = ','.join('?' for _ in common)
    rows = con.execute(f'SELECT {names} FROM local.{table}').fetchall()
    for row in rows:
        try:
            if table in ('predictions', 'model_metrics'):
                values = list(row)
                if common and common[0] in ('prediction_id', 'metric_id'):
                    values[0] = None
                con.execute(f'INSERT INTO {table} ({names}) VALUES ({placeholders})', values)
            else:
                con.execute(f'INSERT OR REPLACE INTO {table} ({names}) VALUES ({placeholders})', row)
        except sqlite3.IntegrityError:
            pass

con.commit()
con.execute('DETACH DATABASE local')
con.close()
print('prediction-state merge: OK')
