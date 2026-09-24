from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prediction_identity import (
    SETTLEMENT_COLUMNS,
    SETTLEMENT_TIMESTAMP_COLUMNS,
    canonical_compaction_snapshot,
    canonical_settlement_timestamp,
    prediction_identity,
)


def expected_compacted_event_count(con):
    """Return the number of immutable prediction events before compaction."""
    return int(canonical_compaction_snapshot(con)["unique_event_count"])

def settlement_score(row):
    present = sum(row.get(c) is not None for c in SETTLEMENT_COLUMNS)
    latest = max(
        str(row.get('settled_5m_at_utc') or ''),
        str(row.get('settled_10m_at_utc') or ''),
    )
    return (present, latest, -int(row['rowid']))


def compact_predictions(con):
    """Collapse only exact immutable duplicates after checking settlement conflicts."""
    before_snapshot = canonical_compaction_snapshot(con)
    # canonical_compaction_snapshot now normalizes valid replicated settlement
    # timestamps. Any remaining conflict is either an outcome contradiction or
    # malformed settlement metadata, so both remain fail-closed.
    if before_snapshot["settlement_conflicts"]:
        raise RuntimeError(
            "conflicting or malformed settlement state for immutable prediction event: "
            + "; ".join(before_snapshot["settlement_conflicts"][:10])
        )
    info = con.execute('PRAGMA table_info(predictions)').fetchall()
    if not info:
        return {'compacted_duplicates': 0, 'total_events': 0}
    cols = [r[1] for r in info]
    names = ','.join('"' + c + '"' for c in cols)
    rows = con.execute(f'SELECT rowid, {names} FROM predictions').fetchall()
    groups = {}
    for item in rows:
        row = dict(zip(['rowid'] + cols, item))
        groups.setdefault(prediction_identity(row), []).append(row)

    compacted = 0
    for group in groups.values():
        if len(group) <= 1:
            continue
        survivor = max(group, key=settlement_score)
        duplicate_ids = []
        for candidate in group:
            if candidate['rowid'] == survivor['rowid']:
                continue
            duplicate_ids.append(candidate['rowid'])
            for c in SETTLEMENT_COLUMNS:
                if survivor.get(c) is None and candidate.get(c) is not None:
                    survivor[c] = candidate[c]

        # Canonicalize replicated settlement timestamps deterministically. A
        # later retry may write a newer observation timestamp for the same
        # immutable event; that is not a contradictory market outcome.
        for c in SETTLEMENT_TIMESTAMP_COLUMNS:
            values = [r.get(c) for r in group if r.get(c) is not None]
            if values:
                survivor[c] = canonical_settlement_timestamp(values)
        updates = {c: survivor.get(c) for c in SETTLEMENT_COLUMNS if survivor.get(c) is not None}
        if updates:
            set_clause = ','.join(f'"{c}"=?' for c in updates)
            con.execute(
                f'UPDATE predictions SET {set_clause} WHERE rowid=?',
                [updates[c] for c in updates] + [survivor['rowid']],
            )
        if duplicate_ids:
            placeholders = ','.join('?' for _ in duplicate_ids)
            con.execute(f'DELETE FROM predictions WHERE rowid IN ({placeholders})', duplicate_ids)
            compacted += len(duplicate_ids)

    result = {'compacted_duplicates': compacted, 'total_events': len(groups)}
    print(json.dumps({'prediction-state compaction': result}, sort_keys=True))
    return result


def _merge_mode():
    return len(sys.argv) == 3 and sys.argv[1] == '--compact'


def main():
    if _merge_mode():
        target_path = sys.argv[2]
        con = sqlite3.connect(target_path)
        try:
            result = compact_predictions(con)
            con.commit()
        finally:
            con.close()
        if result['compacted_duplicates'] >= 0:
            print('prediction-state compaction: OK')
        raise SystemExit(0)


    # Conflict recovery can run this merge repeatedly. Auto-increment IDs are not
    # logical identity. Prediction settlement fields are mutable, so they must not
    # participate in duplicate detection; otherwise a prediction that was settled
    # on one side and unsettled on the other can be inserted twice.
    if len(sys.argv) != 3:
        raise SystemExit('usage: merge_prediction_state.py LOCAL_DB TARGET_DB | --compact TARGET_DB')
    local_path, target_path = sys.argv[1:]
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


if __name__ == "__main__":
    main()
