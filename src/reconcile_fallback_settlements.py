"""Reopen legacy fallback settlements that were scored against the wrong venue."""
from __future__ import annotations
import json
import sqlite3
from db import DB, init_db

EXPECTED = {
    "coinbase_fallback": "coinbase_exchange",
    "bybit_fallback": "bybit_linear",
}
FIELDS = {
    "5m": ("actual_price_5m", "actual_direction_5m", "correct_5m", "settled_5m_at_utc", "settlement_source_5m"),
    "10m": ("actual_price_10m", "actual_direction_10m", "correct_10m", "settled_10m_at_utc", "settlement_source_10m"),
}

def main():
    init_db()
    reopened = 0
    with sqlite3.connect(DB) as con:
        rows = con.execute(
            "SELECT prediction_id, scenario_json, settlement_source_5m, settlement_source_10m, actual_price_5m, actual_price_10m FROM predictions"
        ).fetchall()
        for prediction_id, scenario_text, source5, source10, actual5, actual10 in rows:
            try:
                scenario = json.loads(scenario_text or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            expected = EXPECTED.get(str(scenario.get("production_mode", "")))
            if expected is None:
                continue
            for horizon, recorded, actual in (("5m", source5, actual5), ("10m", source10, actual10)):
                if actual is None or recorded == expected:
                    continue
                a, b, c, d, e = FIELDS[horizon]
                con.execute(
                    f"UPDATE predictions SET {a}=NULL,{b}=NULL,{c}=NULL,{d}=NULL,{e}=NULL WHERE prediction_id=?",
                    (prediction_id,),
                )
                reopened += 1
    print(f"fallback_settlement_reopened={reopened}")

if __name__ == "__main__":
    raise SystemExit(main())
