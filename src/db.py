import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "data" / "predictions.db"

def connect():
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with connect() as con:
        con.executescript('''
        CREATE TABLE IF NOT EXISTS predictions (
          prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at_utc TEXT NOT NULL,
          target_5m TEXT NOT NULL,
          target_10m TEXT NOT NULL,
          base_price REAL NOT NULL,
          p_up_5m REAL NOT NULL, p_down_5m REAL NOT NULL, p_flat_5m REAL NOT NULL,
          p_up_10m REAL NOT NULL, p_down_10m REAL NOT NULL, p_flat_10m REAL NOT NULL,
          model_version TEXT NOT NULL,
          feature_json TEXT NOT NULL,
          scenario_json TEXT NOT NULL,
          actual_price_5m REAL, actual_price_10m REAL,
          actual_direction_5m TEXT, actual_direction_10m TEXT,
          correct_5m INTEGER, correct_10m INTEGER,
          settled_5m_at_utc TEXT, settled_10m_at_utc TEXT,
          settlement_source_5m TEXT, settlement_source_10m TEXT
        );
        CREATE TABLE IF NOT EXISTS model_metrics (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          evaluated_at_utc TEXT NOT NULL,
          horizon TEXT NOT NULL,
          model_version TEXT NOT NULL,
          n INTEGER NOT NULL,
          accuracy REAL,
          logloss REAL,
          brier REAL,
          calibration_error REAL
        );
        CREATE TABLE IF NOT EXISTS model_registry (
          horizon TEXT PRIMARY KEY,
          production_version TEXT NOT NULL,
          updated_at_utc TEXT NOT NULL
        );
        ''')
        columns = {str(row[1]) for row in con.execute('PRAGMA table_info(predictions)').fetchall()}
        for name in ('settlement_source_5m', 'settlement_source_10m'):
            if name not in columns:
                con.execute(f'ALTER TABLE predictions ADD COLUMN {name} TEXT')

if __name__ == '__main__':
    init_db()
    print(DB)
