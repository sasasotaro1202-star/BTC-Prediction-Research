import sqlite3

from src.db import init_connection
from src.horizon_registry import EXTENDED_RESEARCH_HORIZONS


def test_extended_horizon_settlement_columns_use_canonical_names():
    with sqlite3.connect(":memory:") as con:
        init_connection(con)
        columns = {row[1] for row in con.execute("PRAGMA table_info(predictions)")}
    for horizon in EXTENDED_RESEARCH_HORIZONS:
        assert f"settled_{horizon}_at_utc" in columns
        assert f"settlement_source_{horizon}" in columns
        assert f"settled_{horizon}" not in columns
