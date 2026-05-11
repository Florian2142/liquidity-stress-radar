"""DuckDB I/O. Connection management and upsert helpers."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import duckdb
import pandas as pd

from liquidity_radar.config import DB_PATH, ensure_dirs
from liquidity_radar.data.schema import init_schema

logger = logging.getLogger(__name__)


@contextmanager
def get_connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a DuckDB connection with the schema initialised.

    Use as a context manager so the connection always closes::

        with get_connection() as con:
            con.execute("SELECT * FROM spy_prices LIMIT 5")
    """
    ensure_dirs()
    con = duckdb.connect(str(DB_PATH))
    try:
        init_schema(con)
        yield con
    finally:
        con.close()


def upsert_dataframe(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    table: str,
    key: str = "date",
) -> int:
    """Upsert a DataFrame into a DuckDB table on the given primary key.

    DuckDB's ``INSERT OR REPLACE`` requires a temp registration. We use
    ``register`` to expose ``df`` as a virtual table, then run ``INSERT OR REPLACE``.
    Returns the number of rows attempted.
    """
    if df.empty:
        logger.warning("upsert_dataframe: empty frame for table=%s; skipping", table)
        return 0

    df = df.copy()
    if df.index.name == key:
        df = df.reset_index()

    # Make sure columns match the destination table column order.
    # PRAGMA table_info returns (cid, name, type, notnull, dflt_value, pk) — name is index 1.
    table_cols = [r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()]
    missing = set(table_cols) - set(df.columns)
    for col in missing:
        df[col] = None
    df = df[table_cols]

    con.register("df_tmp", df)
    con.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM df_tmp")
    con.unregister("df_tmp")
    return len(df)


def get_features_panel(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Return a single DataFrame indexed by date with all base series joined.

    Used as the input to feature engineering. Forward-fills macro series at most
    1 trading day to handle weekends and holidays.
    """
    query = """
    SELECT
        s.date,
        s.open, s.high, s.low, s.close, s.adj_close, s.volume,
        v.vix, v.vix9d, v.vix3m,
        m.yield_10y, m.yield_2y, m.fed_funds
    FROM spy_prices s
    LEFT JOIN vol_indicators v ON s.date = v.date
    LEFT JOIN macro_series m ON s.date = m.date
    ORDER BY s.date
    """
    df = con.execute(query).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Forward-fill macro and vol series at most 1 day (handles holidays)
    macro_cols = ["vix", "vix9d", "vix3m", "yield_10y", "yield_2y", "fed_funds"]
    df[macro_cols] = df[macro_cols].ffill(limit=1)
    return df
