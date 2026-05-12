"""DuckDB schema. Run :func:`init_schema` once; safe to re-run."""

from __future__ import annotations

import duckdb

CREATE_STATEMENTS: list[str] = [
    # SPY daily prices and volume
    """
    CREATE TABLE IF NOT EXISTS spy_prices (
        date DATE PRIMARY KEY,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        adj_close DOUBLE,
        volume BIGINT
    )
    """,
    # Volatility indicators (one row per date, multiple series as columns)
    """
    CREATE TABLE IF NOT EXISTS vol_indicators (
        date DATE PRIMARY KEY,
        vix DOUBLE,
        vix9d DOUBLE,
        vix3m DOUBLE
    )
    """,
    # Macro series (Treasury yields, FFR)
    """
    CREATE TABLE IF NOT EXISTS macro_series (
        date DATE PRIMARY KEY,
        yield_10y DOUBLE,
        yield_2y DOUBLE,
        fed_funds DOUBLE
    )
    """,
    # Computed features (one row per date; columns added per phase)
    """
    CREATE TABLE IF NOT EXISTS features (
        date DATE PRIMARY KEY,
        amihud DOUBLE,
        amihud_zscore DOUBLE,
        amihud_5d_change DOUBLE,
        cs_spread DOUBLE,
        edge DOUBLE,
        vix_5d_change DOUBLE,
        vix_term_ratio DOUBLE,
        yield_curve_slope DOUBLE,
        spy_drawdown DOUBLE,
        realized_vol_20d DOUBLE
    )
    """,
    # Target labels
    """
    CREATE TABLE IF NOT EXISTS targets (
        date DATE PRIMARY KEY,
        forward_drawdown_pct DOUBLE,
        label INTEGER  -- 1 if drawdown >= TARGET_DRAWDOWN_PCT in next TARGET_HORIZON_DAYS
    )
    """,
    # Predictions (one row per date × model)
    """
    CREATE TABLE IF NOT EXISTS predictions (
        date DATE,
        model VARCHAR,
        prob DOUBLE,
        fold INTEGER,
        PRIMARY KEY (date, model)
    )
    """,
]


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create all tables. Safe to call repeatedly."""
    for stmt in CREATE_STATEMENTS:
        con.execute(stmt)
    # Migrate: add new Amihud variant columns when upgrading an existing DB.
    for col in ("amihud_zscore", "amihud_5d_change"):
        try:
            con.execute(f"ALTER TABLE features ADD COLUMN IF NOT EXISTS {col} DOUBLE")
        except Exception:
            pass


def list_tables(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Return user table names in the connected database."""
    rows = con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
    return sorted(r[0] for r in rows)
