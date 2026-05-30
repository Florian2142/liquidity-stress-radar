"""Smoke tests for the data layer. Run with: pytest tests/ -v"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from liquidity_radar.data.schema import init_schema, list_tables  # noqa: E402
from liquidity_radar.data.store import get_connection, upsert_dataframe  # noqa: E402
from liquidity_radar.features.liquidity import amihud_illiquidity  # noqa: E402
from liquidity_radar.features.target import forward_drawdown_label  # noqa: E402


def test_schema_creates_all_tables(tmp_path, monkeypatch):
    """init_schema creates all expected tables and is idempotent."""
    db_file = tmp_path / "test.duckdb"
    monkeypatch.setattr("liquidity_radar.config.DB_PATH", db_file)
    monkeypatch.setattr("liquidity_radar.data.store.DB_PATH", db_file)

    import duckdb

    con = duckdb.connect(str(db_file))
    init_schema(con)
    init_schema(con)  # idempotency

    tables = list_tables(con)
    expected = {
        "spy_prices",
        "vol_indicators",
        "macro_series",
        "features",
        "targets",
        "predictions",
    }
    assert expected.issubset(set(tables)), f"missing tables: {expected - set(tables)}"
    con.close()


def test_amihud_basic_shape():
    """Amihud returns a Series with the expected number of leading NaNs."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    df = pd.DataFrame(
        {
            "adj_close": 100 + pd.Series(range(100), index=dates) * 0.5,
            "volume": 1_000_000,
        },
        index=dates,
    )
    out = amihud_illiquidity(df, window=20)
    assert isinstance(out, pd.Series)
    assert len(out) == 100
    # First 20 rows are NaN (window not yet full)
    assert out.iloc[:19].isna().all()
    assert out.iloc[20:].notna().all()


def test_forward_drawdown_label_logic():
    """Synthetic series with a known 10% drop produces the expected label."""
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    px = pd.Series(100.0, index=dates)
    # Inject a 10% drop at day 30
    px.iloc[30:] = 90.0
    df = pd.DataFrame({"adj_close": px}, index=dates)

    out = forward_drawdown_label(df, horizon=20, threshold=0.05)
    # Day 25's forward window (days 25-44) includes the drop at day 30.
    assert out.loc[dates[25], "label"] == 1
    # Day 15's forward window (days 15-34) also includes the drop at day 30.
    assert out.loc[dates[15], "label"] == 1
    # Day 5's forward window (days 5-24) does NOT include the drop.
    assert out.loc[dates[5], "label"] == 0


def test_upsert_idempotent(tmp_path, monkeypatch):
    """Upserting the same frame twice doesn't duplicate rows."""
    db_file = tmp_path / "test.duckdb"
    monkeypatch.setattr("liquidity_radar.config.DB_PATH", db_file)
    monkeypatch.setattr("liquidity_radar.data.store.DB_PATH", db_file)

    df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.05, 2.05],
            "adj_close": [1.05, 2.05],
            "volume": [100, 200],
        },
        index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
    )
    df.index.name = "date"

    with get_connection() as con:
        upsert_dataframe(con, df, "spy_prices")
        upsert_dataframe(con, df, "spy_prices")
        n = con.execute("SELECT COUNT(*) FROM spy_prices").fetchone()[0]
    assert n == 2
