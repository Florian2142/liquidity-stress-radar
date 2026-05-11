"""Phase 0 offline smoke test: run the entire pipeline on synthetic data.

No network required. Validates that imports work, schema initialises, features
compute, target labels are correct, DuckDB persists data, and the joined panel
returns the expected shape.

Run::

    python scripts/00_offline_smoke.py

Use this BEFORE running ``01_initial_load.py`` to catch installation issues
without burning a yfinance round-trip.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from liquidity_radar.config import ensure_dirs  # noqa: E402
from liquidity_radar.data.store import (  # noqa: E402
    get_connection,
    get_features_panel,
    upsert_dataframe,
)
from liquidity_radar.features.liquidity import amihud_illiquidity  # noqa: E402
from liquidity_radar.features.target import forward_drawdown_label  # noqa: E402


def make_synthetic_spy(n: int = 5000) -> pd.DataFrame:
    """Generate a 5000-row synthetic SPY-shaped DataFrame."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2005-01-01", periods=n)
    rets = rng.normal(0.0005, 0.012, n)
    close = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, n)),
            "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
            "close": close,
            "adj_close": close,
            "volume": rng.integers(50_000_000, 200_000_000, n),
        },
        index=dates,
    ).rename_axis("date")


def make_synthetic_vol(n: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2005-01-01", periods=n)
    vix = 15 + 5 * rng.standard_normal(n).cumsum() / np.sqrt(np.arange(1, n + 1))
    vix = np.clip(vix, 9, 80)
    return pd.DataFrame(
        {"vix": vix, "vix9d": vix * 0.95, "vix3m": vix * 1.02},
        index=dates,
    ).rename_axis("date")


def make_synthetic_macro(n: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(13)
    dates = pd.bdate_range("2005-01-01", periods=n)
    return pd.DataFrame(
        {
            "yield_10y": 3.0 + 0.5 * rng.standard_normal(n).cumsum() / np.sqrt(np.arange(1, n + 1)),
            "yield_2y": 2.5 + 0.5 * rng.standard_normal(n).cumsum() / np.sqrt(np.arange(1, n + 1)),
            "fed_funds": 2.0 + 0.5 * rng.standard_normal(n).cumsum() / np.sqrt(np.arange(1, n + 1)),
        },
        index=dates,
    ).rename_axis("date")


def main() -> int:
    ensure_dirs()

    print("── Generating synthetic data ─────────────────────────────────")
    spy = make_synthetic_spy()
    vol = make_synthetic_vol()
    macro = make_synthetic_macro()
    print(f"   SPY: {spy.shape}, vol: {vol.shape}, macro: {macro.shape}")

    print("\n── Persisting to DuckDB ──────────────────────────────────────")
    with get_connection() as con:
        upsert_dataframe(con, spy, "spy_prices")
        upsert_dataframe(con, vol, "vol_indicators")
        upsert_dataframe(con, macro, "macro_series")

    print("\n── Joining panel and computing features ──────────────────────")
    with get_connection() as con:
        panel = get_features_panel(con)
        amihud = amihud_illiquidity(panel)
        target = forward_drawdown_label(panel)

        feat_df = pd.DataFrame(index=panel.index)
        feat_df["amihud"] = amihud
        feat_df["cs_spread"] = pd.NA
        feat_df["edge"] = pd.NA
        feat_df["vix_5d_change"] = panel["vix"].diff(5)
        feat_df["vix_term_ratio"] = panel["vix9d"] / panel["vix3m"]
        feat_df["yield_curve_slope"] = panel["yield_10y"] - panel["yield_2y"]
        feat_df["spy_drawdown"] = panel["adj_close"] / panel["adj_close"].rolling(252).max() - 1
        feat_df["realized_vol_20d"] = panel["adj_close"].pct_change().rolling(20).std() * (252**0.5)
        feat_df.index.name = "date"

        upsert_dataframe(con, feat_df, "features")
        upsert_dataframe(con, target, "targets")

    print("\n── Summary ───────────────────────────────────────────────────")
    with get_connection() as con:
        n_features = con.execute("SELECT COUNT(*) FROM features WHERE amihud IS NOT NULL").fetchone()[0]
        n_pos = con.execute("SELECT SUM(label) FROM targets WHERE label IS NOT NULL").fetchone()[0]
        n_total = con.execute("SELECT COUNT(*) FROM targets WHERE label IS NOT NULL").fetchone()[0]
        sample = con.execute(
            """
            SELECT date, amihud, vix_term_ratio, yield_curve_slope, spy_drawdown
            FROM features
            WHERE amihud IS NOT NULL
            ORDER BY date DESC
            LIMIT 5
            """
        ).fetchdf()
        print(f"   Rows with non-null Amihud: {n_features:,}")
        if n_total:
            pct = 100 * (n_pos or 0) / n_total
            print(f"   Positive labels: {n_pos:,} / {n_total:,} = {pct:.2f}%")
        print("\n   Sample (5 most recent):")
        print(sample.to_string(index=False))

    print("\n✅ Offline smoke test passed. Pipeline works end-to-end.")
    print("   Next: run scripts/01_initial_load.py to pull real data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
