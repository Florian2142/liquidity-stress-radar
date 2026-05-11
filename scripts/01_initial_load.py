"""Phase 1 smoke test: pull all data, compute Amihud + target, store in DuckDB, print summary.

Run::

    python scripts/01_initial_load.py

Expected runtime: 1–3 minutes, depending on yfinance latency.

Success criterion: the script exits 0 with a summary table showing 7,500+ rows
of joined data and a non-trivial fraction of positive labels.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Allow `python scripts/...` invocation from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from liquidity_radar.config import DB_PATH, ensure_dirs  # noqa: E402
from liquidity_radar.data.ingest import (  # noqa: E402
    fetch_macro,
    fetch_spy,
    fetch_vol_indicators,
)
from liquidity_radar.data.store import (  # noqa: E402
    get_connection,
    get_features_panel,
    upsert_dataframe,
)
from liquidity_radar.features.liquidity import amihud_illiquidity  # noqa: E402
from liquidity_radar.features.target import forward_drawdown_label  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
logger = logging.getLogger("01_initial_load")


def step(label: str) -> float:
    """Print a step header and return the start time."""
    print(f"\n── {label} " + "─" * (70 - len(label)))
    return time.time()


def done(t0: float) -> None:
    print(f"   ✓ done in {time.time() - t0:.1f}s")


def main() -> int:
    ensure_dirs()
    print(f"DuckDB path: {DB_PATH}")

    # ── 1. Fetch SPY ─────────────────────────────────────────────────────
    t0 = step("Fetching SPY from yfinance")
    spy = fetch_spy()
    print(f"   {len(spy):,} rows from {spy.index.min().date()} to {spy.index.max().date()}")
    done(t0)

    # ── 2. Fetch volatility indicators ───────────────────────────────────
    t0 = step("Fetching VIX, VIX9D, VIX3M from yfinance")
    vol = fetch_vol_indicators()
    print(f"   {len(vol):,} rows; columns: {list(vol.columns)}")
    done(t0)

    # ── 3. Fetch macro from FRED ─────────────────────────────────────────
    t0 = step("Fetching macro series from FRED")
    macro = fetch_macro()
    print(f"   {len(macro):,} rows; columns: {list(macro.columns)}")
    done(t0)

    # ── 4. Persist to DuckDB ─────────────────────────────────────────────
    t0 = step("Writing base tables to DuckDB")
    with get_connection() as con:
        n_spy = upsert_dataframe(con, spy, "spy_prices")
        n_vol = upsert_dataframe(con, vol, "vol_indicators")
        n_macro = upsert_dataframe(con, macro, "macro_series")
        print(f"   spy_prices: {n_spy:,} rows; vol_indicators: {n_vol:,}; macro_series: {n_macro:,}")
    done(t0)

    # ── 5. Build features panel + compute Amihud + target ────────────────
    t0 = step("Building joined panel + computing Amihud + target")
    with get_connection() as con:
        panel = get_features_panel(con)
        print(f"   Panel: {len(panel):,} rows × {panel.shape[1]} cols")

        amihud = amihud_illiquidity(panel)
        target = forward_drawdown_label(panel)

        # Persist a minimal feature row (just amihud for now; rest filled in Phase 2)
        feat_df = pd.DataFrame(index=panel.index)
        feat_df["amihud"] = amihud
        feat_df["cs_spread"] = pd.NA
        feat_df["edge"] = pd.NA
        feat_df["vix_5d_change"] = panel["vix"].diff(5)
        feat_df["vix_term_ratio"] = panel["vix9d"] / panel["vix3m"]
        feat_df["yield_curve_slope"] = panel["yield_10y"] - panel["yield_2y"]
        feat_df["spy_drawdown"] = (
            panel["adj_close"] / panel["adj_close"].rolling(252).max() - 1
        )
        feat_df["realized_vol_20d"] = panel["adj_close"].pct_change().rolling(20).std() * (252**0.5)
        feat_df.index.name = "date"

        upsert_dataframe(con, feat_df, "features")
        upsert_dataframe(con, target, "targets")
    done(t0)

    # ── 6. Summary ────────────────────────────────────────────────────────
    t0 = step("Summary")
    with get_connection() as con:
        n_features = con.execute("SELECT COUNT(*) FROM features WHERE amihud IS NOT NULL").fetchone()[0]
        n_pos = con.execute("SELECT SUM(label) FROM targets WHERE label IS NOT NULL").fetchone()[0]
        n_total = con.execute("SELECT COUNT(*) FROM targets WHERE label IS NOT NULL").fetchone()[0]
        latest = con.execute("SELECT MAX(date) FROM spy_prices").fetchone()[0]
        sample = con.execute(
            """
            SELECT date, amihud, vix_term_ratio, yield_curve_slope, spy_drawdown
            FROM features
            WHERE amihud IS NOT NULL
            ORDER BY date DESC
            LIMIT 5
            """
        ).fetchdf()

        print(f"   Latest SPY date: {latest}")
        print(f"   Rows with non-null Amihud: {n_features:,}")
        if n_total:
            pos_pct = 100 * (n_pos or 0) / n_total
            print(f"   Positive labels: {n_pos:,} / {n_total:,} = {pos_pct:.2f}%")
        print("\n   Sample (5 most recent):")
        print(sample.to_string(index=False))
    done(t0)

    print("\n✅ Phase 1 smoke test complete. Next: implement Corwin-Schultz + EDGE in features/liquidity.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
