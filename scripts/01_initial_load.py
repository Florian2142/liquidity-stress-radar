"""Refresh all market data: pull SPY, VIX term structure, and FRED macro into DuckDB.

Run::

    python scripts/01_initial_load.py

Expected runtime: 1–3 minutes, depending on yfinance latency. Each external
source is fetched independently — a transient FRED outage still yields a usable
price/VIX panel and is reported rather than aborting the run. On completion a
data-quality report summarises coverage, gaps, and freshness.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Allow `python scripts/...` invocation from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from liquidity_radar.config import DB_PATH, ensure_dirs  # noqa: E402
from liquidity_radar.data.ingest import fetch_market_panel  # noqa: E402
from liquidity_radar.data.quality import build_quality_report  # noqa: E402
from liquidity_radar.data.store import (  # noqa: E402
    get_connection,
    get_features_panel,
    upsert_dataframe,
)
from liquidity_radar.features.build import build_feature_matrix  # noqa: E402
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

    # ── 1. Fetch all sources (failures recorded, not fatal) ──────────────
    t0 = step("Fetching SPY, VIX term structure, and FRED macro")
    panel, status = fetch_market_panel()
    for source, state in status.items():
        flag = "ok" if state == "ok" else "FAILED"
        print(f"   {source:<6} {flag}: {state if state != 'ok' else ''}".rstrip())
    print(
        f"   Panel: {len(panel):,} rows from {panel.index.min().date()} to {panel.index.max().date()}"
    )
    done(t0)

    # ── 2. Persist base tables to DuckDB ─────────────────────────────────
    t0 = step("Writing base tables to DuckDB")
    spy_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    vol_cols = [c for c in ["vix", "vix9d", "vix3m"] if c in panel.columns]
    macro_cols = [c for c in ["yield_10y", "yield_2y", "fed_funds"] if c in panel.columns]
    with get_connection() as con:
        n_spy = upsert_dataframe(con, panel[spy_cols], "spy_prices")
        n_vol = upsert_dataframe(con, panel[vol_cols], "vol_indicators") if vol_cols else 0
        n_macro = upsert_dataframe(con, panel[macro_cols], "macro_series") if macro_cols else 0
        print(f"   spy_prices: {n_spy:,}; vol_indicators: {n_vol:,}; macro_series: {n_macro:,}")
    done(t0)

    # ── 3. Build features + target and persist ───────────────────────────
    t0 = step("Computing features + drawdown target")
    with get_connection() as con:
        joined = get_features_panel(con)
        features_full = build_feature_matrix(joined, include_raw_amihud=True)
        target = forward_drawdown_label(joined)
        upsert_dataframe(con, features_full, "features")
        upsert_dataframe(con, target, "targets")
    done(t0)

    # ── 4. Data-quality report ───────────────────────────────────────────
    t0 = step("Data-quality report")
    report = build_quality_report(joined)
    print(f"   Rows:            {report.n_rows:,}")
    print(f"   Coverage:        {report.start_date.date()} → {report.end_date.date()}")
    print(f"   Duplicate dates: {report.n_duplicate_dates}")
    print(f"   Calendar gaps:   {report.n_calendar_gaps} (max {report.max_gap_days} bdays)")
    print(
        f"   Freshness:       {report.stale_days} days old "
        f"({'fresh' if report.is_fresh else 'STALE'})"
    )
    n_pos = int(target["label"].sum())
    n_total = int(target["label"].notna().sum())
    if n_total:
        print(f"   Positive labels: {n_pos:,} / {n_total:,} = {100 * n_pos / n_total:.2f}%")
    done(t0)

    print("\nData refresh complete. Next: python scripts/02_train_logistic.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
