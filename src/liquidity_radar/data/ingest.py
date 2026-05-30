"""Data ingestion. Pulls from yfinance and FRED, returns clean pandas DataFrames.

These functions never write to disk. They return DataFrames; the caller (typically
``store.upsert_*``) handles persistence.
"""

from __future__ import annotations

import io
import logging
import time

import httpx
import pandas as pd
import yfinance as yf

from liquidity_radar.config import (
    FRED_SERIES,
    SPY_TICKER,
    START_DATE,
    VOL_TICKERS,
)

logger = logging.getLogger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def _retry_yf(ticker: str, start: str, attempts: int = 3, sleep: float = 2.0) -> pd.DataFrame:
    """Pull one ticker from yfinance with retries on empty responses."""
    last_err: Exception | None = None
    for i in range(attempts):
        try:
            df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
            if df is not None and not df.empty:
                # yfinance sometimes returns MultiIndex columns; flatten.
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df
            logger.warning("yfinance empty response for %s, attempt %d/%d", ticker, i + 1, attempts)
        except Exception as e:
            last_err = e
            logger.warning("yfinance error for %s: %s (attempt %d/%d)", ticker, e, i + 1, attempts)
        time.sleep(sleep * (i + 1))
    raise RuntimeError(
        f"Failed to fetch {ticker} from yfinance after {attempts} attempts: {last_err}"
    )


def fetch_spy(start: str = START_DATE) -> pd.DataFrame:
    """Pull SPY daily OHLCV.

    Returns a DataFrame indexed by date with columns:
    ``open, high, low, close, adj_close, volume``.
    """
    df = _retry_yf(SPY_TICKER, start)
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    df = df[["open", "high", "low", "close", "adj_close", "volume"]].copy()
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"
    return df


def fetch_vol_indicators(start: str = START_DATE) -> pd.DataFrame:
    """Pull VIX, VIX9D, VIX3M and join on date."""
    frames: list[pd.DataFrame] = []
    for col_name, ticker in VOL_TICKERS.items():
        try:
            df = _retry_yf(ticker, start)
        except RuntimeError:
            logger.warning("could not fetch %s; column will be NaN", ticker)
            continue
        s = (
            df["Close"].rename(col_name.lower())
            if "Close" in df.columns
            else df["close"].rename(col_name.lower())
        )
        frames.append(s.to_frame())

    if not frames:
        raise RuntimeError("no volatility indicators fetched")

    out = pd.concat(frames, axis=1)
    out.index = pd.to_datetime(out.index).normalize()
    out.index.name = "date"
    return out


def fetch_macro(start: str = START_DATE) -> pd.DataFrame:
    """Pull FRED series via the public CSV endpoint (no API key required).

    Each series is downloaded individually (FRED CSV endpoint serves one series
    per request), then joined on date.
    """
    frames: list[pd.DataFrame] = []
    for series_id, col_name in FRED_SERIES.items():
        url = FRED_CSV_URL.format(series_id=series_id)
        try:
            resp = httpx.get(url, timeout=30.0, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning("FRED fetch failed for %s: %s", series_id, e)
            continue

        # FRED CSV: columns are "observation_date,<SERIES_ID>" (or "DATE,<SERIES_ID>" for legacy)
        df = pd.read_csv(io.StringIO(resp.text))
        date_col = df.columns[0]
        value_col = df.columns[1]
        df = df.rename(columns={date_col: "date", value_col: col_name})
        df["date"] = pd.to_datetime(df["date"])
        # FRED uses "." for missing values
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
        df = df.set_index("date").sort_index()
        df = df[df.index >= pd.Timestamp(start)]
        frames.append(df)

    if not frames:
        raise RuntimeError("no FRED series fetched")

    out = pd.concat(frames, axis=1)
    out.index = pd.to_datetime(out.index).normalize()
    out.index.name = "date"
    return out


# Macro columns are forward-filled at most one trading day to bridge holidays.
_MACRO_COLS = ["vix", "vix9d", "vix3m", "yield_10y", "yield_2y", "fed_funds"]


def fetch_market_panel(start: str = START_DATE) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch and join SPY, volatility, and macro data into one panel.

    Each source is fetched independently and failures are recorded rather than
    raised, so a transient FRED outage still yields a usable price/VIX panel.
    SPY is mandatory; if it cannot be fetched the function raises.

    Returns
    -------
    panel : DataFrame
        DatetimeIndex named ``date`` with SPY OHLCV joined to vol/macro series.
    status : dict
        Maps each source name to ``"ok"`` or a short error string.
    """
    status: dict[str, str] = {}

    spy = fetch_spy(start)  # mandatory — let failure propagate
    status["spy"] = "ok"

    try:
        vol = fetch_vol_indicators(start)
        status["vol"] = "ok"
    except Exception as e:  # noqa: BLE001 — recorded and surfaced to the caller
        logger.warning("volatility fetch failed: %s", e)
        vol = pd.DataFrame(index=spy.index)
        status["vol"] = f"failed: {e}"

    try:
        macro = fetch_macro(start)
        status["macro"] = "ok"
    except Exception as e:  # noqa: BLE001 — recorded and surfaced to the caller
        logger.warning("macro fetch failed: %s", e)
        macro = pd.DataFrame(index=spy.index)
        status["macro"] = f"failed: {e}"

    panel = spy.join(vol, how="left").join(macro, how="left")
    present = [c for c in _MACRO_COLS if c in panel.columns]
    panel[present] = panel[present].ffill(limit=1)
    panel.index.name = "date"
    return panel, status
