"""Technical features derived from SPY price history."""

from __future__ import annotations

import pandas as pd

from liquidity_radar.config import REALIZED_VOL_WINDOW


def spy_drawdown_from_high(panel: pd.DataFrame, window: int = 252) -> pd.Series:
    """SPY drawdown from its rolling 1-year (252-day) high.

    Returns a non-positive fraction: -0.15 means the index is 15% below
    its 252-day peak. Zero means a new all-time high was set today.

    Parameters
    ----------
    panel : DataFrame
        Must contain ``adj_close``.
    window : int
        Rolling look-back for the peak (default 252 trading days ≈ 1 year).

    Returns
    -------
    Series named ``spy_drawdown``.
    """
    if "adj_close" not in panel.columns:
        raise ValueError("panel must have an 'adj_close' column")
    rolling_peak = panel["adj_close"].rolling(window, min_periods=window).max()
    return (panel["adj_close"] / rolling_peak - 1.0).rename("spy_drawdown")


def realized_vol_20d(panel: pd.DataFrame, window: int = REALIZED_VOL_WINDOW) -> pd.Series:
    """20-day annualised realised volatility of SPY log-returns.

    Parameters
    ----------
    panel : DataFrame
        Must contain ``adj_close``.
    window : int
        Rolling window length (default ``REALIZED_VOL_WINDOW`` = 20).

    Returns
    -------
    Series named ``realized_vol_20d``. Units: annualised fraction
    (0.20 ≈ 20% vol).
    """
    if "adj_close" not in panel.columns:
        raise ValueError("panel must have an 'adj_close' column")
    daily_ret = panel["adj_close"].pct_change()
    rv = daily_ret.rolling(window, min_periods=window).std() * (252.0**0.5)
    return rv.rename("realized_vol_20d")
