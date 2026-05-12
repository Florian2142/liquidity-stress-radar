"""Liquidity proxies computed from daily OHLCV.

Five estimators:

- :func:`amihud_illiquidity` (Amihud, 2002) — raw 20-day rolling mean
- :func:`amihud_zscore` — Amihud normalised against its own trailing 252-day history
- :func:`amihud_5d_change` — 5-day momentum in Amihud (analogous to vix_5d_change)
- :func:`corwin_schultz_spread` (Corwin & Schultz, 2012)
- :func:`edge_spread` via the ``bidask`` package (Ardia et al., 2024)

All functions take a DataFrame with at minimum the columns they need and return
a Series indexed by date.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from liquidity_radar.config import AMIHUD_WINDOW

AMIHUD_ZSCORE_WINDOW = 252  # trading days of history used for rolling normalisation

logger = logging.getLogger(__name__)


def amihud_illiquidity(
    prices: pd.DataFrame,
    window: int = AMIHUD_WINDOW,
) -> pd.Series:
    """Amihud (2002) illiquidity ratio: |return| / dollar-volume, rolling mean.

    Parameters
    ----------
    prices : DataFrame
        Indexed by date. Must contain ``adj_close`` and ``volume`` columns.
    window : int
        Rolling-mean window length (default: ``AMIHUD_WINDOW`` = 20).

    Returns
    -------
    Series
        Rolling-mean Amihud ratio. The first ``window`` values are NaN by design.

    Notes
    -----
    The original Amihud formula is :math:`|r_t| / V_t` where :math:`V_t` is dollar
    volume. We compute that per-day then take a rolling mean to denoise.
    """
    if "adj_close" not in prices.columns or "volume" not in prices.columns:
        raise ValueError("prices must have columns: adj_close, volume")

    px = prices["adj_close"]
    vol = prices["volume"]
    daily_return = px.pct_change()
    dollar_volume = px * vol

    # Avoid division by zero / inf when volume is zero (rare halts)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_day = daily_return.abs() / dollar_volume
    per_day = per_day.replace([np.inf, -np.inf], np.nan)

    return per_day.rolling(window, min_periods=window).mean().rename("amihud")


def amihud_zscore(
    prices: pd.DataFrame,
    window: int = AMIHUD_WINDOW,
    z_window: int = AMIHUD_ZSCORE_WINDOW,
) -> pd.Series:
    """Amihud illiquidity z-scored against its own trailing 252-day history.

    Removes the secular downward trend in market liquidity (decimalization,
    algorithmic market-making) by asking: *how unusual is today's illiquidity
    relative to the recent regime?* A reading of +2 means twice as illiquid
    as normal for this market environment.

    Parameters
    ----------
    prices : DataFrame
        Indexed by date. Must contain ``adj_close`` and ``volume``.
    window : int
        Inner Amihud rolling window (default: 20 days).
    z_window : int
        Outer rolling window for the z-score normalisation (default: 252 days).

    Returns
    -------
    Series
        Z-scored Amihud. First valid value appears at index ``window + z_window``.
    """
    illiq = amihud_illiquidity(prices, window=window)
    rolling_mean = illiq.rolling(z_window, min_periods=z_window).mean()
    rolling_std = illiq.rolling(z_window, min_periods=z_window).std()
    return ((illiq - rolling_mean) / rolling_std).rename("amihud_zscore")


def amihud_5d_change(
    prices: pd.DataFrame,
    window: int = AMIHUD_WINDOW,
) -> pd.Series:
    """5-day change in the Amihud 20-day rolling mean.

    Captures *deteriorating* liquidity: a sudden spike in illiquidity is more
    predictive of stress than the absolute level (analogous to ``vix_5d_change``).

    Parameters
    ----------
    prices : DataFrame
        Indexed by date. Must contain ``adj_close`` and ``volume``.
    window : int
        Inner Amihud rolling window (default: 20 days).

    Returns
    -------
    Series
        First-difference of the rolling Amihud over 5 trading days.
    """
    illiq = amihud_illiquidity(prices, window=window)
    return illiq.diff(5).rename("amihud_5d_change")


def corwin_schultz_spread(prices: pd.DataFrame) -> pd.Series:
    """Corwin & Schultz (2012) high-low spread proxy.

    Parameters
    ----------
    prices : DataFrame
        Indexed by date. Must contain ``high`` and ``low`` columns.

    Returns
    -------
    Series
        Daily spread estimate, clipped to [0, inf). The first row is NaN
        because the formula requires two consecutive days.

    Notes
    -----
    β_t = [ln(H_t/L_t)]² + [ln(H_{t-1}/L_{t-1})]²
    γ_t = [ln(max(H_t,H_{t-1}) / min(L_t,L_{t-1}))]²
    α_t = (√(2β) - √β) / (3 - 2√2) - √(γ / (3 - 2√2))
    S_t = 2(eᵅ - 1) / (1 + eᵅ),  clipped to 0 when α < 0 (noise artifact).
    """
    if "high" not in prices.columns or "low" not in prices.columns:
        raise ValueError("prices must have columns: high, low")

    log_hl = np.log(prices["high"] / prices["low"])
    beta = log_hl**2 + log_hl.shift(1) ** 2

    h2 = np.maximum(prices["high"], prices["high"].shift(1))
    l2 = np.minimum(prices["low"], prices["low"].shift(1))
    gamma = np.log(h2 / l2) ** 2

    k = 3.0 - 2.0 * np.sqrt(2.0)  # ≈ 0.1716
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    alpha = alpha.clip(lower=0)  # negative α produces nonsensical spreads

    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return spread.rename("cs_spread")


def edge_spread(prices: pd.DataFrame, window: int = AMIHUD_WINDOW) -> pd.Series:
    """EDGE estimator (Ardia, Guidotti & Kroencke, JFE 2024).

    Parameters
    ----------
    prices : DataFrame
        Indexed by date. Must contain ``open``, ``high``, ``low``, ``close``.
    window : int
        Rolling window length (default: ``AMIHUD_WINDOW`` = 20).

    Returns
    -------
    Series
        Rolling EDGE spread estimate. Values near 0 indicate tight spreads.
        Clipped to [0, inf).
    """
    required = {"open", "high", "low", "close"}
    if not required.issubset(prices.columns):
        raise ValueError(f"prices must have columns: {required}")

    from bidask import edge_rolling  # local import keeps top-level imports clean

    result = edge_rolling(
        prices[["open", "high", "low", "close"]],
        window=window,
        min_periods=window,
    )
    return result.clip(lower=0).rename("edge")
