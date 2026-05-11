"""Volatility features derived from VIX term structure."""

from __future__ import annotations

import pandas as pd


def vix_5d_change(panel: pd.DataFrame) -> pd.Series:
    """5-trading-day change in the VIX level.

    Parameters
    ----------
    panel : DataFrame
        Output of ``get_features_panel``; must contain a ``vix`` column.

    Returns
    -------
    Series named ``vix_5d_change``. Positive values mean rising fear.
    """
    if "vix" not in panel.columns:
        raise ValueError("panel must have a 'vix' column")
    return panel["vix"].diff(5).rename("vix_5d_change")


def vix_term_ratio(panel: pd.DataFrame) -> pd.Series:
    """VIX9D / VIX3M — short-term vs. medium-term implied vol ratio.

    A ratio > 1 signals near-term fear above the medium-term baseline
    (inverted term structure), which historically precedes stress events.

    Parameters
    ----------
    panel : DataFrame
        Must contain ``vix9d`` and ``vix3m`` columns.

    Returns
    -------
    Series named ``vix_term_ratio``.
    """
    required = {"vix9d", "vix3m"}
    if not required.issubset(panel.columns):
        raise ValueError(f"panel must have columns: {required}")
    return (panel["vix9d"] / panel["vix3m"]).rename("vix_term_ratio")
