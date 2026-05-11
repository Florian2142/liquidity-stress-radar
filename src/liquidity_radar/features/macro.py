"""Macro features derived from Treasury yields and the Fed Funds Rate."""

from __future__ import annotations

import pandas as pd


def yield_curve_slope(panel: pd.DataFrame) -> pd.Series:
    """10-year minus 2-year Treasury yield spread (basis points as decimals).

    An inverted yield curve (negative slope) has historically preceded
    recessions and equity drawdowns.

    Parameters
    ----------
    panel : DataFrame
        Output of ``get_features_panel``; must contain ``yield_10y``
        and ``yield_2y`` columns.

    Returns
    -------
    Series named ``yield_curve_slope``.
    """
    required = {"yield_10y", "yield_2y"}
    if not required.issubset(panel.columns):
        raise ValueError(f"panel must have columns: {required}")
    return (panel["yield_10y"] - panel["yield_2y"]).rename("yield_curve_slope")


def ffr_change(panel: pd.DataFrame, window: int = 5) -> pd.Series:
    """N-day change in the Effective Federal Funds Rate.

    Rapid FFR increases can tighten financial conditions and pressure equities.

    Parameters
    ----------
    panel : DataFrame
        Must contain a ``fed_funds`` column.
    window : int
        Look-back in trading days (default 5 ≈ one week).

    Returns
    -------
    Series named ``ffr_change``.
    """
    if "fed_funds" not in panel.columns:
        raise ValueError("panel must have a 'fed_funds' column")
    return panel["fed_funds"].diff(window).rename("ffr_change")
