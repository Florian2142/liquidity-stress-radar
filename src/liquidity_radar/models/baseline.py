"""Naïve VIX-threshold baseline for comparison with the logistic model."""

from __future__ import annotations

import pandas as pd


def vix_threshold_predict(
    panel: pd.DataFrame,
    threshold: float = 25.0,
) -> pd.Series:
    """Return 1 on days when VIX > threshold, else 0.

    This is the simplest possible stress detector: if fear is already
    elevated (VIX above 25), flag as a stress period. Used as the
    low-bar baseline that the full model must beat.

    Parameters
    ----------
    panel : DataFrame
        Must contain a ``vix`` column.
    threshold : float
        VIX level above which we predict stress (default 25.0).

    Returns
    -------
    Series of int (0/1) named ``baseline_pred``, indexed by date.
    """
    if "vix" not in panel.columns:
        raise ValueError("panel must have a 'vix' column")
    return (panel["vix"] > threshold).astype(int).rename("baseline_pred")
