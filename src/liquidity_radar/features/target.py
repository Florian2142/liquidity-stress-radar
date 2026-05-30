"""Target variable construction.

The target is binary: did SPY fall by at least :data:`TARGET_DRAWDOWN_PCT` from
today's close at any point in the next :data:`TARGET_HORIZON_DAYS` trading days?

This is the only place in the code where forward-looking computations are
allowed. Everywhere else, use only past data.
"""

from __future__ import annotations

import pandas as pd

from liquidity_radar.config import TARGET_DRAWDOWN_PCT, TARGET_HORIZON_DAYS


def forward_drawdown_label(
    prices: pd.DataFrame,
    horizon: int = TARGET_HORIZON_DAYS,
    threshold: float = TARGET_DRAWDOWN_PCT,
) -> pd.DataFrame:
    """Compute forward drawdown and binary label.

    Parameters
    ----------
    prices : DataFrame
        Indexed by date with ``adj_close`` column.
    horizon : int
        Number of trading days forward to look (default 20).
    threshold : float
        Drawdown magnitude that triggers a positive label (default 0.05 ⇒ 5%).

    Returns
    -------
    DataFrame
        Indexed by date with columns:

        - ``forward_drawdown_pct``: maximum drawdown over the next ``horizon`` days,
          expressed as a *positive* fraction. 0.07 means a 7% drop.
        - ``label``: 1 if ``forward_drawdown_pct >= threshold``, else 0.

        The last ``horizon`` rows are NaN/NA because the future is not yet known;
        the caller must drop them before training.
    """
    if "adj_close" not in prices.columns:
        raise ValueError("prices must have an adj_close column")

    px = prices["adj_close"]

    # Rolling minimum over the FUTURE horizon. Implementation: shift by -horizon
    # and take rolling min on the reversed-time view.
    future_min = px.rolling(window=horizon, min_periods=horizon).min().shift(-horizon + 1)

    # Drawdown = (today_close - min_over_next_horizon) / today_close
    forward_dd = (px - future_min) / px
    forward_dd = forward_dd.clip(lower=0)  # ignore upward moves

    label = (forward_dd >= threshold).astype("Int8")

    out = pd.DataFrame(
        {
            "forward_drawdown_pct": forward_dd,
            "label": label,
        }
    )

    # The last `horizon` rows have undefined labels (we can't see the future).
    out.loc[out.index[-(horizon - 1) :], ["forward_drawdown_pct", "label"]] = pd.NA
    out.index.name = "date"
    return out
