"""Single source of truth for the model feature matrix.

Every consumer — the training script, the robustness pipeline, and the
Streamlit dashboard — builds features through :func:`build_feature_matrix`
so the recipe can never drift between code paths.
"""

from __future__ import annotations

import pandas as pd

from liquidity_radar.features.liquidity import (
    amihud_5d_change,
    amihud_illiquidity,
    amihud_zscore,
    corwin_schultz_spread,
    edge_spread,
)
from liquidity_radar.features.macro import yield_curve_slope
from liquidity_radar.features.technical import realized_vol_20d, spy_drawdown_from_high
from liquidity_radar.features.volatility import vix_5d_change, vix_term_ratio
from liquidity_radar.models.logistic import FEATURE_COLS


def build_feature_matrix(panel: pd.DataFrame, include_raw_amihud: bool = False) -> pd.DataFrame:
    """Compute the nine model features from a joined market panel.

    Parameters
    ----------
    panel : DataFrame
        Output of :func:`liquidity_radar.data.store.get_features_panel`.
        Must contain SPY OHLCV plus ``vix``, ``vix9d``, ``vix3m``,
        ``yield_10y``, ``yield_2y``.
    include_raw_amihud : bool
        If True, also return the raw Amihud level as an ``amihud`` column.
        Kept for the DuckDB ``features`` table and reference plots; the model
        itself uses only the columns in :data:`FEATURE_COLS`.

    Returns
    -------
    DataFrame
        DatetimeIndex named ``date``. Columns are exactly ``FEATURE_COLS``
        (plus ``amihud`` when requested), in canonical order.
    """
    feat = pd.DataFrame(index=panel.index)
    if include_raw_amihud:
        feat["amihud"] = amihud_illiquidity(panel)
    feat["amihud_zscore"] = amihud_zscore(panel)
    feat["amihud_5d_change"] = amihud_5d_change(panel)
    feat["cs_spread"] = corwin_schultz_spread(panel)
    feat["edge"] = edge_spread(panel)
    feat["vix_5d_change"] = vix_5d_change(panel)
    feat["vix_term_ratio"] = vix_term_ratio(panel)
    feat["yield_curve_slope"] = yield_curve_slope(panel)
    feat["spy_drawdown"] = spy_drawdown_from_high(panel)
    feat["realized_vol_20d"] = realized_vol_20d(panel)
    feat.index.name = "date"

    cols = (["amihud"] if include_raw_amihud else []) + FEATURE_COLS
    return feat[cols]
