"""Cached data loaders and small numerical helpers for the dashboard.

Kept separate from ``app.py`` so the layout code stays readable and the loading
logic can be unit-tested without importing Streamlit's page machinery.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from liquidity_radar.config import DATA_DIR
from liquidity_radar.data.store import get_connection, get_features_panel
from liquidity_radar.features.build import build_feature_matrix
from liquidity_radar.models.logistic import FEATURE_COLS

ROBUST_DIR = DATA_DIR / "robustness"

FEATURE_LABELS: dict[str, str] = {
    "amihud_zscore": "Amihud Z-Score (regime-adj.)",
    "amihud_5d_change": "Amihud 5-Day Change",
    "cs_spread": "Corwin-Schultz Spread",
    "edge": "EDGE Spread",
    "vix_5d_change": "VIX 5-Day Change",
    "vix_term_ratio": "VIX Term Ratio (9D/3M)",
    "yield_curve_slope": "Yield Curve Slope (10Y-2Y)",
    "spy_drawdown": "SPY Drawdown from 1Y High",
    "realized_vol_20d": "Realised Vol (20D Ann.)",
}

# Features where a lower (more negative) reading signals stress.
LOW_IS_STRESS: set[str] = {"yield_curve_slope", "spy_drawdown"}


# ── Panel / feature / model loaders ───────────────────────────────────────


@st.cache_data(ttl=3600)
def load_panel() -> pd.DataFrame:
    """Joined market panel — DuckDB locally, committed snapshot on the cloud."""
    db_path = DATA_DIR / "lsr.duckdb"
    if db_path.exists():
        with get_connection() as con:
            return get_features_panel(con)
    snapshot = DATA_DIR / "panel_snapshot.parquet"
    if snapshot.exists():
        df = pd.read_parquet(snapshot)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df
    from liquidity_radar.data.ingest import fetch_market_panel

    panel, _ = fetch_market_panel()
    return panel


@st.cache_data(ttl=3600)
def load_features(_panel: pd.DataFrame) -> pd.DataFrame:
    """Model feature matrix built from the panel (underscore skips hashing)."""
    return build_feature_matrix(_panel)[FEATURE_COLS]


@st.cache_data(ttl=3600)
def load_model_params() -> dict:
    path = DATA_DIR / "model_params.npz"
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


@st.cache_data(ttl=3600)
def load_predictions() -> pd.DataFrame:
    path = DATA_DIR / "predictions.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=3600)
def load_fold_coefs() -> pd.DataFrame:
    path = DATA_DIR / "fold_coefs.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


# ── Robustness artefact loaders ───────────────────────────────────────────


@st.cache_data(ttl=3600)
def load_robust_csv(name: str) -> pd.DataFrame:
    path = ROBUST_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=3600)
def load_robust_summary() -> dict:
    path = ROBUST_DIR / "summary.json"
    return json.loads(Path(path).read_text()) if path.exists() else {}


# ── Inference helpers ─────────────────────────────────────────────────────


def predict_one(x: np.ndarray, params: dict) -> float:
    x_scaled = (x - params["scaler_mean"]) / params["scaler_scale"]
    log_odds = float(x_scaled @ params["coef"] + params["intercept"][0])
    return float(1.0 / (1.0 + np.exp(-log_odds)))


def batch_predict(X: np.ndarray, params: dict) -> np.ndarray:
    X_scaled = (X - params["scaler_mean"]) / params["scaler_scale"]
    return 1.0 / (1.0 + np.exp(-(X_scaled @ params["coef"] + params["intercept"][0])))


def feature_contributions(x: np.ndarray, params: dict) -> np.ndarray:
    x_scaled = (x - params["scaler_mean"]) / params["scaler_scale"]
    return params["coef"] * x_scaled


def stress_percentile(col: str, series: pd.Series, today_val: float) -> float:
    """Percentile of today's reading where higher = more stressed."""
    pct = float((series < today_val).mean())
    return 1.0 - pct if col in LOW_IS_STRESS else pct
