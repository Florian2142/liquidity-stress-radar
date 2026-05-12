"""Liquidity Stress Radar — Streamlit dashboard.

Run locally::

    streamlit run src/liquidity_radar/dashboard/app.py

Three-panel layout (Dashboard tab) + Methods tab.
All heavy data loads are cached with @st.cache_data(ttl=3600).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `streamlit run src/liquidity_radar/dashboard/app.py` from project root.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from liquidity_radar.config import DATA_DIR, FIGURES_DIR  # noqa: E402
from liquidity_radar.data.store import get_connection, get_features_panel  # noqa: E402
from liquidity_radar.features.liquidity import amihud_illiquidity, corwin_schultz_spread, edge_spread  # noqa: E402
from liquidity_radar.features.macro import yield_curve_slope  # noqa: E402
from liquidity_radar.features.technical import realized_vol_20d, spy_drawdown_from_high  # noqa: E402
from liquidity_radar.features.volatility import vix_5d_change, vix_term_ratio  # noqa: E402
from liquidity_radar.models.logistic import FEATURE_COLS  # noqa: E402

st.set_page_config(
    page_title="Liquidity Stress Radar",
    page_icon="📡",
    layout="wide",
)

FEATURE_LABELS = {
    "amihud": "Amihud Illiquidity",
    "cs_spread": "Corwin-Schultz Spread",
    "edge": "EDGE Spread",
    "vix_5d_change": "VIX 5-Day Change",
    "vix_term_ratio": "VIX Term Ratio (9D/3M)",
    "yield_curve_slope": "Yield Curve Slope (10Y-2Y)",
    "spy_drawdown": "SPY Drawdown from 1Y High",
    "realized_vol_20d": "Realised Vol (20D Ann.)",
}

STATUS_LEVELS = [
    (0.25, "Calm", "#2ecc71"),
    (0.50, "Watch", "#f39c12"),
    (0.75, "Elevated", "#e67e22"),
    (1.01, "Stress", "#e74c3c"),
]


def _status(prob: float) -> tuple[str, str]:
    for threshold, label, color in STATUS_LEVELS:
        if prob < threshold:
            return label, color
    return "Stress", "#e74c3c"


# ── Cached loaders ────────────────────────────────────────────────────────


@st.cache_data(ttl=3600)
def load_panel() -> pd.DataFrame:
    """Load joined panel — from DuckDB locally, or live from yfinance/FRED on the cloud."""
    db_path = DATA_DIR / "lsr.duckdb"
    if db_path.exists():
        with get_connection() as con:
            return get_features_panel(con)
    # Fallback: fetch directly (used when running on Streamlit Community Cloud)
    from liquidity_radar.data.ingest import fetch_macro, fetch_spy, fetch_vol_indicators

    spy = fetch_spy()
    vol = fetch_vol_indicators()
    macro = fetch_macro()
    panel = spy.join(vol, how="left").join(macro, how="left")
    macro_cols = ["vix", "vix9d", "vix3m", "yield_10y", "yield_2y", "fed_funds"]
    panel[macro_cols] = panel[macro_cols].ffill(limit=1)
    panel.index.name = "date"
    return panel


@st.cache_data(ttl=3600)
def load_features(panel: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=panel.index)
    feat["amihud"] = amihud_illiquidity(panel)
    feat["cs_spread"] = corwin_schultz_spread(panel)
    feat["edge"] = edge_spread(panel)
    feat["vix_5d_change"] = vix_5d_change(panel)
    feat["vix_term_ratio"] = vix_term_ratio(panel)
    feat["yield_curve_slope"] = yield_curve_slope(panel)
    feat["spy_drawdown"] = spy_drawdown_from_high(panel)
    feat["realized_vol_20d"] = realized_vol_20d(panel)
    return feat[FEATURE_COLS]


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
    return pd.read_parquet(path)


def predict_one(x: np.ndarray, params: dict) -> float:
    """Apply saved scaler + logistic coefs to a single feature vector."""
    x_scaled = (x - params["scaler_mean"]) / params["scaler_scale"]
    log_odds = float(x_scaled @ params["coef"] + params["intercept"][0])
    return float(1.0 / (1.0 + np.exp(-log_odds)))


def feature_contributions(x: np.ndarray, params: dict) -> np.ndarray:
    """Standardised coefficient × standardised feature value."""
    x_scaled = (x - params["scaler_mean"]) / params["scaler_scale"]
    return params["coef"] * x_scaled


# ── Main layout ───────────────────────────────────────────────────────────

st.title("📡 Liquidity Stress Radar")
st.caption(
    "Binary classifier predicting S&P 500 drawdowns ≥ 5% in the next 20 trading days. "
    "TUM CEFS term project · yfinance + FRED · Walk-forward CV."
)

tab_dash, tab_methods = st.tabs(["Dashboard", "Methods"])

with tab_dash:
    # Load data
    with st.spinner("Loading data…"):
        panel = load_panel()
        features = load_features(panel)
        params = load_model_params()
        predictions = load_predictions()

    if not params:
        st.error(
            "Model not trained yet. "
            "Run `python scripts/02_train_logistic.py` from the project root first."
        )
        st.stop()

    # Today's features
    today_feat_row = features.dropna().iloc[-1]
    today_date = today_feat_row.name
    x_today = today_feat_row.to_numpy(dtype=float)
    today_prob = predict_one(x_today, params)
    status_label, status_color = _status(today_prob)

    oos_auc = float(params["oos_auc"][0]) if "oos_auc" in params else float("nan")

    # ── Top row: gauge + sparkline + status ──────────────────────────────
    col_gauge, col_spark, col_status = st.columns([1, 2, 1])

    with col_gauge:
        st.subheader("Today's Stress Probability")
        st.caption(f"As of {today_date.date()}")
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=round(today_prob * 100, 1),
                number={"suffix": "%", "font": {"size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%"},
                    "bar": {"color": status_color},
                    "steps": [
                        {"range": [0, 25], "color": "#d5f5e3"},
                        {"range": [25, 50], "color": "#fef9e7"},
                        {"range": [50, 75], "color": "#fde8d8"},
                        {"range": [75, 100], "color": "#fadbd8"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            )
        )
        gauge.update_layout(height=220, margin=dict(t=20, b=10, l=20, r=20))
        st.plotly_chart(gauge, use_container_width=True)

    with col_spark:
        st.subheader("90-Day Probability Trend")
        recent = features.dropna().iloc[-90:]
        X_90 = recent.to_numpy(dtype=float)
        X_90_scaled = (X_90 - params["scaler_mean"]) / params["scaler_scale"]
        log_odds_90 = X_90_scaled @ params["coef"] + params["intercept"][0]
        probs_90 = 1.0 / (1.0 + np.exp(-log_odds_90))
        spark = go.Figure()
        spark.add_trace(
            go.Scatter(
                x=recent.index,
                y=probs_90 * 100,
                mode="lines",
                line=dict(color="#2166ac", width=2),
                fill="tozeroy",
                fillcolor="rgba(33,102,172,0.15)",
            )
        )
        spark.add_hline(y=50, line_dash="dash", line_color="grey", line_width=1)
        spark.update_layout(
            height=220,
            margin=dict(t=20, b=10, l=10, r=10),
            yaxis=dict(range=[0, 100], ticksuffix="%"),
            xaxis_title=None,
            yaxis_title="Probability",
            showlegend=False,
        )
        st.plotly_chart(spark, use_container_width=True)

    with col_status:
        st.subheader("Status")
        st.markdown(
            f"""
            <div style="
                background:{status_color};
                color:white;
                border-radius:12px;
                padding:18px 10px;
                text-align:center;
                font-size:2rem;
                font-weight:700;
                margin-top:30px;
            ">{status_label}</div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Calm < 25% · Watch 25–50% · Elevated 50–75% · Stress ≥ 75%"
        )

    st.divider()

    # ── Bottom row: feature contributions + KPI tiles ────────────────────
    col_contrib, col_kpi = st.columns([3, 2])

    with col_contrib:
        st.subheader("Today's Feature Contributions to Log-Odds")
        contribs = feature_contributions(x_today, params)
        contrib_df = pd.DataFrame(
            {"feature": [FEATURE_LABELS[c] for c in FEATURE_COLS], "contribution": contribs}
        ).sort_values("contribution")
        colors = ["#d6604d" if v > 0 else "#2166ac" for v in contrib_df["contribution"]]
        bar_fig = go.Figure(
            go.Bar(
                x=contrib_df["contribution"],
                y=contrib_df["feature"],
                orientation="h",
                marker_color=colors,
            )
        )
        bar_fig.add_vline(x=0, line_color="black", line_width=1)
        bar_fig.update_layout(
            height=320,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title="Contribution to log-odds (positive = raises alert)",
            yaxis_title=None,
            showlegend=False,
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    with col_kpi:
        st.subheader("Model Performance (Out-of-Sample)")
        st.metric("ROC-AUC", f"{oos_auc:.3f}", help="Out-of-sample walk-forward CV")
        st.metric(
            "VIX-only baseline",
            "~0.49",
            delta=f"+{oos_auc - 0.49:.3f} vs VIX-only",
            delta_color="normal",
        )

        if not predictions.empty:
            from liquidity_radar.eval.metrics import compute_lead_time

            lead = compute_lead_time(predictions, lookback=30, threshold=0.5)
            st.metric(
                "Mean lead time",
                f"{lead['mean']:.0f} days" if not pd.isna(lead["mean"]) else "N/A",
                help="Days before stress onset that model first signals prob > 0.5",
            )

        today_vals = pd.Series(x_today, index=FEATURE_COLS)
        st.caption("**Today's raw feature values:**")
        for col, val in today_vals.items():
            st.caption(f"  {FEATURE_LABELS[col]}: `{val:.4g}`")


with tab_methods:
    st.header("Methods")

    st.subheader("Research question")
    st.markdown(
        "> Do liquidity-based features improve prediction of S&P 500 drawdowns ≥ 5% "
        "in the next 20 days, beyond a VIX-only baseline?"
    )

    st.subheader("Target variable")
    st.markdown(
        "Binary label = 1 if SPY adj. close falls ≥ 5% at any point in the next "
        "20 trading days. Labels are constructed look-forward only; no leakage."
    )

    st.subheader("Features (8 total)")
    st.markdown(
        """
| Group | Feature | Description |
|---|---|---|
| Liquidity | Amihud illiquidity | abs(return) / dollar-volume, 20-day rolling mean (Amihud 2002) |
| Liquidity | Corwin-Schultz spread | High-low spread proxy, 2-day rolling window (Corwin & Schultz 2012) |
| Liquidity | EDGE spread | Efficient bid-ask estimator from OHLC prices (Ardia et al. 2024) |
| Volatility | VIX 5-day change | Short-term fear momentum |
| Volatility | VIX term ratio | VIX9D / VIX3M — inverted term structure signals near-term stress |
| Macro | Yield curve slope | 10Y − 2Y Treasury yield; negative = inverted curve |
| Technical | SPY drawdown | Distance from 252-day rolling high |
| Technical | Realised vol (20D) | Annualised standard deviation of daily returns |
        """
    )

    st.subheader("Model")
    st.markdown(
        "Logistic regression with L2 regularisation (C = 1.0). Features are "
        "standardised within each fold using `StandardScaler` — scaler is fit "
        "on training data only, preventing any leakage."
    )

    st.subheader("Validation")
    st.markdown(
        "Expanding-window walk-forward cross-validation. Parameters: "
        "minimum 5 years of training history, 6-month purge gap between "
        "train and test, 6-month test windows stepping every 6 months. "
        "The purge gap prevents label overlap (20-day forward return window)."
    )

    st.subheader("Data sources")
    st.markdown(
        "SPY OHLCV, ^VIX, ^VIX9D, ^VIX3M via **yfinance** (free). "
        "DGS10, DGS2, DFF via **FRED** public CSV endpoint (no API key required). "
        "All data stored in DuckDB at `data/lsr.duckdb`."
    )

    st.subheader("Reproducibility")
    st.code(
        "pip install -r requirements.txt\n"
        "python scripts/01_initial_load.py   # fetch data\n"
        "python scripts/02_train_logistic.py # train model\n"
        "python scripts/03_evaluate.py       # produce plots\n"
        "streamlit run src/liquidity_radar/dashboard/app.py",
        language="bash",
    )
