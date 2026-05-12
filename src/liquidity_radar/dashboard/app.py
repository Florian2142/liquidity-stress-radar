"""Liquidity Stress Radar — Streamlit dashboard.

Run locally::

    streamlit run src/liquidity_radar/dashboard/app.py

Five-tab layout: Dashboard · Market Snapshot · History · Stress Events · Methods.
Live market panel auto-refreshes every 2 minutes using @st.fragment(run_every=...).
All heavy data loads are cached with @st.cache_data(ttl=3600).
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
import yfinance as yf  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from liquidity_radar.config import DATA_DIR  # noqa: E402
from liquidity_radar.data.store import get_connection, get_features_panel  # noqa: E402
from liquidity_radar.features.liquidity import (  # noqa: E402
    amihud_5d_change,
    amihud_zscore,
    corwin_schultz_spread,
    edge_spread,
)
from liquidity_radar.features.macro import yield_curve_slope  # noqa: E402
from liquidity_radar.features.technical import realized_vol_20d, spy_drawdown_from_high  # noqa: E402
from liquidity_radar.features.volatility import vix_5d_change, vix_term_ratio  # noqa: E402
from liquidity_radar.models.logistic import FEATURE_COLS  # noqa: E402

st.set_page_config(
    page_title="Liquidity Stress Radar",
    page_icon="📡",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────

FEATURE_LABELS: dict[str, str] = {
    "amihud_zscore": "Amihud Z-Score (regime-adj.)",
    "amihud_5d_change": "Amihud 5D Change (momentum)",
    "cs_spread": "Corwin-Schultz Spread",
    "edge": "EDGE Spread",
    "vix_5d_change": "VIX 5-Day Change",
    "vix_term_ratio": "VIX Term Ratio (9D/3M)",
    "yield_curve_slope": "Yield Curve Slope (10Y-2Y)",
    "spy_drawdown": "SPY Drawdown from 1Y High",
    "realized_vol_20d": "Realised Vol (20D Ann.)",
}

LOW_IS_STRESS: set[str] = {"yield_curve_slope", "spy_drawdown"}

STATUS_LEVELS = [
    (0.25, "Calm", "#2ecc71"),
    (0.50, "Watch", "#f39c12"),
    (0.75, "Elevated", "#e67e22"),
    (1.01, "Stress", "#e74c3c"),
]

MAJOR_EVENTS: dict[str, str] = {
    "2008-09-15": "GFC",
    "2010-05-06": "Flash Crash",
    "2011-08-08": "Euro Crisis",
    "2013-06-24": "Taper Tantrum",
    "2015-08-24": "China Selloff",
    "2018-02-05": "Vol Shock",
    "2020-02-20": "COVID",
    "2022-01-03": "Rate Hikes",
    "2023-03-10": "SVB",
    "2025-04-07": "Tariff Shock",
}

STRESS_THRESHOLD = 0.5
ET = ZoneInfo("America/New_York")


# ── General helpers ───────────────────────────────────────────────────────


def _status(prob: float) -> tuple[str, str]:
    for threshold, label, color in STATUS_LEVELS:
        if prob < threshold:
            return label, color
    return "Stress", "#e74c3c"


def _batch_predict(X: np.ndarray, params: dict) -> np.ndarray:
    X_scaled = (X - params["scaler_mean"]) / params["scaler_scale"]
    return 1.0 / (1.0 + np.exp(-(X_scaled @ params["coef"] + params["intercept"][0])))


def _stress_percentile(col: str, series: pd.Series, today_val: float) -> float:
    pct = float((series < today_val).mean())
    return 1.0 - pct if col in LOW_IS_STRESS else pct


def _percentile_color(pct: float) -> str:
    if pct < 0.50:
        return "#2ecc71"
    if pct < 0.75:
        return "#f39c12"
    if pct < 0.90:
        return "#e67e22"
    return "#e74c3c"


def _compute_stress_events(preds: pd.DataFrame, threshold: float = STRESS_THRESHOLD) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame()
    df = preds.sort_values("date").copy()
    df["above"] = (df["prob"] > threshold).astype(int)
    df["onset"] = (df["above"] == 1) & (df["above"].shift(1, fill_value=0) == 0)
    onsets = df[df["onset"]].copy()
    if onsets.empty:
        return pd.DataFrame()
    rows = []
    for _, row in onsets.iterrows():
        onset_date = row["date"]
        lookback_start = onset_date - pd.Timedelta(days=45)
        window = df[(df["date"] >= lookback_start) & (df["date"] < onset_date)]
        first_signal = window[window["prob"] > threshold]
        lead_days = int((onset_date - first_signal["date"].min()).days) if not first_signal.empty else 0
        rows.append(
            {
                "onset_date": onset_date,
                "peak_prob": df[df["date"] >= onset_date].head(30)["prob"].max(),
                "lead_days": lead_days,
            }
        )
    return pd.DataFrame(rows)


# ── Live market helpers ───────────────────────────────────────────────────


def _market_status() -> tuple[bool, str]:
    """Return (is_open, human label) in US Eastern time."""
    now = datetime.datetime.now(ET)
    open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if now.weekday() >= 5:
        return False, "Weekend"
    if now < open_t:
        mins = int((open_t - now).total_seconds() // 60)
        h, m = divmod(mins, 60)
        return False, f"Pre-Market · opens in {h}h {m}m" if h else f"Pre-Market · opens in {m}m"
    if now > close_t:
        return False, "After Hours"
    mins = int((close_t - now).total_seconds() // 60)
    h, m = divmod(mins, 60)
    return True, f"Market Open · closes in {h}h {m}m" if h else f"Market Open · closes in {m}m"


@st.cache_data(ttl=120)
def fetch_live_snapshot() -> dict | None:
    """Fetch latest SPY + VIX prices. Cached 2 minutes."""
    try:
        hist = yf.Tickers("SPY ^VIX ^VIX9D ^VIX3M").history(
            period="7d", interval="1d", auto_adjust=True
        )
        close = hist["Close"]

        def _g(col: str, i: int = 0) -> float:
            try:
                return float(close[col].dropna().iloc[-(1 + i)])
            except Exception:
                return float("nan")

        return {
            "spy": _g("SPY"),
            "spy_prev": _g("SPY", 1),
            "vix": _g("^VIX"),
            "vix_prev": _g("^VIX", 1),
            "vix9d": _g("^VIX9D"),
            "vix3m": _g("^VIX3M"),
            "fetched_at": datetime.datetime.now(ET),
        }
    except Exception:
        return None


@st.cache_data(ttl=60)
def fetch_intraday_spy() -> pd.DataFrame:
    """Fetch today's 5-min SPY bars. Cached 60 seconds."""
    try:
        df = yf.download("SPY", period="1d", interval="5m", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


def _intraday_prob(
    live: dict,
    feat_clean: pd.DataFrame,
    panel: pd.DataFrame,
    params: dict,
) -> float:
    """Update yesterday's feature vector with live VIX/SPY and rerun the model.

    Three features can be updated intraday:
    - vix_5d_change: live VIX vs. 5 trading days ago
    - vix_term_ratio: live VIX9D / VIX3M
    - spy_drawdown: live SPY price vs. 252-day rolling high
    The two Amihud features (amihud_zscore, amihud_5d_change) need full-day OHLCV
    so they stay at yesterday's values.
    """
    x = feat_clean.iloc[-1].to_numpy(dtype=float).copy()
    vix = live["vix"]
    if not np.isnan(vix):
        vix_s = panel["vix"].dropna()
        if len(vix_s) >= 5:
            x[FEATURE_COLS.index("vix_5d_change")] = vix - float(vix_s.iloc[-5])
    v9, v3 = live["vix9d"], live["vix3m"]
    if not np.isnan(v9) and not np.isnan(v3) and v3 > 0:
        x[FEATURE_COLS.index("vix_term_ratio")] = v9 / v3
    spy_px = live["spy"]
    if not np.isnan(spy_px):
        spy_s = panel["adj_close"].dropna()
        if len(spy_s) >= 252:
            x[FEATURE_COLS.index("spy_drawdown")] = spy_px / float(spy_s.iloc[-252:].max()) - 1.0
    x_scaled = (x - params["scaler_mean"]) / params["scaler_scale"]
    return float(1.0 / (1.0 + np.exp(-float(x_scaled @ params["coef"] + params["intercept"][0]))))


# ── Cached data loaders ───────────────────────────────────────────────────


@st.cache_data(ttl=3600)
def load_panel() -> pd.DataFrame:
    # 1. Local dev: full DuckDB
    db_path = DATA_DIR / "lsr.duckdb"
    if db_path.exists():
        with get_connection() as con:
            return get_features_panel(con)
    # 2. Streamlit Cloud: committed parquet snapshot (avoids yfinance rate-limits)
    snapshot_path = DATA_DIR / "panel_snapshot.parquet"
    if snapshot_path.exists():
        df = pd.read_parquet(snapshot_path)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df
    # 3. Last resort: live fetch (slow, may hit rate limits)
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
def load_features(_panel: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=_panel.index)
    feat["amihud_zscore"]    = amihud_zscore(_panel)
    feat["amihud_5d_change"] = amihud_5d_change(_panel)
    feat["cs_spread"]        = corwin_schultz_spread(_panel)
    feat["edge"]             = edge_spread(_panel)
    feat["vix_5d_change"]    = vix_5d_change(_panel)
    feat["vix_term_ratio"]   = vix_term_ratio(_panel)
    feat["yield_curve_slope"]= yield_curve_slope(_panel)
    feat["spy_drawdown"]     = spy_drawdown_from_high(_panel)
    feat["realized_vol_20d"] = realized_vol_20d(_panel)
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
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def predict_one(x: np.ndarray, params: dict) -> float:
    x_scaled = (x - params["scaler_mean"]) / params["scaler_scale"]
    return float(1.0 / (1.0 + np.exp(-(float(x_scaled @ params["coef"] + params["intercept"][0])))))


def feature_contributions(x: np.ndarray, params: dict) -> np.ndarray:
    x_scaled = (x - params["scaler_mean"]) / params["scaler_scale"]
    return params["coef"] * x_scaled


# ── Page header ───────────────────────────────────────────────────────────

st.title("📡 Liquidity Stress Radar")
st.caption(
    "Binary classifier predicting S&P 500 drawdowns ≥ 5% in the next 20 trading days. "
    "TUM CEFS term project · yfinance + FRED · Walk-forward CV."
)

tab_dash, tab_snap, tab_hist, tab_events, tab_methods = st.tabs(
    ["🎯 Dashboard", "📊 Market Snapshot", "📈 History", "⚡ Stress Events", "📚 Methods"]
)

# ── Shared data (loaded once, tabs share it) ──────────────────────────────

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

feat_clean = features.dropna()
today_feat_row = feat_clean.iloc[-1]
today_date = today_feat_row.name
x_today = today_feat_row.to_numpy(dtype=float)
today_prob = predict_one(x_today, params)
status_label, status_color = _status(today_prob)
oos_auc = float(params["oos_auc"][0]) if "oos_auc" in params else float("nan")


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard
# ══════════════════════════════════════════════════════════════════════════

with tab_dash:

    # ── Live market panel (auto-refreshes every 2 min) ────────────────────
    @st.fragment(run_every=datetime.timedelta(seconds=120))
    def _live_panel() -> None:
        live = fetch_live_snapshot()
        intraday = fetch_intraday_spy()
        is_open, mkt_status = _market_status()

        dot_color = "#2ecc71" if is_open else "#95a5a6"
        pulse_anim = "pulse 1.5s infinite" if is_open else "none"
        fetched_str = live["fetched_at"].strftime("%H:%M:%S ET") if live else "unavailable"

        st.markdown(
            f"""
            <style>
            @keyframes pulse {{
                0%   {{ opacity: 1; }}
                50%  {{ opacity: 0.3; }}
                100% {{ opacity: 1; }}
            }}
            .live-dot {{
                display: inline-block;
                width: 10px; height: 10px;
                border-radius: 50%;
                background: {dot_color};
                animation: {pulse_anim};
                margin-right: 6px;
                vertical-align: middle;
            }}
            .live-bar {{
                display: flex; align-items: center;
                padding: 8px 14px;
                background: {"rgba(46,204,113,0.08)" if is_open else "rgba(149,165,166,0.08)"};
                border-radius: 8px;
                border-left: 3px solid {dot_color};
                margin-bottom: 12px;
            }}
            </style>
            <div class="live-bar">
                <span class="live-dot"></span>
                <span style="font-weight:600; font-size:0.95rem;">{mkt_status}</span>
                <span style="margin-left:auto; color:#888; font-size:0.8rem;">
                    Live data · refreshes every 2 min · last at {fetched_str}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if live is None:
            st.warning("Could not reach yfinance — live data unavailable.")
            return

        spy_chg = (
            (live["spy"] - live["spy_prev"]) / live["spy_prev"] * 100
            if not np.isnan(live["spy_prev"]) and live["spy_prev"] > 0
            else float("nan")
        )
        vix_chg = live["vix"] - live["vix_prev"] if not np.isnan(live["vix_prev"]) else float("nan")
        live_vtr = live["vix9d"] / live["vix3m"] if live["vix3m"] > 0 else float("nan")

        col_chart, col_metrics = st.columns([3, 1])

        with col_metrics:
            st.metric(
                "SPY",
                f"${live['spy']:.2f}" if not np.isnan(live["spy"]) else "—",
                f"{spy_chg:+.2f}%" if not np.isnan(spy_chg) else None,
            )
            st.metric(
                "VIX",
                f"{live['vix']:.2f}" if not np.isnan(live["vix"]) else "—",
                f"{vix_chg:+.2f}" if not np.isnan(vix_chg) else None,
                delta_color="inverse",
            )
            st.metric(
                "VIX Term Ratio",
                f"{live_vtr:.3f}" if not np.isnan(live_vtr) else "—",
                help="VIX9D / VIX3M — above 1 = near-term fear elevated vs 3-month",
            )
            if is_open:
                ip = _intraday_prob(live, feat_clean, panel, params)
                ip_label, ip_color = _status(ip)
                st.metric(
                    "Intraday Estimate",
                    f"{ip*100:.1f}%",
                    help="Live VIX + SPY drawdown plugged into yesterday's model",
                )
                st.markdown(
                    f"<div style='text-align:center;background:{ip_color}22;"
                    f"border:1px solid {ip_color};border-radius:6px;"
                    f"padding:4px 8px;font-weight:600;color:{ip_color};"
                    f"font-size:0.85rem;'>{ip_label}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Intraday estimate available during market hours (9:30–16:00 ET).")

        with col_chart:
            if not intraday.empty and "Close" in intraday.columns:
                close_s = intraday["Close"].dropna()
                if len(close_s) > 1:
                    open_px = float(close_s.iloc[0])
                    line_color = "#2ecc71" if close_s.iloc[-1] >= open_px else "#e74c3c"
                    fill_color = (
                        "rgba(46,204,113,0.10)" if close_s.iloc[-1] >= open_px else "rgba(231,76,60,0.10)"
                    )
                    fig_intra = go.Figure()
                    fig_intra.add_trace(
                        go.Scatter(
                            x=intraday.index,
                            y=close_s,
                            mode="lines",
                            line=dict(color=line_color, width=2),
                            fill="tozeroy",
                            fillcolor=fill_color,
                            name="SPY intraday",
                        )
                    )
                    fig_intra.add_hline(
                        y=open_px,
                        line_dash="dot",
                        line_color="grey",
                        line_width=1,
                        annotation_text="Open",
                        annotation_position="right",
                    )
                    fig_intra.update_layout(
                        height=220,
                        margin=dict(t=10, b=10, l=10, r=60),
                        xaxis_title=None,
                        yaxis_title="SPY ($)",
                        yaxis_tickprefix="$",
                        showlegend=False,
                        title=dict(
                            text="SPY — Today (5-min bars)",
                            font=dict(size=13),
                            x=0,
                        ),
                    )
                    st.plotly_chart(fig_intra, use_container_width=True)
                else:
                    st.caption("Intraday chart: waiting for market open data.")
            else:
                st.caption("Intraday chart unavailable outside market hours.")

        st.divider()

    _live_panel()

    # ── Today's gauge + sparkline + status ───────────────────────────────
    col_gauge, col_spark, col_status = st.columns([1, 2, 1])

    with col_gauge:
        st.subheader("Yesterday's Close Probability")
        st.caption(f"Model run on {today_date.date()} close data")
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
        recent = feat_clean.iloc[-90:]
        probs_90 = _batch_predict(recent.to_numpy(dtype=float), params)
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
        st.caption("Calm < 25% · Watch 25-50% · Elevated 50-75% · Stress >= 75%")

    st.divider()

    # ── Feature contributions + KPI ───────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Market Snapshot
# ══════════════════════════════════════════════════════════════════════════

with tab_snap:
    st.subheader(f"Market Snapshot — {today_date.date()}")
    st.caption(
        "Each card shows today's value and its **stress percentile** vs. full history. "
        "Red = top decile alarm."
    )

    pct_data: list[dict] = []
    for col in FEATURE_COLS:
        series = feat_clean[col].dropna()
        today_val = float(today_feat_row[col])
        pct = _stress_percentile(col, series, today_val)
        pct_data.append(
            {
                "col": col,
                "label": FEATURE_LABELS[col],
                "value": today_val,
                "percentile": pct,
                "color": _percentile_color(pct),
            }
        )

    cols = st.columns(4)
    for i, d in enumerate(pct_data):
        with cols[i % 4]:
            pct_pct = d["percentile"] * 100
            arrow = "▲" if d["percentile"] > 0.5 else "▼"
            st.markdown(
                f"""
                <div style="
                    background:{d['color']}22;
                    border-left: 4px solid {d['color']};
                    border-radius:8px;
                    padding:14px 12px;
                    margin-bottom:12px;
                ">
                    <div style="font-size:0.78rem;color:#555;font-weight:600;">{d['label']}</div>
                    <div style="font-size:1.5rem;font-weight:700;color:#111;">{d['value']:.4g}</div>
                    <div style="font-size:0.85rem;color:{d['color']};font-weight:600;">
                        {arrow} {pct_pct:.0f}th stress pct
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("Stress Percentile Overview")
    pct_df = pd.DataFrame(pct_data).sort_values("percentile", ascending=True)
    bar_colors = [d["color"] for d in pct_df.to_dict("records")]
    pct_fig = go.Figure(
        go.Bar(
            x=pct_df["percentile"] * 100,
            y=pct_df["label"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v*100:.0f}%" for v in pct_df["percentile"]],
            textposition="outside",
        )
    )
    pct_fig.add_vline(x=50, line_dash="dot", line_color="#999", line_width=1)
    pct_fig.add_vline(x=90, line_dash="dash", line_color="#e74c3c", line_width=1)
    pct_fig.update_layout(
        height=340,
        margin=dict(t=10, b=10, l=10, r=60),
        xaxis=dict(range=[0, 110], ticksuffix="%", title="Stress percentile vs. full history"),
        yaxis_title=None,
        showlegend=False,
    )
    st.plotly_chart(pct_fig, use_container_width=True)

    st.divider()
    st.subheader("Feature Summary Table")
    hist_stats = feat_clean.describe(percentiles=[0.25, 0.5, 0.75, 0.90]).T
    hist_stats = hist_stats[["min", "25%", "50%", "75%", "90%", "max"]]
    hist_stats.index = [FEATURE_LABELS[c] for c in hist_stats.index]
    today_series = pd.Series(
        {FEATURE_LABELS[c]: float(today_feat_row[c]) for c in FEATURE_COLS}, name="Today"
    )
    st.dataframe(hist_stats.join(today_series).round(4), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — History
# ══════════════════════════════════════════════════════════════════════════

with tab_hist:
    st.subheader("Historical Feature & Probability Explorer")

    c1, c2 = st.columns([2, 2])
    with c1:
        selected_feature = st.selectbox(
            "Feature to plot",
            options=FEATURE_COLS,
            format_func=lambda c: FEATURE_LABELS[c],
        )
    with c2:
        min_year = int(feat_clean.index.year.min())
        max_year = int(feat_clean.index.year.max())
        year_range = st.slider(
            "Year range",
            min_value=min_year,
            max_value=max_year,
            value=(max(min_year, max_year - 10), max_year),
        )

    date_mask = (feat_clean.index.year >= year_range[0]) & (feat_clean.index.year <= year_range[1])
    hist_feat = feat_clean.loc[date_mask]
    hist_probs = _batch_predict(hist_feat.to_numpy(dtype=float), params)

    fig_hist = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.06,
        subplot_titles=[FEATURE_LABELS[selected_feature], "Stress Probability"],
    )
    fig_hist.add_trace(
        go.Scatter(
            x=hist_feat.index,
            y=hist_feat[selected_feature],
            mode="lines",
            line=dict(color="#2c7bb6", width=1.5),
            name=FEATURE_LABELS[selected_feature],
        ),
        row=1,
        col=1,
    )
    fig_hist.add_trace(
        go.Scatter(
            x=hist_feat.index,
            y=hist_probs * 100,
            mode="lines",
            line=dict(color="#d7191c", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(215,25,28,0.12)",
            name="Stress prob %",
        ),
        row=2,
        col=1,
    )
    fig_hist.add_hline(y=50, line_dash="dash", line_color="grey", line_width=1, row=2, col=1)

    for date_str, label in MAJOR_EVENTS.items():
        ev_date = pd.Timestamp(date_str)
        if year_range[0] <= ev_date.year <= year_range[1]:
            fig_hist.add_vline(
                x=ev_date,
                line_dash="dot",
                line_color="rgba(100,100,100,0.5)",
                line_width=1,
                row="all",
                col=1,
            )
            fig_hist.add_annotation(
                x=ev_date,
                y=1,
                yref="paper",
                text=label,
                showarrow=False,
                textangle=-55,
                font=dict(size=9, color="#555"),
                xanchor="left",
            )

    fig_hist.update_layout(
        height=520,
        margin=dict(t=40, b=10, l=10, r=10),
        showlegend=False,
    )
    fig_hist.update_yaxes(ticksuffix="%", row=2, col=1)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()
    st.subheader("Full OOS Probability History")
    if not predictions.empty:
        pred_mask = (predictions["date"].dt.year >= year_range[0]) & (
            predictions["date"].dt.year <= year_range[1]
        )
        pred_view = predictions.loc[pred_mask].sort_values("date")
        fig_oos = go.Figure()
        fig_oos.add_trace(
            go.Scatter(
                x=pred_view["date"],
                y=pred_view["prob"] * 100,
                mode="lines",
                line=dict(color="#4393c3", width=1),
                fill="tozeroy",
                fillcolor="rgba(67,147,195,0.15)",
                name="OOS prob",
            )
        )
        if "actual" in pred_view.columns:
            stress_days = pred_view[pred_view["actual"] == 1]
            fig_oos.add_trace(
                go.Scatter(
                    x=stress_days["date"],
                    y=[5] * len(stress_days),
                    mode="markers",
                    marker=dict(color="#d6604d", size=4, symbol="circle"),
                    name="Actual stress day",
                )
            )
        fig_oos.add_hline(y=50, line_dash="dash", line_color="grey", line_width=1)
        fig_oos.update_layout(
            height=280,
            margin=dict(t=10, b=10, l=10, r=10),
            yaxis=dict(range=[0, 105], ticksuffix="%"),
            xaxis_title=None,
            legend=dict(orientation="h", y=1.02, x=0),
        )
        st.plotly_chart(fig_oos, use_container_width=True)
    else:
        st.info("Run `scripts/02_train_logistic.py` to generate out-of-sample predictions.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — Stress Events
# ══════════════════════════════════════════════════════════════════════════

with tab_events:
    st.subheader("Stress Event Analysis")
    st.caption(
        f"A stress event onset is the first day the model probability exceeds "
        f"{STRESS_THRESHOLD*100:.0f}% after a period below that threshold."
    )

    if predictions.empty:
        st.info("No predictions found. Run `scripts/02_train_logistic.py` first.")
    else:
        events_df = _compute_stress_events(predictions, threshold=STRESS_THRESHOLD)
        if events_df.empty:
            st.info("No stress onsets detected in out-of-sample predictions.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total stress onsets", len(events_df))
            with m2:
                valid_lead = events_df["lead_days"].replace(0, np.nan).dropna()
                avg_lead = valid_lead.mean() if not valid_lead.empty else float("nan")
                st.metric(
                    "Mean lead time",
                    f"{avg_lead:.0f} days" if not pd.isna(avg_lead) else "N/A",
                )
            with m3:
                med_lead = valid_lead.median() if not valid_lead.empty else float("nan")
                st.metric(
                    "Median lead time",
                    f"{med_lead:.0f} days" if not pd.isna(med_lead) else "N/A",
                )
            with m4:
                st.metric("Max peak prob", f"{events_df['peak_prob'].max()*100:.1f}%")

            st.divider()
            col_hist_lead, col_event_table = st.columns([1, 1])

            with col_hist_lead:
                st.subheader("Lead-Time Distribution")
                lead_vals = events_df["lead_days"].replace(0, np.nan).dropna()
                if not lead_vals.empty:
                    lead_fig = go.Figure(
                        go.Histogram(x=lead_vals, nbinsx=20, marker_color="#4393c3", opacity=0.8)
                    )
                    lead_fig.add_vline(
                        x=float(lead_vals.mean()),
                        line_dash="dash",
                        line_color="#d6604d",
                        annotation_text=f"Mean: {lead_vals.mean():.0f}d",
                        annotation_position="top right",
                    )
                    lead_fig.update_layout(
                        height=300,
                        margin=dict(t=10, b=10, l=10, r=10),
                        xaxis_title="Lead days",
                        yaxis_title="Count",
                        showlegend=False,
                    )
                    st.plotly_chart(lead_fig, use_container_width=True)

            with col_event_table:
                st.subheader("Onset Table")
                display_events = events_df.copy()
                display_events["onset_date"] = display_events["onset_date"].dt.date
                display_events["peak_prob"] = (
                    (display_events["peak_prob"] * 100).round(1).astype(str) + "%"
                )
                display_events["lead_days"] = display_events["lead_days"].astype(int)
                display_events = display_events.rename(
                    columns={
                        "onset_date": "Onset Date",
                        "peak_prob": "Peak Prob",
                        "lead_days": "Lead Days",
                    }
                )
                st.dataframe(
                    display_events.sort_values("Onset Date", ascending=False),
                    use_container_width=True,
                )

            st.divider()
            st.subheader("Stress Probability with Onset Markers")
            pred_sorted = predictions.sort_values("date")
            fig_ev = go.Figure()
            fig_ev.add_trace(
                go.Scatter(
                    x=pred_sorted["date"],
                    y=pred_sorted["prob"] * 100,
                    mode="lines",
                    line=dict(color="#4393c3", width=1),
                    fill="tozeroy",
                    fillcolor="rgba(67,147,195,0.12)",
                    name="Stress probability",
                )
            )
            above = pred_sorted[pred_sorted["prob"] > STRESS_THRESHOLD]
            fig_ev.add_trace(
                go.Scatter(
                    x=above["date"],
                    y=above["prob"] * 100,
                    mode="markers",
                    marker=dict(color="rgba(214,96,77,0.3)", size=3),
                    name="Above threshold",
                )
            )
            for _, ev_row in events_df.iterrows():
                fig_ev.add_vline(
                    x=ev_row["onset_date"], line_color="#e74c3c", line_width=1.5, line_dash="dot"
                )
            fig_ev.add_hline(
                y=STRESS_THRESHOLD * 100,
                line_dash="dash",
                line_color="grey",
                line_width=1,
                annotation_text=f"{STRESS_THRESHOLD*100:.0f}% threshold",
                annotation_position="bottom right",
            )
            for date_str, label in MAJOR_EVENTS.items():
                ev_date = pd.Timestamp(date_str)
                if pred_sorted["date"].min() <= ev_date <= pred_sorted["date"].max():
                    fig_ev.add_annotation(
                        x=ev_date,
                        y=95,
                        text=label,
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="#888",
                        ax=0,
                        ay=-30,
                        font=dict(size=9, color="#555"),
                    )
            fig_ev.update_layout(
                height=380,
                margin=dict(t=20, b=10, l=10, r=10),
                yaxis=dict(range=[0, 105], ticksuffix="%"),
                xaxis_title=None,
                legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(fig_ev, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — Methods
# ══════════════════════════════════════════════════════════════════════════

with tab_methods:
    st.header("Methods")

    st.subheader("Research question")
    st.markdown(
        "> Do liquidity-based features improve prediction of S&P 500 drawdowns >= 5% "
        "in the next 20 days, beyond a VIX-only baseline?"
    )

    st.subheader("Target variable")
    st.markdown(
        "Binary label = 1 if SPY adj. close falls >= 5% at any point in the next "
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
| Volatility | VIX term ratio | VIX9D / VIX3M - inverted term structure signals near-term stress |
| Macro | Yield curve slope | 10Y - 2Y Treasury yield; negative = inverted curve |
| Technical | SPY drawdown | Distance from 252-day rolling high |
| Technical | Realised vol (20D) | Annualised standard deviation of daily returns |
        """
    )

    st.subheader("Live intraday estimate")
    st.markdown(
        "The **Intraday Estimate** on the Dashboard tab updates every 2 minutes during market "
        "hours. It takes the previous close's feature vector and replaces three features with "
        "live values from yfinance: `vix_5d_change` (live VIX vs. 5 trading days ago), "
        "`vix_term_ratio` (live VIX9D / VIX3M), and `spy_drawdown` (live SPY price vs. "
        "252-day rolling high). The four OHLC-dependent features (Amihud, CS, EDGE, realised "
        "vol) stay at yesterday's values since they require a completed trading session."
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
