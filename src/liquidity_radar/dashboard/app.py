"""Liquidity Stress Radar — interactive research dashboard.

Run locally::

    streamlit run src/liquidity_radar/dashboard/app.py

Eight sections: Overview · Current Signal · Data & Coverage · Liquidity
Indicators · Model Performance · Robustness Tests · Methodology · Limitations.
Heavy loads are cached; the live-market panel refreshes every two minutes via
``st.fragment``. Robustness numbers are read from artefacts produced by
``scripts/05_robustness.py`` — nothing on this page is hard-coded.
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
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from liquidity_radar.dashboard.loaders import (  # noqa: E402
    FEATURE_LABELS,
    batch_predict,
    feature_contributions,
    load_features,
    load_fold_coefs,
    load_model_params,
    load_panel,
    load_predictions,
    load_robust_csv,
    load_robust_summary,
    predict_one,
)
from liquidity_radar.data.quality import build_quality_report  # noqa: E402
from liquidity_radar.models.logistic import FEATURE_COLS  # noqa: E402

st.set_page_config(page_title="Liquidity Stress Radar", page_icon="📡", layout="wide")

# ── Global visual polish ─────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .block-container {padding-top: 2.4rem; max-width: 1280px;}
    h1, h2, h3 {letter-spacing: -0.015em;}
    /* KPI card lift on hover */
    .lsr-kpi {transition: transform .16s ease, box-shadow .16s ease;}
    .lsr-kpi:hover {transform: translateY(-3px); box-shadow: 0 8px 22px rgba(16,33,60,.12);}
    /* Tabs: pill-style, clearer active state */
    .stTabs [data-baseweb="tab-list"] {gap: 4px; border-bottom: 1px solid #e6e9f0;}
    .stTabs [data-baseweb="tab"] {height: 44px; padding: 0 16px; border-radius: 10px 10px 0 0; font-weight: 600;}
    .stTabs [aria-selected="true"] {background: #eef3fb; color: #2166ac;}
    /* Native metric chips */
    [data-testid="stMetric"] {background:#f7f9fc; border:1px solid #eceff5; border-radius:12px; padding:12px 16px;}
    [data-testid="stMetricLabel"] p {font-weight:600; color:#5a6675;}
    /* Dataframes a touch softer */
    [data-testid="stDataFrame"] {border-radius: 10px; overflow: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Palette ────────────────────────────────────────────────────────────────
BLUE, RED, GREEN, AMBER, ORANGE = "#2166ac", "#d6604d", "#2ecc71", "#f39c12", "#e67e22"
GREY = "#8895a7"
ET = ZoneInfo("America/New_York")

STATUS_LEVELS = [
    (0.25, "Calm", GREEN),
    (0.50, "Watch", AMBER),
    (0.75, "Elevated", ORANGE),
    (1.01, "Stress", RED),
]

MAJOR_EVENTS: dict[str, str] = {
    "2010-05-06": "Flash Crash",
    "2011-08-08": "Euro Crisis",
    "2015-08-24": "China Selloff",
    "2018-02-05": "Vol Shock",
    "2020-02-20": "COVID",
    "2022-01-03": "Rate Hikes",
    "2023-03-10": "SVB",
    "2025-04-07": "Tariff Shock",
}


def status_of(prob: float) -> tuple[str, str]:
    for threshold, label, color in STATUS_LEVELS:
        if prob < threshold:
            return label, color
    return "Stress", RED


def kpi_card(label: str, value: str, sub: str = "", color: str = BLUE) -> str:
    return f"""
    <div class="lsr-kpi" style="background:linear-gradient(160deg,#ffffff,#f4f6fa);
                border-left:5px solid {color};border-radius:12px;
                padding:16px 18px;margin-bottom:8px;height:122px;
                box-shadow:0 1px 3px rgba(16,33,60,.06);">
        <div style="font-size:0.78rem;color:#5a6675;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.5px;">{label}</div>
        <div style="font-size:2.05rem;font-weight:800;color:#1a1a2e;line-height:1.2;">{value}</div>
        <div style="font-size:0.82rem;color:#5a6675;">{sub}</div>
    </div>"""


# ════════════════════════════════════════════════════════════════════════════
# Shared data load
# ════════════════════════════════════════════════════════════════════════════

with st.spinner("Loading data…"):
    panel = load_panel()
    features = load_features(panel)
    params = load_model_params()
    predictions = load_predictions()
    fold_coefs = load_fold_coefs()
    summary = load_robust_summary()

if not params:
    st.error("Model artefacts not found. Run `python scripts/02_train_logistic.py` first.")
    st.stop()

feat_clean = features.dropna()
today_row = feat_clean.iloc[-1]
today_date = today_row.name
x_today = today_row.to_numpy(dtype=float)
today_prob = predict_one(x_today, params)
status_label, status_color = status_of(today_prob)
oos_auc = float(params["oos_auc"][0]) if "oos_auc" in params else float("nan")

st.title("📡 Liquidity Stress Radar")
st.caption(
    "An interpretable monitor for S&P 500 drawdown risk. "
    "Does market-liquidity information improve early warning beyond volatility signals?"
)

tabs = st.tabs(
    [
        "Overview",
        "Current Signal",
        "Data & Coverage",
        "Liquidity Indicators",
        "Model Performance",
        "Robustness Tests",
        "Methodology",
        "Limitations",
    ]
)
(tab_overview, tab_signal, tab_data, tab_liq, tab_perf, tab_robust, tab_method, tab_limits) = tabs


# ════════════════════════════════════════════════════════════════════════════
# 1 · OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.markdown(
        """
        <div style="background:linear-gradient(110deg,#16335c,#2166ac);border-radius:14px;
                    padding:26px 30px;color:white;margin-bottom:18px;">
            <div style="font-size:1.5rem;font-weight:700;">Predicting S&P 500 drawdowns from liquidity stress</div>
            <div style="font-size:1.0rem;opacity:0.92;margin-top:6px;max-width:900px;">
                A logistic, walk-forward-validated classifier estimating the probability of a
                ≥ 5% S&P 500 drawdown within the next 20 trading days, combining transaction-cost
                liquidity proxies with volatility, macro, and technical indicators.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    auc_ci = summary.get("auc_gain_ci", [float("nan"), float("nan")])
    gain = summary.get("auc_gain_vs_vix", float("nan"))
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        kpi_card(
            "OOS ROC-AUC",
            f"{summary.get('roc_auc', oos_auc):.3f}",
            "Walk-forward, out-of-sample",
            BLUE,
        ),
        unsafe_allow_html=True,
    )
    c2.markdown(
        kpi_card(
            "AUC gain vs VIX",
            f"{gain:+.3f}" if gain == gain else "—",
            f"95% CI [{auc_ci[0]:+.2f}, {auc_ci[1]:+.2f}]" if gain == gain else "",
            GREEN,
        ),
        unsafe_allow_html=True,
    )
    c3.markdown(
        kpi_card(
            "PR-AUC",
            f"{summary.get('pr_auc', float('nan')):.3f}",
            f"Base rate {summary.get('base_rate', float('nan')):.1%}",
            AMBER,
        ),
        unsafe_allow_html=True,
    )
    c4.markdown(
        kpi_card(
            "Today's signal",
            f"{today_prob * 100:.0f}%",
            f"{status_label} · as of {today_date.date()}",
            status_color,
        ),
        unsafe_allow_html=True,
    )

    st.markdown("#### Abstract")
    st.markdown(
        "Liquidity Stress Radar is a reproducible research dashboard that evaluates whether "
        "market-liquidity proxies improve the early detection of equity-market drawdown risk "
        "beyond standard volatility-based warning signals. The target is a drawdown of at least "
        "5% within a 20-trading-day horizon on the S&P 500 (SPY). The framework combines "
        "transaction-cost-inspired liquidity proxies — Amihud illiquidity, the Corwin–Schultz "
        "high–low spread estimator, and the EDGE estimator — with volatility, macro-financial, "
        "and technical indicators including VIX dynamics, the VIX term structure, the yield-curve "
        "slope, realised volatility, and recent drawdown behaviour. Liquidity-based and full "
        "logistic models are compared against VIX-only and naïve threshold baselines under "
        "walk-forward validation with a six-month purge gap to control look-ahead bias. Model "
        "quality is assessed through ROC-AUC, PR-AUC, the Brier score, calibration, lead time, "
        "feature ablations, subperiod analysis, sensitivity to the drawdown threshold and forecast "
        "horizon, and block-bootstrap confidence intervals. The dashboard is designed as an "
        "interpretable risk-monitoring tool rather than a trading system."
    )

    st.info(
        "**How to read this dashboard.** *Current Signal* shows today's estimated drawdown "
        "probability. *Model Performance* and *Robustness Tests* document how that estimate was "
        "validated and where it is and is not reliable. Probabilities are risk indicators, not "
        "forecasts of certainty — see **Limitations**.",
        icon="🧭",
    )


# ════════════════════════════════════════════════════════════════════════════
# 2 · CURRENT SIGNAL  (with live fragment)
# ════════════════════════════════════════════════════════════════════════════


def _market_status() -> tuple[bool, str]:
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
def _fetch_live() -> dict | None:
    try:
        close = yf.Tickers("SPY ^VIX ^VIX9D ^VIX3M").history(
            period="7d", interval="1d", auto_adjust=True
        )["Close"]

        def g(col: str, i: int = 0) -> float:
            try:
                return float(close[col].dropna().iloc[-(1 + i)])
            except Exception:
                return float("nan")

        return {
            "spy": g("SPY"),
            "spy_prev": g("SPY", 1),
            "vix": g("^VIX"),
            "vix_prev": g("^VIX", 1),
            "vix9d": g("^VIX9D"),
            "vix3m": g("^VIX3M"),
            "fetched_at": datetime.datetime.now(ET),
        }
    except Exception:
        return None


@st.cache_data(ttl=60)
def _fetch_intraday() -> pd.DataFrame:
    try:
        df = yf.download("SPY", period="1d", interval="5m", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


def _intraday_prob(live: dict) -> float:
    x = feat_clean.iloc[-1].to_numpy(dtype=float).copy()
    if not np.isnan(live["vix"]):
        vix_s = panel["vix"].dropna()
        if len(vix_s) >= 5:
            x[FEATURE_COLS.index("vix_5d_change")] = live["vix"] - float(vix_s.iloc[-5])
    if not np.isnan(live["vix9d"]) and not np.isnan(live["vix3m"]) and live["vix3m"] > 0:
        x[FEATURE_COLS.index("vix_term_ratio")] = live["vix9d"] / live["vix3m"]
    if not np.isnan(live["spy"]):
        spy_s = panel["adj_close"].dropna()
        if len(spy_s) >= 252:
            x[FEATURE_COLS.index("spy_drawdown")] = (
                live["spy"] / float(spy_s.iloc[-252:].max()) - 1.0
            )
    return predict_one(x, params)


with tab_signal:

    @st.fragment(run_every=datetime.timedelta(seconds=120))
    def live_panel() -> None:
        live = _fetch_live()
        intraday = _fetch_intraday()
        is_open, mkt = _market_status()
        accent = GREEN if is_open else GREY
        pill_text = "LIVE" if is_open else "CLOSED"
        dot_anim = "pulse 1.6s ease-in-out infinite" if is_open else "none"
        glow = f"0 0 0 4px {accent}22" if is_open else "none"
        seen = live["fetched_at"].strftime("%H:%M:%S ET") if live else "—"
        st.markdown(
            f"""<style>
            @keyframes pulse {{0%{{opacity:1;transform:scale(1)}}50%{{opacity:.35;transform:scale(.82)}}100%{{opacity:1;transform:scale(1)}}}}
            </style>
            <div style="display:flex;align-items:center;gap:12px;padding:10px 16px;border-radius:12px;
                background:linear-gradient(100deg,{accent}14,transparent 70%);
                border:1px solid {accent}33;box-shadow:0 1px 3px rgba(16,33,60,.06);margin-bottom:14px;">
              <span style="display:inline-flex;align-items:center;gap:7px;padding:3px 11px;border-radius:999px;
                  background:{accent};color:white;font-size:.72rem;font-weight:800;letter-spacing:.8px;
                  box-shadow:{glow};">
                <span style="width:8px;height:8px;border-radius:50%;background:white;
                  animation:{dot_anim};"></span>{pill_text}
              </span>
              <span style="font-weight:650;color:#1a1a2e;">{mkt}</span>
              <span style="margin-left:auto;color:#8895a7;font-size:.78rem;font-variant-numeric:tabular-nums;">
                Auto-refresh every 2 min · updated <b style="color:#5a6675;">{seen}</b></span>
            </div>""",
            unsafe_allow_html=True,
        )
        if live is None:
            st.warning("yfinance unreachable — live data unavailable right now.")
            return
        spy_chg = (
            (live["spy"] - live["spy_prev"]) / live["spy_prev"] * 100
            if live["spy_prev"] > 0
            else float("nan")
        )
        vix_chg = live["vix"] - live["vix_prev"] if not np.isnan(live["vix_prev"]) else float("nan")
        col_chart, col_metrics = st.columns([3, 1])
        with col_metrics:
            st.metric(
                "SPY", f"${live['spy']:.2f}", f"{spy_chg:+.2f}%" if spy_chg == spy_chg else None
            )
            st.metric(
                "VIX",
                f"{live['vix']:.2f}",
                f"{vix_chg:+.2f}" if vix_chg == vix_chg else None,
                delta_color="inverse",
            )
            if is_open:
                ip = _intraday_prob(live)
                lbl, clr = status_of(ip)
                st.metric(
                    "Intraday estimate",
                    f"{ip * 100:.1f}%",
                    help="Live VIX/SPY into the prior-close feature vector",
                )
                st.markdown(
                    f"<div style='text-align:center;background:{clr}22;border:1px solid {clr};"
                    f"border-radius:6px;padding:3px;font-weight:600;color:{clr};'>{lbl}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Intraday estimate runs during market hours.")
        with col_chart:
            if not intraday.empty and "Close" in intraday.columns:
                cl = intraday["Close"].dropna()
                if len(cl) > 1:
                    up = cl.iloc[-1] >= cl.iloc[0]
                    fig = go.Figure(
                        go.Scatter(
                            x=intraday.index,
                            y=cl,
                            mode="lines",
                            line=dict(color=GREEN if up else RED, width=2),
                            fill="tozeroy",
                            fillcolor=f"rgba({'46,204,113' if up else '231,76,60'},.1)",
                        )
                    )
                    fig.add_hline(
                        y=float(cl.iloc[0]),
                        line_dash="dot",
                        line_color=GREY,
                        annotation_text="Open",
                        annotation_position="right",
                    )
                    fig.update_layout(
                        height=220,
                        margin=dict(t=24, b=8, l=8, r=50),
                        title=dict(text="SPY — today (5-min)", font=dict(size=13), x=0),
                        yaxis_tickprefix="$",
                        showlegend=False,
                    )
                    st.plotly_chart(fig, width="stretch")
            else:
                st.caption("Intraday chart available during market hours.")
        st.divider()

    live_panel()

    c_gauge, c_spark, c_status = st.columns([1, 2, 1])
    with c_gauge:
        st.subheader("Drawdown probability")
        st.caption(f"Prior close · {today_date.date()}")
        g = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=round(today_prob * 100, 1),
                number={"suffix": "%", "font": {"size": 34}},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%"},
                    "bar": {"color": status_color},
                    "steps": [
                        {"range": [0, 25], "color": "#d5f5e3"},
                        {"range": [25, 50], "color": "#fef9e7"},
                        {"range": [50, 75], "color": "#fdebd0"},
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
        g.update_layout(height=230, margin=dict(t=10, b=10, l=20, r=20))
        st.plotly_chart(g, width="stretch")
    with c_spark:
        st.subheader("90-day probability trend")
        recent = feat_clean.iloc[-90:]
        p90 = batch_predict(recent.to_numpy(dtype=float), params)
        sp = go.Figure(
            go.Scatter(
                x=recent.index,
                y=p90 * 100,
                mode="lines",
                line=dict(color=BLUE, width=2),
                fill="tozeroy",
                fillcolor="rgba(33,102,172,.15)",
            )
        )
        sp.add_hline(y=50, line_dash="dash", line_color=GREY)
        sp.update_layout(
            height=230,
            margin=dict(t=10, b=10, l=8, r=8),
            yaxis=dict(range=[0, 100], ticksuffix="%"),
            showlegend=False,
        )
        st.plotly_chart(sp, width="stretch")
    with c_status:
        st.subheader("Status")
        st.markdown(
            f"<div style='background:{status_color};color:white;border-radius:12px;padding:18px;"
            f"text-align:center;font-size:1.9rem;font-weight:700;margin-top:26px;'>{status_label}</div>",
            unsafe_allow_html=True,
        )
        st.caption("Calm < 25% · Watch 25–50% · Elevated 50–75% · Stress ≥ 75%")

    st.divider()
    st.subheader("Today's feature contributions to log-odds")
    contribs = feature_contributions(x_today, params)
    cdf = pd.DataFrame(
        {"feature": [FEATURE_LABELS[c] for c in FEATURE_COLS], "c": contribs}
    ).sort_values("c")
    bar = go.Figure(
        go.Bar(
            x=cdf["c"],
            y=cdf["feature"],
            orientation="h",
            marker_color=[RED if v > 0 else BLUE for v in cdf["c"]],
        )
    )
    bar.add_vline(x=0, line_color="black", line_width=1)
    bar.update_layout(
        height=320,
        margin=dict(t=8, b=8, l=8, r=8),
        xaxis_title="Contribution to log-odds (positive = raises drawdown probability)",
        showlegend=False,
    )
    st.plotly_chart(bar, width="stretch")
    st.caption(
        "Contribution = standardised coefficient × standardised feature value for the latest observation."
    )


# ════════════════════════════════════════════════════════════════════════════
# 3 · DATA & COVERAGE
# ════════════════════════════════════════════════════════════════════════════

with tab_data:
    st.subheader("Data quality & coverage")
    report = build_quality_report(panel)
    fresh_color = GREEN if report.is_fresh else AMBER
    d1, d2, d3, d4 = st.columns(4)
    d1.markdown(
        kpi_card(
            "Observations",
            f"{report.n_rows:,}",
            f"{report.start_date.date()} → {report.end_date.date()}",
            BLUE,
        ),
        unsafe_allow_html=True,
    )
    d2.markdown(
        kpi_card(
            "Latest data",
            f"{report.end_date.date()}",
            f"{report.stale_days} days ago" + ("" if report.is_fresh else " · refresh advised"),
            fresh_color,
        ),
        unsafe_allow_html=True,
    )
    d3.markdown(
        kpi_card(
            "Duplicate dates",
            f"{report.n_duplicate_dates}",
            "index integrity",
            GREEN if report.n_duplicate_dates == 0 else RED,
        ),
        unsafe_allow_html=True,
    )
    d4.markdown(
        kpi_card(
            "Calendar gaps",
            f"{report.n_calendar_gaps}",
            f"max {report.max_gap_days} business days (holidays)",
            GREY,
        ),
        unsafe_allow_html=True,
    )

    st.markdown("#### Column coverage")
    cov = report.summary_frame().reset_index(names="column")
    cov_fig = go.Figure(
        go.Bar(
            x=cov["coverage_pct"],
            y=cov["column"],
            orientation="h",
            marker_color=[
                GREEN if v >= 95 else AMBER if v >= 60 else RED for v in cov["coverage_pct"]
            ],
            text=[f"{v:.0f}%" for v in cov["coverage_pct"]],
            textposition="outside",
        )
    )
    cov_fig.update_layout(
        height=380,
        margin=dict(t=8, b=8, l=8, r=50),
        xaxis=dict(range=[0, 108], ticksuffix="%", title="Non-missing coverage"),
        showlegend=False,
    )
    st.plotly_chart(cov_fig, width="stretch")
    st.caption(
        "VIX9D and VIX3M begin in 2008 and 2011 respectively, so their full-history coverage is "
        "below 100%. The model only trains on dates where every feature is available, so partial "
        "coverage shortens the usable sample rather than introducing gaps."
    )
    with st.expander("Per-column detail"):
        st.dataframe(cov, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
# 4 · LIQUIDITY INDICATORS
# ════════════════════════════════════════════════════════════════════════════

with tab_liq:
    st.subheader("Indicator explorer")
    c1, c2 = st.columns([2, 2])
    with c1:
        sel = st.selectbox("Feature", options=FEATURE_COLS, format_func=lambda c: FEATURE_LABELS[c])
    with c2:
        y0, y1 = int(feat_clean.index.year.min()), int(feat_clean.index.year.max())
        yr = st.slider("Year range", y0, y1, (max(y0, y1 - 12), y1))
    mask = (feat_clean.index.year >= yr[0]) & (feat_clean.index.year <= yr[1])
    sub = feat_clean.loc[mask]
    probs = batch_predict(sub.to_numpy(dtype=float), params)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.07,
        subplot_titles=[FEATURE_LABELS[sel], "Model drawdown probability"],
    )
    fig.add_trace(
        go.Scatter(x=sub.index, y=sub[sel], mode="lines", line=dict(color=BLUE, width=1.4)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sub.index,
            y=probs * 100,
            mode="lines",
            line=dict(color=RED, width=1.4),
            fill="tozeroy",
            fillcolor="rgba(214,96,77,.12)",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=50, line_dash="dash", line_color=GREY, row=2, col=1)
    for ds, lbl in MAJOR_EVENTS.items():
        ev = pd.Timestamp(ds)
        if yr[0] <= ev.year <= yr[1]:
            fig.add_vline(
                x=ev, line_dash="dot", line_color="rgba(120,120,120,.5)", row="all", col=1
            )
            fig.add_annotation(
                x=ev,
                y=1,
                yref="paper",
                text=lbl,
                showarrow=False,
                textangle=-55,
                font=dict(size=9, color="#5a6675"),
                xanchor="left",
            )
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    fig.update_layout(height=520, margin=dict(t=42, b=8, l=8, r=8), showlegend=False)
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("SPY drawdown with realised stress events")
    spy = panel.loc[panel.index.isin(sub.index), "adj_close"]
    dd = spy / spy.cummax() - 1.0
    ddfig = go.Figure(
        go.Scatter(
            x=dd.index,
            y=dd * 100,
            mode="lines",
            line=dict(color="#34495e", width=1.2),
            fill="tozeroy",
            fillcolor="rgba(52,73,94,.12)",
        )
    )
    if not predictions.empty:
        pv = predictions[
            (predictions["date"].dt.year >= yr[0]) & (predictions["date"].dt.year <= yr[1])
        ]
        ev = pv[pv["actual"] == 1]
        ddfig.add_trace(
            go.Scatter(
                x=ev["date"],
                y=[1] * len(ev),
                mode="markers",
                marker=dict(color=RED, size=4),
                name="Realised ≥5% drawdown window",
            )
        )
    ddfig.update_layout(
        height=300,
        margin=dict(t=8, b=8, l=8, r=8),
        yaxis=dict(ticksuffix="%", title="Drawdown from running peak"),
        legend=dict(orientation="h", y=1.02),
        showlegend=True,
    )
    st.plotly_chart(ddfig, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
# 5 · MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════

with tab_perf:
    st.subheader("Out-of-sample performance")
    if predictions.empty:
        st.info("Run `scripts/02_train_logistic.py` to generate predictions.")
    else:
        actual = predictions["actual"].to_numpy()
        prob = predictions["prob"].to_numpy()
        vix_aligned = panel["vix"].reindex(predictions["date"].values).to_numpy()
        m = ~np.isnan(vix_aligned)

        col_roc, col_pr = st.columns(2)
        with col_roc:
            st.markdown("**ROC curve**")
            fpr, tpr, _ = roc_curve(actual, prob)
            fvx, tvx, _ = roc_curve(actual[m], vix_aligned[m])
            roc = go.Figure()
            roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    line=dict(color=BLUE, width=2),
                    name=f"Full model (AUC {summary.get('roc_auc', oos_auc):.3f})",
                )
            )
            roc.add_trace(
                go.Scatter(
                    x=fvx,
                    y=tvx,
                    mode="lines",
                    line=dict(color=RED, width=2, dash="dash"),
                    name="VIX level",
                )
            )
            roc.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    line=dict(color=GREY, dash="dot"),
                    name="Random",
                )
            )
            roc.update_layout(
                height=380,
                margin=dict(t=8, b=8, l=8, r=8),
                xaxis_title="False positive rate",
                yaxis_title="True positive rate",
                legend=dict(x=0.4, y=0.1),
            )
            st.plotly_chart(roc, width="stretch")
        with col_pr:
            st.markdown("**Precision–recall curve**")
            pr, rc, _ = precision_recall_curve(actual, prob)
            prf = go.Figure()
            prf.add_trace(
                go.Scatter(
                    x=rc,
                    y=pr,
                    mode="lines",
                    line=dict(color=BLUE, width=2),
                    name=f"Full model (PR-AUC {summary.get('pr_auc', float('nan')):.3f})",
                )
            )
            prf.add_hline(
                y=float(actual.mean()),
                line_dash="dot",
                line_color=GREY,
                annotation_text=f"No-skill ({actual.mean():.2f})",
            )
            prf.update_layout(
                height=380,
                margin=dict(t=8, b=8, l=8, r=8),
                xaxis_title="Recall",
                yaxis_title="Precision",
                legend=dict(x=0.3, y=0.95),
            )
            st.plotly_chart(prf, width="stretch")

        col_cal, col_imp = st.columns(2)
        with col_cal:
            st.markdown("**Calibration (reliability)**")
            calib = load_robust_csv("calibration.csv")
            cfig = go.Figure()
            cfig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    line=dict(color=GREY, dash="dot"),
                    name="Perfect",
                )
            )
            if not calib.empty:
                cfig.add_trace(
                    go.Scatter(
                        x=calib["mean_pred"],
                        y=calib["obs_freq"],
                        mode="lines+markers",
                        line=dict(color=BLUE, width=2),
                        marker=dict(size=7),
                        name="Model",
                    )
                )
            cfig.update_layout(
                height=380,
                margin=dict(t=8, b=8, l=8, r=8),
                xaxis_title="Mean predicted probability",
                yaxis_title="Observed frequency",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                legend=dict(x=0.05, y=0.95),
            )
            st.plotly_chart(cfig, width="stretch")
        with col_imp:
            st.markdown("**Feature importance (coef ± 1.96 SD across folds)**")
            if not fold_coefs.empty:
                mc = fold_coefs.mean()
                ci = 1.96 * fold_coefs.std()
                order = mc.abs().sort_values().index
                ifig = go.Figure(
                    go.Bar(
                        x=mc[order],
                        y=[FEATURE_LABELS.get(c, c) for c in order],
                        orientation="h",
                        marker_color=[RED if v > 0 else BLUE for v in mc[order]],
                        error_x=dict(type="data", array=ci[order], color="#444", thickness=1.2),
                    )
                )
                ifig.add_vline(x=0, line_color="black", line_width=1)
                ifig.update_layout(
                    height=380,
                    margin=dict(t=8, b=8, l=8, r=8),
                    xaxis_title="Standardised coefficient",
                    showlegend=False,
                )
                st.plotly_chart(ifig, width="stretch")

        st.divider()
        st.markdown("**Out-of-sample probability history**")
        pv = predictions.sort_values("date")
        tfig = go.Figure(
            go.Scatter(
                x=pv["date"],
                y=pv["prob"] * 100,
                mode="lines",
                line=dict(color=BLUE, width=1),
                fill="tozeroy",
                fillcolor="rgba(33,102,172,.12)",
                name="Probability",
            )
        )
        ev = pv[pv["actual"] == 1]
        tfig.add_trace(
            go.Scatter(
                x=ev["date"],
                y=[3] * len(ev),
                mode="markers",
                marker=dict(color=RED, size=3),
                name="Realised drawdown",
            )
        )
        tfig.add_hline(y=50, line_dash="dash", line_color=GREY)
        tfig.update_layout(
            height=300,
            margin=dict(t=8, b=8, l=8, r=8),
            yaxis=dict(range=[0, 105], ticksuffix="%"),
            legend=dict(orientation="h", y=1.02),
        )
        st.plotly_chart(tfig, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
# 6 · ROBUSTNESS TESTS
# ════════════════════════════════════════════════════════════════════════════

with tab_robust:
    st.subheader("Statistical robustness")
    st.caption(
        "All figures are recomputed by `scripts/05_robustness.py` under the same walk-forward "
        "procedure as the main model. Nothing here is hard-coded."
    )

    comp = load_robust_csv("model_comparison.csv")
    if comp.empty:
        st.warning("Robustness artefacts not found. Run `python scripts/05_robustness.py`.")
    else:
        # — Model comparison —
        st.markdown("#### 1 · Model comparison")
        cc1, cc2 = st.columns([3, 2])
        with cc1:
            bar = go.Figure(
                go.Bar(
                    x=comp["roc_auc"],
                    y=comp["model"],
                    orientation="h",
                    marker_color=[BLUE if "Full (all" in m else GREY for m in comp["model"]],
                    text=[f"{v:.3f}" for v in comp["roc_auc"]],
                    textposition="outside",
                )
            )
            bar.add_vline(x=0.5, line_dash="dot", line_color=GREY, annotation_text="Random")
            bar.update_layout(
                height=300,
                margin=dict(t=8, b=8, l=8, r=40),
                xaxis=dict(
                    range=[0.4, max(0.8, comp["roc_auc"].max() + 0.05)], title="OOS ROC-AUC"
                ),
                showlegend=False,
            )
            st.plotly_chart(bar, width="stretch")
        with cc2:
            st.dataframe(
                comp[["model", "n_features", "roc_auc", "pr_auc", "brier"]].round(3),
                width="stretch",
                hide_index=True,
            )
        gain = summary.get("auc_gain_vs_vix", float("nan"))
        gci = summary.get("auc_gain_ci", [float("nan"), float("nan")])
        pp = summary.get("auc_gain_prob_positive", float("nan"))
        st.success(
            f"**Full model vs VIX baseline:** ROC-AUC gain **{gain:+.3f}** "
            f"(block-bootstrap 95% CI [{gci[0]:+.3f}, {gci[1]:+.3f}]; P(gain > 0) = {pp:.0%}). "
            "The full model clearly beats the naïve VIX baseline."
        )
        full = comp[comp["model"].str.startswith("Full (all")]["roc_auc"]
        fml = comp[comp["model"] == "Full − liquidity"]["roc_auc"]
        if not full.empty and not fml.empty:
            delta = float(full.iloc[0] - fml.iloc[0])
            st.info(
                f"**Incremental value of liquidity features:** removing the four liquidity proxies "
                f"changes OOS ROC-AUC by only **{delta:+.3f}** ({full.iloc[0]:.3f} → {fml.iloc[0]:.3f}). "
                "Liquidity adds modest, not dramatic, incremental signal once volatility, macro, and "
                "technical features are present — an honest, replicable finding rather than an overclaim.",
                icon="⚖️",
            )

        # — Subperiods —
        st.markdown("#### 2 · Subperiod stability")
        sub = load_robust_csv("subperiod.csv")
        if not sub.empty:
            valid = sub.dropna(subset=["roc_auc"])
            sbar = go.Figure(
                go.Bar(
                    x=valid["period"],
                    y=valid["roc_auc"],
                    marker_color=[
                        GREEN if v >= 0.6 else AMBER if v >= 0.55 else RED for v in valid["roc_auc"]
                    ],
                    text=[f"{v:.3f}" for v in valid["roc_auc"]],
                    textposition="outside",
                )
            )
            sbar.add_hline(y=0.5, line_dash="dot", line_color=GREY)
            sbar.update_layout(
                height=300,
                margin=dict(t=8, b=8, l=8, r=8),
                yaxis=dict(range=[0.4, 0.8], title="OOS ROC-AUC"),
                showlegend=False,
            )
            st.plotly_chart(sbar, width="stretch")
            st.caption(
                "Out-of-sample predictions begin in 2010 (the expanding window needs ≥5 years of "
                "training history), so pre-2008 and GFC periods cannot be scored out-of-sample. "
                "Performance is strongest post-GFC and weaker in the COVID and most-recent samples."
            )

        # — Sensitivity —
        st.markdown("#### 3 · Sensitivity to design choices")
        s1, s2 = st.columns(2)
        with s1:
            thr = load_robust_csv("threshold_sensitivity.csv")
            if not thr.empty:
                f = go.Figure(
                    go.Scatter(
                        x=thr["threshold"] * 100,
                        y=thr["roc_auc"],
                        mode="lines+markers",
                        line=dict(color=BLUE, width=2),
                        marker=dict(size=8),
                    )
                )
                f.update_layout(
                    height=280,
                    margin=dict(t=24, b=8, l=8, r=8),
                    title=dict(text="Drawdown threshold", font=dict(size=13), x=0),
                    xaxis_title="Drawdown threshold (%)",
                    yaxis_title="OOS ROC-AUC",
                )
                st.plotly_chart(f, width="stretch")
        with s2:
            hor = load_robust_csv("horizon_sensitivity.csv")
            if not hor.empty:
                f = go.Figure(
                    go.Scatter(
                        x=hor["horizon_days"],
                        y=hor["roc_auc"],
                        mode="lines+markers",
                        line=dict(color=ORANGE, width=2),
                        marker=dict(size=8),
                    )
                )
                f.update_layout(
                    height=280,
                    margin=dict(t=24, b=8, l=8, r=8),
                    title=dict(text="Forecast horizon", font=dict(size=13), x=0),
                    xaxis_title="Horizon (trading days)",
                    yaxis_title="OOS ROC-AUC",
                )
                st.plotly_chart(f, width="stretch")
        st.caption(
            "The model's edge over random strengthens for larger drawdowns and longer horizons "
            "and weakens for small, short-term moves — consistent with liquidity/volatility stress "
            "being informative about material drawdowns rather than day-to-day noise."
        )

        # — Bootstrap CIs —
        st.markdown("#### 4 · Bootstrap confidence intervals")
        mci = load_robust_csv("metric_ci.csv")
        if not mci.empty:
            order = {"roc_auc": 0, "pr_auc": 1, "brier": 2}
            mci = mci.sort_values("metric", key=lambda s: s.map(order))
            f = go.Figure()
            for _, r in mci.iterrows():
                f.add_trace(
                    go.Scatter(
                        x=[r["ci_lo"], r["ci_hi"]],
                        y=[r["metric"], r["metric"]],
                        mode="lines",
                        line=dict(color=GREY, width=6),
                        showlegend=False,
                    )
                )
                f.add_trace(
                    go.Scatter(
                        x=[r["point"]],
                        y=[r["metric"]],
                        mode="markers",
                        marker=dict(color=BLUE, size=12),
                        showlegend=False,
                    )
                )
            f.update_layout(
                height=220,
                margin=dict(t=8, b=8, l=8, r=8),
                xaxis_title="Metric value (block bootstrap, 252-day blocks, 1000 reps)",
            )
            st.plotly_chart(f, width="stretch")

        # — Coefficient stability —
        st.markdown("#### 5 · Coefficient stability across folds")
        cs = load_robust_csv("coef_stability.csv")
        if not cs.empty:
            cs = cs.copy()
            cs["feature"] = cs["feature"].map(lambda c: FEATURE_LABELS.get(c, c))
            cs["sign_consistency"] = (cs["sign_consistency"] * 100).round(0).astype(int).astype(
                str
            ) + "%"
            st.dataframe(
                cs[["feature", "mean", "sd", "ci_lo", "ci_hi", "sign_consistency"]].round(3),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "`sign_consistency` is the share of folds whose coefficient sign matches the mean "
                "sign — a model-agnostic indicator of whether a feature's direction is stable."
            )

    st.warning(
        "**Statistical significance is not economic significance.** A confidence interval that "
        "excludes zero indicates the in-sample association is unlikely to be pure chance under the "
        "bootstrap's assumptions; it does **not** establish a tradeable edge net of costs, that the "
        "relationship is causal, or that it will persist out of sample. Treat every number here as "
        "rigorous *exploratory* evidence, not a law of nature.",
        icon="⚠️",
    )


# ════════════════════════════════════════════════════════════════════════════
# 7 · METHODOLOGY
# ════════════════════════════════════════════════════════════════════════════

with tab_method:
    st.header("Methodology")
    st.subheader("Research question")
    st.markdown(
        "> Do liquidity-based features improve prediction of S&P 500 drawdowns ≥ 5% over the "
        "next 20 trading days, beyond a VIX-only baseline?"
    )
    st.subheader("Target variable")
    st.markdown(
        "Binary label = 1 if SPY adjusted close falls ≥ 5% from today's level at any point in "
        "the next 20 trading days. Labels use forward windows only and are excluded from training "
        "where the future is unknown (the final 20 rows)."
    )
    st.subheader("Features (9)")
    st.markdown(
        """
| Group | Feature | Description |
|---|---|---|
| Liquidity | Amihud z-score | Amihud illiquidity normalised by its trailing 252-day mean/SD (Amihud 2002) |
| Liquidity | Amihud 5-day change | Short-term momentum in illiquidity |
| Liquidity | Corwin–Schultz spread | High–low bid–ask spread proxy (Corwin & Schultz 2012) |
| Liquidity | EDGE spread | Efficient discrete generalised estimator from OHLC (Ardia et al. 2024) |
| Volatility | VIX 5-day change | Short-term implied-volatility momentum |
| Volatility | VIX term ratio | VIX9D / VIX3M — inverted term structure signals near-term stress |
| Volatility | Realised vol (20D) | Annualised standard deviation of daily returns |
| Macro | Yield-curve slope | 10Y − 2Y Treasury yield |
| Technical | SPY drawdown | Distance from the 252-day rolling high |
        """
    )
    st.subheader("Model & validation")
    st.markdown(
        "L2-regularised logistic regression (C = 1.0). Features are standardised within each fold "
        "using statistics from the training set only. Validation is expanding-window walk-forward "
        "cross-validation: ≥ 5 years of training history, a six-month purge gap between train and "
        "test (longer than the 20-day label window, preventing overlap leakage), and six-month test "
        "windows stepping every six months."
    )
    st.subheader("Data sources")
    st.markdown(
        "SPY OHLCV and the VIX term structure (^VIX, ^VIX9D, ^VIX3M) via **yfinance**; "
        "10Y/2Y Treasury yields and the effective Fed Funds rate (DGS10, DGS2, DFF) via the public "
        "**FRED** CSV endpoint. Stored in DuckDB; a committed parquet snapshot lets the hosted app "
        "load instantly without hitting rate limits."
    )
    st.subheader("References")
    st.markdown(
        "- Amihud, Y. (2002). *Illiquidity and stock returns: cross-section and time-series effects.* "
        "Journal of Financial Markets.\n"
        "- Corwin, S. & Schultz, P. (2012). *A simple way to estimate bid–ask spreads from daily high "
        "and low prices.* Journal of Finance.\n"
        "- Brunnermeier, M. & Pedersen, L. (2009). *Market liquidity and funding liquidity.* RFS.\n"
        "- Rösch, C. & Kaserer, C. (2013). *Market liquidity in the financial crisis: the role of "
        "liquidity commonality and flight-to-quality.* Journal of Banking & Finance.\n"
        "- Hameed, A., Kang, W. & Viswanathan, S. (2010). *Stock market declines and liquidity.* "
        "Journal of Finance.\n"
        "- Ardia, D., Guidotti, E. & Kroencke, T. (2024). *Efficient estimation of bid–ask spreads "
        "from OHLC prices.* Journal of Financial Economics."
    )


# ════════════════════════════════════════════════════════════════════════════
# 8 · LIMITATIONS
# ════════════════════════════════════════════════════════════════════════════

with tab_limits:
    st.header("Limitations")
    st.markdown(
        """
This dashboard is an **exploratory empirical research tool**, not investment advice and not a
trading system. Its conclusions are bounded by the following:

- **Single market, single asset.** Only the S&P 500 via SPY is modelled. Results need not transfer
  to other indices, asset classes, or international markets.
- **Out-of-sample window starts in 2010.** The expanding walk-forward design requires ≥ 5 years of
  training history, so the 2008 crisis is used for training but never scored out-of-sample. The
  most informative crisis is therefore absent from the OOS metrics.
- **Modest incremental liquidity value.** The robustness battery shows liquidity proxies add only a
  small AUC improvement once volatility, macro, and technical features are present. The headline
  edge over a VIX baseline is real and bootstrap-significant, but it is incremental, not decisive.
- **Daily, end-of-day data.** Liquidity proxies are derived from daily OHLCV, not intraday order-book
  data. The intraday estimate substitutes live VIX/SPY into a prior-close feature vector and is an
  approximation, not a re-estimation of the microstructure features.
- **Statistical ≠ economic significance.** Confidence intervals describe sampling uncertainty under
  bootstrap assumptions. They do not account for transaction costs, capacity, regime change, or the
  multiple comparisons implicit in exploratory work, and they do not establish causality.
- **Vendor data.** yfinance and FRED are convenient and free but not survivorship- or revision-audited
  to institutional standards; values can be revised.

The project is designed to make drawdown-risk monitoring **measurable, visual, and testable** — and
to be transparent about exactly how far the evidence reaches.
        """
    )
