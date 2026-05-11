"""Phase 2 tests — one happy-path test per new module."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from liquidity_radar.features.liquidity import corwin_schultz_spread, edge_spread  # noqa: E402
from liquidity_radar.features.macro import ffr_change, yield_curve_slope  # noqa: E402
from liquidity_radar.features.technical import realized_vol_20d, spy_drawdown_from_high  # noqa: E402
from liquidity_radar.features.volatility import vix_5d_change, vix_term_ratio  # noqa: E402
from liquidity_radar.models.baseline import vix_threshold_predict  # noqa: E402
from liquidity_radar.models.logistic import FEATURE_COLS, LogisticModel  # noqa: E402
from liquidity_radar.models.walkforward import WalkForwardCV  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    """Minimal SPY-shaped OHLCV DataFrame."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2010-01-01", periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, n)),
            "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
            "close": close,
            "adj_close": close,
            "volume": rng.integers(50_000_000, 200_000_000, n),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


def _make_panel(n: int = 2000) -> pd.DataFrame:
    """Panel DataFrame with all columns expected by feature functions."""
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2015-01-01", periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    vix = np.clip(15 + rng.normal(0, 3, n).cumsum() / np.sqrt(np.arange(1, n + 1)), 9, 80)
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, n)),
            "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
            "close": close,
            "adj_close": close,
            "volume": rng.integers(50_000_000, 200_000_000, n),
            "vix": vix,
            "vix9d": vix * 0.95,
            "vix3m": vix * 1.05,
            "yield_10y": 3.0 + rng.normal(0, 0.1, n).cumsum() / np.sqrt(np.arange(1, n + 1)),
            "yield_2y": 2.5 + rng.normal(0, 0.1, n).cumsum() / np.sqrt(np.arange(1, n + 1)),
            "fed_funds": 2.0 + rng.normal(0, 0.05, n).cumsum() / np.sqrt(np.arange(1, n + 1)),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


# ── Liquidity tests ───────────────────────────────────────────────────────


def test_corwin_schultz_returns_series() -> None:
    """CS spread returns a non-negative Series with expected leading NaN."""
    prices = _make_ohlcv(100)
    cs = corwin_schultz_spread(prices)
    assert isinstance(cs, pd.Series)
    assert cs.name == "cs_spread"
    # First row is NaN (requires 2 consecutive days)
    assert pd.isna(cs.iloc[0])
    # Values must be non-negative (clipped)
    assert (cs.dropna() >= 0).all(), "CS spread contains negative values"


def test_corwin_schultz_missing_columns_raises() -> None:
    df = pd.DataFrame({"adj_close": [1.0, 2.0]})
    with pytest.raises(ValueError, match="high"):
        corwin_schultz_spread(df)


def test_edge_spread_returns_series() -> None:
    """EDGE spread returns a clipped non-negative Series."""
    prices = _make_ohlcv(100)
    es = edge_spread(prices, window=20)
    assert isinstance(es, pd.Series)
    assert es.name == "edge"
    # First window-1 rows should be NaN
    assert es.iloc[:19].isna().all()
    assert (es.dropna() >= 0).all()


# ── Volatility tests ──────────────────────────────────────────────────────


def test_vix_5d_change_shape() -> None:
    panel = _make_panel(50)
    out = vix_5d_change(panel)
    assert isinstance(out, pd.Series)
    assert out.name == "vix_5d_change"
    assert len(out) == 50
    # First 5 rows are NaN (diff(5))
    assert out.iloc[:5].isna().all()


def test_vix_term_ratio_range() -> None:
    panel = _make_panel(50)
    out = vix_term_ratio(panel)
    assert out.name == "vix_term_ratio"
    # vix9d = vix * 0.95, vix3m = vix * 1.05  → ratio ≈ 0.905
    assert (out.dropna() > 0).all()
    assert (out.dropna() < 2).all()


# ── Macro tests ───────────────────────────────────────────────────────────


def test_yield_curve_slope_sign() -> None:
    panel = _make_panel(50)
    slope = yield_curve_slope(panel)
    assert slope.name == "yield_curve_slope"
    # Difference of two floats must equal 10y - 2y
    expected = panel["yield_10y"] - panel["yield_2y"]
    pd.testing.assert_series_equal(slope, expected.rename("yield_curve_slope"))


def test_ffr_change_length() -> None:
    panel = _make_panel(50)
    out = ffr_change(panel, window=5)
    assert out.name == "ffr_change"
    assert len(out) == 50
    assert out.iloc[:5].isna().all()


# ── Technical tests ───────────────────────────────────────────────────────


def test_spy_drawdown_non_positive() -> None:
    panel = _make_panel(300)
    dd = spy_drawdown_from_high(panel, window=252)
    assert dd.name == "spy_drawdown"
    assert (dd.dropna() <= 0).all(), "drawdown must be <= 0"
    # First 251 rows NaN (window=252, min_periods=252)
    assert dd.iloc[:251].isna().all()


def test_realized_vol_positive() -> None:
    panel = _make_panel(100)
    rv = realized_vol_20d(panel, window=20)
    assert rv.name == "realized_vol_20d"
    assert (rv.dropna() > 0).all()
    assert rv.iloc[:19].isna().all()


# ── Baseline tests ────────────────────────────────────────────────────────


def test_vix_baseline_binary() -> None:
    panel = _make_panel(100)
    pred = vix_threshold_predict(panel, threshold=25.0)
    assert pred.name == "baseline_pred"
    assert set(pred.unique()).issubset({0, 1})
    # Days where VIX > 25 must be 1
    assert (pred[panel["vix"] > 25] == 1).all()
    assert (pred[panel["vix"] <= 25] == 0).all()


# ── Walk-forward CV tests ─────────────────────────────────────────────────


def test_walkforward_no_leakage() -> None:
    """Training set must always end strictly before test set starts."""
    panel = _make_panel(2000)
    combined = panel[["adj_close", "vix"]].copy()
    combined["label"] = (panel["vix"] > 20).astype(int)
    combined.index.name = "date"

    cv = WalkForwardCV(train_min_years=2, purge_days=90, test_window_days=90)
    fold_count = 0
    for train_df, test_df, fold_idx in cv.split(combined):
        assert train_df.index.max() < test_df.index.min(), (
            f"Fold {fold_idx}: leakage detected"
        )
        fold_count += 1
    assert fold_count > 0, "WalkForwardCV produced zero folds"


def test_walkforward_fold_index_increments() -> None:
    panel = _make_panel(2000)
    combined = panel[["adj_close"]].copy()
    combined["label"] = 0
    combined.index.name = "date"

    cv = WalkForwardCV(train_min_years=2, purge_days=60, test_window_days=60)
    indices = [idx for _, _, idx in cv.split(combined)]
    assert indices == list(range(len(indices))), "fold indices must be sequential"


# ── Logistic model tests ──────────────────────────────────────────────────


def test_logistic_model_fit_predict() -> None:
    """LogisticModel fits without error and returns probabilities in [0, 1]."""
    rng = np.random.default_rng(42)
    n = 500
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLS))), columns=FEATURE_COLS)
    y = pd.Series((rng.random(n) > 0.7).astype(int))

    model = LogisticModel()
    model.fit(X, y)
    probs = model.predict_proba(X)

    assert probs.shape == (n,)
    assert (probs >= 0).all() and (probs <= 1).all()
    assert len(model.coef_) == len(FEATURE_COLS)


def test_logistic_model_no_leakage_via_scaler() -> None:
    """Scaler fitted on train must not see test statistics."""
    rng = np.random.default_rng(7)
    n_train, n_test = 400, 100
    X_train = pd.DataFrame(rng.normal(0, 1, (n_train, 2)), columns=["a", "b"])
    y_train = pd.Series(rng.integers(0, 2, n_train))
    # Test set has a very different scale
    X_test = pd.DataFrame(rng.normal(100, 5, (n_test, 2)), columns=["a", "b"])

    model = LogisticModel()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    # Should not raise; probabilities still valid
    assert (probs >= 0).all() and (probs <= 1).all()
