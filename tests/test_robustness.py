"""Tests for the data-quality, backtest, and robustness layers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from liquidity_radar.data.quality import build_quality_report  # noqa: E402
from liquidity_radar.eval.backtest import walk_forward_predict  # noqa: E402
from liquidity_radar.eval.metrics import (  # noqa: E402
    bootstrap_metric_ci,
    calibration_table,
    classification_report,
)
from liquidity_radar.eval.robustness import coefficient_stability  # noqa: E402
from liquidity_radar.features.build import build_feature_matrix  # noqa: E402
from liquidity_radar.features.target import forward_drawdown_label  # noqa: E402
from liquidity_radar.models.logistic import FEATURE_COLS  # noqa: E402


def _make_panel(n: int = 2200) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2008-01-01", periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.011, n)))
    vix = np.clip(15 + rng.normal(0, 4, n).cumsum() / np.sqrt(np.arange(1, n + 1)), 9, 80)
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, n)),
            "high": close * (1 + np.abs(rng.normal(0, 0.006, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.006, n))),
            "close": close,
            "adj_close": close,
            "volume": rng.integers(50_000_000, 200_000_000, n),
            "vix": vix,
            "vix9d": vix * 0.97,
            "vix3m": vix * 1.04,
            "yield_10y": 3.0 + rng.normal(0, 0.1, n).cumsum() / np.sqrt(np.arange(1, n + 1)),
            "yield_2y": 2.5 + rng.normal(0, 0.1, n).cumsum() / np.sqrt(np.arange(1, n + 1)),
            "fed_funds": 2.0 + rng.normal(0, 0.05, n).cumsum() / np.sqrt(np.arange(1, n + 1)),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


# ── Data quality ───────────────────────────────────────────────────────────


def test_quality_report_clean_panel() -> None:
    panel = _make_panel(500)
    report = build_quality_report(panel, asof=panel.index.max())
    assert report.n_rows == 500
    assert report.n_duplicate_dates == 0
    assert report.start_date == panel.index.min()
    assert report.end_date == panel.index.max()
    assert report.stale_days == 0 and report.is_fresh


def test_quality_report_detects_duplicates() -> None:
    panel = _make_panel(100)
    dupe = pd.concat([panel, panel.iloc[[50]]])
    report = build_quality_report(dupe)
    assert report.n_duplicate_dates == 1


def test_quality_report_coverage_fraction() -> None:
    panel = _make_panel(100)
    panel.loc[panel.index[:10], "vix9d"] = np.nan
    report = build_quality_report(panel)
    assert report.missing_by_column["vix9d"] == 10
    assert abs(report.coverage_by_column["vix9d"] - 0.9) < 1e-9


# ── Feature build / no look-ahead ──────────────────────────────────────────


def test_feature_matrix_columns() -> None:
    panel = _make_panel(800)
    feat = build_feature_matrix(panel)
    assert list(feat.columns) == FEATURE_COLS
    feat_raw = build_feature_matrix(panel, include_raw_amihud=True)
    assert feat_raw.columns[0] == "amihud"


def test_features_have_no_lookahead_warmup() -> None:
    """Rolling features must be NaN until their window is full (no peeking ahead)."""
    panel = _make_panel(800)
    feat = build_feature_matrix(panel)
    # amihud_zscore needs 20 (amihud) + 252 (zscore) warmup → deep leading NaN block.
    assert feat["amihud_zscore"].iloc[:250].isna().all()
    # realised vol needs 20 observations.
    assert feat["realized_vol_20d"].iloc[:19].isna().all()


def test_target_excludes_unknown_future() -> None:
    panel = _make_panel(300)
    target = forward_drawdown_label(panel, horizon=20, threshold=0.05)
    # The final horizon-1 rows cannot be labelled (future unknown).
    assert target["label"].iloc[-19:].isna().all()


# ── Backtest ───────────────────────────────────────────────────────────────


def _combined(panel: pd.DataFrame) -> pd.DataFrame:
    feat = build_feature_matrix(panel)
    target = forward_drawdown_label(panel)
    return feat.join(target[["label"]]).dropna()


def test_walk_forward_schema_and_no_leakage() -> None:
    combined = _combined(_make_panel(2200))
    preds, coefs = walk_forward_predict(combined, FEATURE_COLS)
    assert list(preds.columns) == ["date", "prob", "actual", "fold"]
    assert (preds["prob"] >= 0).all() and (preds["prob"] <= 1).all()
    assert list(coefs.columns) == FEATURE_COLS
    # No fold may evaluate on dates at or before its training cutoff: folds are ordered
    # and each fold's min test date must exceed the previous fold's data — checked inside
    # walk_forward_predict via assertion; here we confirm it produced multiple folds.
    assert preds["fold"].nunique() >= 2


# ── Metrics ────────────────────────────────────────────────────────────────


def test_classification_report_keys_and_ranges() -> None:
    rng = np.random.default_rng(0)
    actual = rng.integers(0, 2, 500)
    prob = rng.random(500)
    rep = classification_report(actual, prob)
    for key in ("roc_auc", "pr_auc", "brier", "precision", "recall", "f1", "base_rate"):
        assert key in rep
    assert 0 <= rep["roc_auc"] <= 1
    assert 0 <= rep["precision"] <= 1


def test_calibration_table_shape() -> None:
    rng = np.random.default_rng(1)
    actual = rng.integers(0, 2, 1000)
    prob = rng.random(1000)
    calib = calibration_table(actual, prob, n_bins=10)
    assert set(calib.columns) == {"bin_mid", "mean_pred", "obs_freq", "count"}
    assert calib["count"].sum() == 1000
    assert (calib["obs_freq"].between(0, 1)).all()


def test_bootstrap_ci_orders_and_brackets_point() -> None:
    rng = np.random.default_rng(2)
    n = 1000
    actual = rng.integers(0, 2, n)
    # Give the score real signal so AUC > 0.5 and the CI is informative.
    prob = np.clip(0.3 * actual + rng.random(n) * 0.7, 0, 1)
    ci = bootstrap_metric_ci(actual, prob, metric="roc_auc", n_reps=200, block_size=100)
    assert ci["ci_lo"] <= ci["point"] <= ci["ci_hi"]
    assert ci["n_reps"] > 0


# ── Coefficient stability ──────────────────────────────────────────────────


def test_coefficient_stability_output() -> None:
    rng = np.random.default_rng(4)
    coefs = pd.DataFrame(rng.normal(0, 1, (20, len(FEATURE_COLS))), columns=FEATURE_COLS)
    out = coefficient_stability(coefs)
    assert set(out.columns) == {"feature", "mean", "sd", "ci_lo", "ci_hi", "sign_consistency"}
    assert (out["sign_consistency"].between(0, 1)).all()
    assert (out["ci_lo"] <= out["ci_hi"]).all()
