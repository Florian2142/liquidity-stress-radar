"""Robustness battery for the drawdown-risk model.

Each function answers one falsifiable question about whether the headline
result survives a change of assumption: a different model, a different
sub-sample, a different drawdown threshold, or a different forecast horizon.
Every number is computed under the same walk-forward procedure as the main
model — nothing here is hard-coded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from liquidity_radar.eval.backtest import (
    baseline_score_predictions,
    normalise_scores,
    walk_forward_predict,
)
from liquidity_radar.eval.metrics import classification_report
from liquidity_radar.features.target import forward_drawdown_label
from liquidity_radar.models.logistic import (
    FEATURE_COLS,
    LIQUIDITY_FEATURES,
    VOLATILITY_FEATURES,
)

# Subperiods for regime-dependent analysis (inclusive date bounds).
SUBPERIODS: dict[str, tuple[str, str]] = {
    "Pre-GFC (≤2007)": ("1990-01-01", "2007-12-31"),
    "GFC (2008–2009)": ("2008-01-01", "2009-12-31"),
    "Post-GFC (2010–2019)": ("2010-01-01", "2019-12-31"),
    "COVID (2020)": ("2020-01-01", "2020-12-31"),
    "Recent (2021+)": ("2021-01-01", "2100-01-01"),
}


def model_comparison(combined: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample metrics for nested model specifications.

    Compares, on the identical target and evaluation window:
    VIX-level baseline, volatility-only, liquidity-only, full model, and the
    full model with liquidity features removed (to isolate their value).
    """
    specs = {
        "Volatility-only": VOLATILITY_FEATURES,
        "Liquidity-only": LIQUIDITY_FEATURES,
        "Full (all 9)": FEATURE_COLS,
        "Full − liquidity": [c for c in FEATURE_COLS if c not in LIQUIDITY_FEATURES],
    }

    rows: list[dict] = []

    # Naive baseline: the VIX level itself as a ranking score (no training).
    if "vix" in panel.columns:
        base = baseline_score_predictions(combined, panel["vix"])
        if not base.empty:
            scores = normalise_scores(base["prob"].to_numpy())
            rep = classification_report(base["actual"].to_numpy(), scores)
            rows.append({"model": "VIX-level baseline", "n_features": 1, **rep})

    for name, cols in specs.items():
        preds, _ = walk_forward_predict(combined, cols)
        if preds.empty:
            continue
        rep = classification_report(preds["actual"].to_numpy(), preds["prob"].to_numpy())
        rows.append({"model": name, "n_features": len(cols), **rep})

    return pd.DataFrame(rows)


def subperiod_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Headline metrics computed separately within each historical regime."""
    df = predictions.copy()
    df["date"] = pd.to_datetime(df["date"])
    rows: list[dict] = []
    for name, (lo, hi) in SUBPERIODS.items():
        mask = (df["date"] >= pd.Timestamp(lo)) & (df["date"] <= pd.Timestamp(hi))
        sub = df.loc[mask]
        if len(sub) < 30 or sub["actual"].nunique() < 2:
            rows.append(
                {
                    "period": name,
                    "n": len(sub),
                    "roc_auc": np.nan,
                    "n_positive": int(sub["actual"].sum()),
                }
            )
            continue
        rep = classification_report(sub["actual"].to_numpy(), sub["prob"].to_numpy())
        rows.append(
            {
                "period": name,
                "n": rep["n"],
                "roc_auc": rep["roc_auc"],
                "pr_auc": rep["pr_auc"],
                "base_rate": rep["base_rate"],
                "n_positive": rep["n_positive"],
            }
        )
    return pd.DataFrame(rows)


def threshold_sensitivity(
    features: pd.DataFrame,
    panel: pd.DataFrame,
    thresholds: tuple[float, ...] = (0.03, 0.05, 0.07, 0.10),
    horizon: int = 20,
) -> pd.DataFrame:
    """Re-run the full model for each drawdown-magnitude definition of stress."""
    rows: list[dict] = []
    for thr in thresholds:
        target = forward_drawdown_label(panel, horizon=horizon, threshold=thr)
        combined = features.join(target[["label"]]).dropna()
        if combined.empty or combined["label"].nunique() < 2:
            continue
        preds, _ = walk_forward_predict(combined, FEATURE_COLS)
        if preds.empty:
            continue
        rep = classification_report(preds["actual"].to_numpy(), preds["prob"].to_numpy())
        rows.append(
            {
                "threshold": thr,
                "base_rate": rep["base_rate"],
                "roc_auc": rep["roc_auc"],
                "pr_auc": rep["pr_auc"],
                "n": rep["n"],
            }
        )
    return pd.DataFrame(rows)


def horizon_sensitivity(
    features: pd.DataFrame,
    panel: pd.DataFrame,
    horizons: tuple[int, ...] = (5, 10, 20, 40),
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Re-run the full model for each forecast-horizon definition of stress."""
    rows: list[dict] = []
    for h in horizons:
        target = forward_drawdown_label(panel, horizon=h, threshold=threshold)
        combined = features.join(target[["label"]]).dropna()
        if combined.empty or combined["label"].nunique() < 2:
            continue
        preds, _ = walk_forward_predict(combined, FEATURE_COLS)
        if preds.empty:
            continue
        rep = classification_report(preds["actual"].to_numpy(), preds["prob"].to_numpy())
        rows.append(
            {
                "horizon_days": h,
                "base_rate": rep["base_rate"],
                "roc_auc": rep["roc_auc"],
                "pr_auc": rep["pr_auc"],
                "n": rep["n"],
            }
        )
    return pd.DataFrame(rows)


def coefficient_stability(fold_coefs: pd.DataFrame) -> pd.DataFrame:
    """Mean, dispersion, 95% interval, and sign-consistency of each coefficient.

    ``sign_consistency`` is the fraction of folds whose coefficient sign matches
    the mean sign — a simple, model-agnostic indicator of whether a feature's
    direction is stable out of sample.
    """
    mean = fold_coefs.mean()
    sd = fold_coefs.std()
    mean_sign = np.sign(mean)
    consistency = (np.sign(fold_coefs) == mean_sign).mean()
    return pd.DataFrame(
        {
            "mean": mean,
            "sd": sd,
            "ci_lo": mean - 1.96 * sd,
            "ci_hi": mean + 1.96 * sd,
            "sign_consistency": consistency,
        }
    ).reset_index(names="feature")


def auc_gain_bootstrap(
    full_preds: pd.DataFrame,
    baseline_preds: pd.DataFrame,
    n_reps: int = 1000,
    block_size: int = 252,
    seed: int = 42,
) -> dict[str, float]:
    """Paired block-bootstrap CI for the full model's ROC-AUC gain over baseline.

    Both prediction frames are aligned to their common dates so each bootstrap
    block resamples the same observations for both models, giving a paired
    estimate of the difference ``AUC_full − AUC_baseline``.
    """
    from sklearn.metrics import roc_auc_score

    f = full_preds.copy()
    b = baseline_preds.copy()
    f["date"] = pd.to_datetime(f["date"])
    b["date"] = pd.to_datetime(b["date"])
    merged = (
        f[["date", "prob", "actual"]]
        .merge(b[["date", "prob"]], on="date", suffixes=("_full", "_base"))
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )
    actual = merged["actual"].to_numpy()
    p_full = merged["prob_full"].to_numpy()
    p_base = normalise_scores(merged["prob_base"].to_numpy())

    point = float(roc_auc_score(actual, p_full) - roc_auc_score(actual, p_base))

    rng = np.random.default_rng(seed)
    n = len(merged)
    n_blocks = max(1, n // block_size)
    starts = np.arange(0, max(1, n - block_size + 1), block_size)
    gains: list[float] = []
    for _ in range(n_reps):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, min(s + block_size, n)) for s in chosen])[:n]
        a = actual[idx]
        if a.sum() == 0 or a.sum() == len(a):
            continue
        gains.append(float(roc_auc_score(a, p_full[idx]) - roc_auc_score(a, p_base[idx])))

    arr = np.array(gains)
    return {
        "gain": point,
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "prob_positive": float((arr > 0).mean()),
        "n_reps": len(gains),
    }
