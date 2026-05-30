"""Publication-quality plots for the evaluation report.

Four plots produced:
1. ROC curve overlay (logistic model vs VIX-score vs random baseline)
2. Precision-recall curve overlay
3. Feature importance bar chart with 95% CI across walk-forward folds
4. Time-series of predicted probability with stress-event shading
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

DPI = 300
FIGSIZE_WIDE = (9, 5)
FIGSIZE_SQUARE = (7, 6)


def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {path}")


def plot_roc_curves(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
) -> None:
    """ROC curve with three lines: logistic model, VIX-score baseline, random.

    Parameters
    ----------
    predictions : DataFrame
        Columns: date, prob, actual, fold.
    panel : DataFrame
        Full features panel (DatetimeIndex named 'date'); must contain 'vix'.
    out_path : Path
        Destination PNG file.
    """
    df = predictions.copy()
    df["date"] = pd.to_datetime(df["date"])
    vix_aligned = panel["vix"].reindex(df["date"].values).to_numpy()

    actual = df["actual"].to_numpy()
    prob = df["prob"].to_numpy()

    # Drop rows where VIX is NaN
    mask = ~np.isnan(vix_aligned)
    actual_m, prob_m, vix_m = actual[mask], prob[mask], vix_aligned[mask]

    fpr_lr, tpr_lr, _ = roc_curve(actual_m, prob_m)
    fpr_vx, tpr_vx, _ = roc_curve(actual_m, vix_m)
    auc_lr = auc(fpr_lr, tpr_lr)
    auc_vx = auc(fpr_vx, tpr_vx)

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    ax.plot(fpr_lr, tpr_lr, lw=2, label=f"Logistic (AUC = {auc_lr:.3f})", color="#2166ac")
    ax.plot(fpr_vx, tpr_vx, lw=2, ls="--", label=f"VIX score (AUC = {auc_vx:.3f})", color="#d6604d")
    ax.plot([0, 1], [0, 1], lw=1, ls=":", color="grey", label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Out-of-Sample Walk-Forward CV")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_path)


def plot_pr_curves(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
) -> None:
    """Precision-recall curves: logistic model vs VIX-score baseline."""
    df = predictions.copy()
    df["date"] = pd.to_datetime(df["date"])
    vix_aligned = panel["vix"].reindex(df["date"].values).to_numpy()

    actual = df["actual"].to_numpy()
    prob = df["prob"].to_numpy()

    mask = ~np.isnan(vix_aligned)
    actual_m, prob_m, vix_m = actual[mask], prob[mask], vix_aligned[mask]

    prec_lr, rec_lr, _ = precision_recall_curve(actual_m, prob_m)
    prec_vx, rec_vx, _ = precision_recall_curve(actual_m, vix_m)
    ap_lr = auc(rec_lr, prec_lr)
    ap_vx = auc(rec_vx, prec_vx)
    base_rate = actual_m.mean()

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    ax.plot(rec_lr, prec_lr, lw=2, label=f"Logistic (AP = {ap_lr:.3f})", color="#2166ac")
    ax.plot(rec_vx, prec_vx, lw=2, ls="--", label=f"VIX score (AP = {ap_vx:.3f})", color="#d6604d")
    ax.axhline(base_rate, lw=1, ls=":", color="grey", label=f"No-skill (AP = {base_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Out-of-Sample Walk-Forward CV")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_path)


def plot_feature_importance(
    fold_coefs: pd.DataFrame,
    out_path: Path,
) -> None:
    """Horizontal bar chart of mean logistic coefficients ± 1.96 SD across folds.

    Parameters
    ----------
    fold_coefs : DataFrame
        Rows = folds, columns = feature names. Values = standardised coefficients.
    out_path : Path
        Destination PNG file.
    """
    mean_coef = fold_coefs.mean()
    sd_coef = fold_coefs.std()
    ci = 1.96 * sd_coef

    # Sort by absolute mean coefficient
    order = mean_coef.abs().sort_values().index
    mean_sorted = mean_coef[order]
    ci_sorted = ci[order]

    colors = ["#d6604d" if v > 0 else "#2166ac" for v in mean_sorted]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.barh(range(len(mean_sorted)), mean_sorted.values, color=colors, alpha=0.85)
    ax.errorbar(
        mean_sorted.values,
        range(len(mean_sorted)),
        xerr=ci_sorted.values,
        fmt="none",
        color="black",
        capsize=4,
        lw=1.2,
    )
    ax.set_yticks(range(len(mean_sorted)))
    ax.set_yticklabels(mean_sorted.index.tolist())
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Standardised Coefficient (mean ± 1.96 SD across folds)")
    ax.set_title("Feature Importance — Logistic Regression Walk-Forward CV")
    ax.grid(axis="x", alpha=0.3)
    _savefig(fig, out_path)


def plot_probability_timeseries(
    predictions: pd.DataFrame,
    out_path: Path,
) -> None:
    """Line plot of predicted probability over time with stress-event shading.

    Stress events (actual=1) are shaded in light red.
    The 0.5 decision threshold is shown as a dashed line.

    Parameters
    ----------
    predictions : DataFrame
        Columns: date, prob, actual, fold.
    out_path : Path
        Destination PNG file.
    """
    df = predictions.copy().sort_values("date")
    df["date"] = pd.to_datetime(df["date"])

    fig, ax = plt.subplots(figsize=(12, 4))

    # Shade stress periods
    in_stress = False
    stress_start = None
    for _, row in df.iterrows():
        if row["actual"] == 1 and not in_stress:
            in_stress = True
            stress_start = row["date"]
        elif row["actual"] == 0 and in_stress:
            ax.axvspan(stress_start, row["date"], alpha=0.18, color="red", lw=0)
            in_stress = False
    if in_stress:
        ax.axvspan(stress_start, df["date"].iloc[-1], alpha=0.18, color="red", lw=0)

    ax.plot(df["date"], df["prob"], lw=1.2, color="#2166ac", label="Stress probability")
    ax.axhline(0.5, color="grey", ls="--", lw=0.9, label="Decision threshold (0.5)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Probability")
    ax.set_title("Out-of-Sample Predicted Stress Probability (shaded = actual stress events)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _savefig(fig, out_path)
