"""Evaluation metrics: ROC-AUC, PR-AUC, Brier score, lead-time, bootstrap CI."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def compute_roc_auc(actual: np.ndarray, prob: np.ndarray) -> float:
    """Return ROC-AUC for binary labels and predicted probabilities."""
    return float(roc_auc_score(actual, prob))


def compute_pr_auc(actual: np.ndarray, prob: np.ndarray) -> float:
    """Return area under the precision-recall curve (average precision)."""
    return float(average_precision_score(actual, prob))


def compute_brier_score(actual: np.ndarray, prob: np.ndarray) -> float:
    """Return Brier score (lower is better; 0 = perfect)."""
    return float(brier_score_loss(actual, prob))


def bootstrap_roc_auc(
    actual: np.ndarray,
    prob: np.ndarray,
    n_reps: int = 1000,
    block_size: int = 252,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Block-bootstrap 95% CI for ROC-AUC.

    Uses non-overlapping blocks of ``block_size`` consecutive observations to
    preserve time-series autocorrelation. Returns (point_estimate, ci_lo, ci_hi).
    """
    rng = np.random.default_rng(seed)
    n = len(actual)
    n_blocks = max(1, n // block_size)
    block_starts = np.arange(0, n - block_size + 1, block_size)

    point = float(roc_auc_score(actual, prob))
    boot_aucs: list[float] = []

    for _ in range(n_reps):
        chosen = rng.choice(block_starts, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, min(s + block_size, n)) for s in chosen])
        idx = idx[:n]
        a_b, p_b = actual[idx], prob[idx]
        if a_b.sum() == 0 or a_b.sum() == len(a_b):
            continue
        boot_aucs.append(float(roc_auc_score(a_b, p_b)))

    arr = np.array(boot_aucs)
    return point, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def compute_lead_time(
    predictions: pd.DataFrame,
    lookback: int = 30,
    threshold: float = 0.5,
) -> dict[str, float]:
    """For each stress onset, find how many days early the model signalled.

    A "stress onset" is the first day ``actual == 1`` in a consecutive run.
    We look back ``lookback`` days from that onset and record the first day
    where ``prob > threshold``. Lead time = days before onset.

    Parameters
    ----------
    predictions : DataFrame
        Must have columns ``date``, ``prob``, ``actual`` (sorted by date).
    lookback : int
        Days before onset to search for a signal.
    threshold : float
        Probability threshold for a signal.

    Returns
    -------
    dict with keys ``mean``, ``median``, ``n_events``.
    """
    df = predictions.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    dates = df["date"].to_numpy()
    probs = df["prob"].to_numpy()
    actuals = df["actual"].to_numpy()

    # Find stress onset days (first day of each contiguous block of label=1)
    onsets: list[int] = []
    for i in range(len(actuals)):
        if actuals[i] == 1 and (i == 0 or actuals[i - 1] == 0):
            onsets.append(i)

    lead_times: list[float] = []
    for onset_idx in onsets:
        onset_date = dates[onset_idx]
        # Search in the lookback window before onset
        window_mask = (dates >= onset_date - np.timedelta64(lookback, "D")) & (dates < onset_date)
        window_indices = np.where(window_mask)[0]
        signal_indices = window_indices[probs[window_indices] > threshold]
        if len(signal_indices) > 0:
            first_signal_date = dates[signal_indices[0]]
            days_early = (onset_date - first_signal_date) / np.timedelta64(1, "D")
            lead_times.append(float(days_early))

    if not lead_times:
        return {"mean": float("nan"), "median": float("nan"), "n_events": len(onsets)}

    return {
        "mean": float(np.mean(lead_times)),
        "median": float(np.median(lead_times)),
        "n_events": len(onsets),
        "n_signalled": len(lead_times),
    }
