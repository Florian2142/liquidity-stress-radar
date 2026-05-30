"""Reusable walk-forward backtest.

A single function runs the expanding-window walk-forward cross-validation used
everywhere in the project (training, ablation, sensitivity analysis). Keeping it
in one place guarantees that every reported number comes from the same
leakage-controlled procedure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from liquidity_radar.models.logistic import LogisticModel
from liquidity_radar.models.walkforward import WalkForwardCV


def walk_forward_predict(
    combined: pd.DataFrame,
    feature_cols: list[str],
    cv: WalkForwardCV | None = None,
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward CV and return out-of-sample predictions and fold coefs.

    Parameters
    ----------
    combined : DataFrame
        DatetimeIndex named ``date``. Must contain ``feature_cols`` and
        ``label_col`` with no NaNs.
    feature_cols : list of str
        Columns to use as model inputs for this run.
    cv : WalkForwardCV, optional
        Splitter to use. Defaults to a fresh :class:`WalkForwardCV` with the
        project's standard parameters.
    label_col : str
        Name of the binary target column.

    Returns
    -------
    predictions : DataFrame
        Columns ``date``, ``prob``, ``actual``, ``fold`` — one row per
        out-of-sample observation, concatenated across folds.
    fold_coefs : DataFrame
        Rows = folds, columns = ``feature_cols``. Standardised coefficients.
    """
    cv = cv or WalkForwardCV()
    all_preds: list[pd.DataFrame] = []
    fold_coefs: list[pd.Series] = []

    for train_df, test_df, fold_idx in cv.split(combined):
        model = LogisticModel()
        model.fit(train_df[feature_cols], train_df[label_col])
        probs = model.predict_proba(test_df[feature_cols])
        fold_coefs.append(model.coef_)

        # Invariant: train must end strictly before test begins.
        assert train_df.index.max() < test_df.index.min(), f"fold {fold_idx}: leakage"

        all_preds.append(
            pd.DataFrame(
                {
                    "date": test_df.index,
                    "prob": probs,
                    "actual": test_df[label_col].to_numpy(dtype=float),
                    "fold": fold_idx,
                }
            )
        )

    if not all_preds:
        return (
            pd.DataFrame(columns=["date", "prob", "actual", "fold"]),
            pd.DataFrame(columns=feature_cols),
        )

    predictions = pd.concat(all_preds, ignore_index=True)
    coef_df = pd.DataFrame(fold_coefs).reset_index(drop=True)
    return predictions, coef_df


def baseline_score_predictions(
    combined: pd.DataFrame,
    score: pd.Series,
    label_col: str = "label",
) -> pd.DataFrame:
    """Align a raw score (e.g. the VIX level) to the OOS evaluation window.

    The baseline needs no training: the score itself ranks observations. We
    restrict it to the same dates the model is evaluated on so the comparison
    is apples-to-apples.
    """
    aligned = score.reindex(combined.index).to_numpy(dtype=float)
    out = pd.DataFrame(
        {
            "date": combined.index,
            "prob": aligned,
            "actual": combined[label_col].to_numpy(dtype=float),
        }
    )
    return out.dropna(subset=["prob"])


def normalise_scores(values: np.ndarray) -> np.ndarray:
    """Min-max scale a score vector to [0, 1] for Brier/calibration comparability."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)
