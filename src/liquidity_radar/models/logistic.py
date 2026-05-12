"""Logistic regression wrapper with per-fold StandardScaler to prevent leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from liquidity_radar.config import RANDOM_SEED

# Canonical ordered feature list — must match the DuckDB ``features`` table schema.
# amihud_zscore and amihud_5d_change replace the raw amihud level (see scripts/04_amihud_variants.py).
# Variant experiment result: zscore+5d_chg OOS AUC = 0.6753 vs. level 0.6659 (+0.0094).
FEATURE_COLS: list[str] = [
    "amihud_zscore",
    "amihud_5d_change",
    "cs_spread",
    "edge",
    "vix_5d_change",
    "vix_term_ratio",
    "yield_curve_slope",
    "spy_drawdown",
    "realized_vol_20d",
]


class LogisticModel:
    """Thin wrapper around sklearn LogisticRegression.

    A fresh ``StandardScaler`` is fit on each training set so that test-set
    standardisation uses only training statistics — no leakage.

    Parameters
    ----------
    C : float
        Inverse of L2 regularisation strength. Smaller C → stronger penalty.
    random_state : int
        Seed for the solver's randomness.
    max_iter : int
        Maximum iterations for the LBFGS solver.
    """

    def __init__(
        self,
        C: float = 1.0,
        random_state: int = RANDOM_SEED,
        max_iter: int = 1000,
    ) -> None:
        self.C = C
        self.random_state = random_state
        self.max_iter = max_iter
        self._scaler = StandardScaler()
        self._model = LogisticRegression(
            C=C,
            penalty="l2",
            solver="lbfgs",
            random_state=random_state,
            max_iter=max_iter,
        )
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticModel":
        """Fit scaler and model on training data.

        Parameters
        ----------
        X : DataFrame
            Feature matrix. Columns must be ``FEATURE_COLS`` (or a subset).
        y : Series
            Binary labels (0 / 1).
        """
        self.feature_names_ = list(X.columns)
        X_scaled = self._scaler.fit_transform(X.to_numpy(dtype=float))
        self._model.fit(X_scaled, y.to_numpy())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted probabilities for the positive class.

        Parameters
        ----------
        X : DataFrame
            Same columns as those seen at fit time.
        """
        X_scaled = self._scaler.transform(X.to_numpy(dtype=float))
        return self._model.predict_proba(X_scaled)[:, 1]

    @property
    def coef_(self) -> pd.Series:
        """Standardised coefficients as a named Series."""
        return pd.Series(self._model.coef_[0], index=self.feature_names_)
