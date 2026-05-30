"""Expanding-window walk-forward cross-validation with a purge gap.

The splitter never exposes future data to the training set. A configurable
purge gap sits between the last training date and the first test date to
prevent label leakage from overlapping forward-return windows.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from liquidity_radar.config import PURGE_DAYS, TEST_WINDOW_DAYS, TRAIN_MIN_YEARS


class WalkForwardCV:
    """Expanding-window walk-forward cross-validator.

    Parameters
    ----------
    train_min_years : int
        Minimum years of history before the first fold is evaluated.
    purge_days : int
        Calendar days between the last training date and the first test date.
        Prevents leakage from the 20-day forward-return label window.
    test_window_days : int
        Calendar days covered by each test window.
    step_freq : str
        Pandas offset alias for how far to step train_end between folds.
        Default "6ME" steps every six months at month-end boundaries.
    """

    def __init__(
        self,
        train_min_years: int = TRAIN_MIN_YEARS,
        purge_days: int = PURGE_DAYS,
        test_window_days: int = TEST_WINDOW_DAYS,
        step_freq: str = "6ME",
    ) -> None:
        self.train_min_years = train_min_years
        self.purge_days = purge_days
        self.test_window_days = test_window_days
        self.step_freq = step_freq

    def split(self, df: pd.DataFrame) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, int]]:
        """Yield (train_df, test_df, fold_index) triples.

        Parameters
        ----------
        df : DataFrame
            Must have a monotonically increasing ``DatetimeIndex`` named ``date``.

        Yields
        ------
        train : DataFrame
            All rows up to and including ``train_end``.
        test : DataFrame
            Rows in ``[test_start, test_end]``, after the purge gap.
        fold_idx : int
            Zero-based fold counter.
        """
        if df.index.name != "date":
            raise ValueError("df.index.name must be 'date'")

        first_date = df.index.min()
        last_date = df.index.max()
        first_train_end = first_date + pd.DateOffset(years=self.train_min_years)

        fold_ends = pd.date_range(first_train_end, last_date, freq=self.step_freq)
        fold_idx = 0

        for train_end in fold_ends:
            test_start = train_end + pd.Timedelta(days=self.purge_days)
            test_end = test_start + pd.Timedelta(days=self.test_window_days)

            if test_start > last_date:
                break

            train = df.loc[:train_end]
            test = df.loc[test_start:test_end]

            if len(train) < 10 or len(test) < 5:
                continue

            # Invariant: training set must end strictly before test set begins.
            assert train.index.max() < test.index.min(), (
                f"Fold {fold_idx}: leakage detected — "
                f"train ends {train.index.max().date()}, "
                f"test starts {test.index.min().date()}"
            )

            yield train, test, fold_idx
            fold_idx += 1
