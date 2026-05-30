"""Data-quality diagnostics for the market panel.

These checks are deliberately non-fatal: they summarise the state of the data
so the dashboard and the load script can surface coverage, gaps, and freshness
to the user rather than failing silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class QualityReport:
    """Structured summary of panel data quality."""

    n_rows: int
    start_date: pd.Timestamp | None
    end_date: pd.Timestamp | None
    n_duplicate_dates: int
    n_calendar_gaps: int
    max_gap_days: int
    missing_by_column: dict[str, int] = field(default_factory=dict)
    coverage_by_column: dict[str, float] = field(default_factory=dict)
    stale_days: int = 0

    @property
    def is_fresh(self) -> bool:
        """True if the most recent observation is within the last 5 calendar days."""
        return self.stale_days <= 5

    def summary_frame(self) -> pd.DataFrame:
        """Per-column missing / coverage table for display."""
        return pd.DataFrame(
            {
                "missing": pd.Series(self.missing_by_column),
                "coverage_pct": pd.Series(self.coverage_by_column).mul(100).round(1),
            }
        )


def build_quality_report(panel: pd.DataFrame, asof: pd.Timestamp | None = None) -> QualityReport:
    """Compute a :class:`QualityReport` for a market panel.

    Parameters
    ----------
    panel : DataFrame
        DatetimeIndex named ``date``. Any set of columns.
    asof : Timestamp, optional
        Reference "today" for the staleness check. Defaults to ``pd.Timestamp.now()``.

    Returns
    -------
    QualityReport
    """
    if asof is None:
        asof = pd.Timestamp.now().normalize()

    idx = pd.DatetimeIndex(panel.index)
    n_rows = len(panel)
    start = idx.min() if n_rows else None
    end = idx.max() if n_rows else None

    n_dupes = int(idx.duplicated().sum())

    # Trading-day gaps: business days between consecutive observations beyond 1.
    n_gaps = 0
    max_gap = 0
    if n_rows > 1:
        sorted_idx = idx.sort_values()
        for prev, nxt in zip(sorted_idx[:-1], sorted_idx[1:], strict=False):
            bdays = len(pd.bdate_range(prev, nxt)) - 1  # exclude the start day
            if bdays > 1:
                n_gaps += 1
                max_gap = max(max_gap, bdays)

    missing = {col: int(panel[col].isna().sum()) for col in panel.columns}
    coverage = {col: float(panel[col].notna().mean()) for col in panel.columns}

    stale_days = int((asof - end).days) if end is not None else 10**6

    return QualityReport(
        n_rows=n_rows,
        start_date=start,
        end_date=end,
        n_duplicate_dates=n_dupes,
        n_calendar_gaps=n_gaps,
        max_gap_days=max_gap,
        missing_by_column=missing,
        coverage_by_column=coverage,
        stale_days=stale_days,
    )
