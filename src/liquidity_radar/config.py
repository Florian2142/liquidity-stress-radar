"""Central configuration. Tickers, paths, parameters."""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
DB_PATH = DATA_DIR / "lsr.duckdb"

# ── Date range ───────────────────────────────────────────────────────────
START_DATE = "1995-01-01"  # earlier than VIX history; harmless to over-request

# ── Tickers ──────────────────────────────────────────────────────────────
SPY_TICKER = "SPY"

VOL_TICKERS = {
    "VIX": "^VIX",  # 30-day implied vol
    "VIX9D": "^VIX9D",  # 9-day implied vol
    "VIX3M": "^VIX3M",  # 3-month implied vol (formerly VXV)
}

# ── FRED series (no API key needed via pandas-datareader) ────────────────
FRED_SERIES = {
    "DGS10": "yield_10y",  # 10-year constant maturity Treasury
    "DGS2": "yield_2y",  # 2-year constant maturity Treasury
    "DFF": "fed_funds",  # Effective Federal Funds Rate
}

# ── Target variable ──────────────────────────────────────────────────────
TARGET_HORIZON_DAYS = 20  # forecast window
TARGET_DRAWDOWN_PCT = 0.05  # 5% drop within horizon ⇒ positive label

# ── Feature parameters ───────────────────────────────────────────────────
AMIHUD_WINDOW = 20  # rolling-window length for Amihud
REALIZED_VOL_WINDOW = 20

# ── Walk-forward CV ──────────────────────────────────────────────────────
TRAIN_MIN_YEARS = 5  # minimum training window
PURGE_DAYS = 180  # 6-month purge between train and test
TEST_WINDOW_DAYS = 180  # 6-month test windows

# ── Random seed ──────────────────────────────────────────────────────────
RANDOM_SEED = 42


def ensure_dirs() -> None:
    """Create runtime directories if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
