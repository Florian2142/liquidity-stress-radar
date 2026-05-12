# Liquidity Stress Radar

A binary classifier that predicts S&P 500 drawdowns of 5% or more in the next 20 trading days, using market microstructure liquidity signals (Amihud illiquidity, Corwin–Schultz spread proxy, EDGE estimator) alongside volatility and macro features (VIX, VIX term structure, yield curve).


## Research question

> Do liquidity-based features improve prediction of S&P 500 drawdowns ≥ 5% in the next 20 days, beyond a VIX-only baseline?

Falsifiable: a positive result reports the AUC and lead-time gain; a negative result reports that VIX is hard to beat (also publishable).

## Architecture

```
yfinance + FRED + CBOE  →  DuckDB  →  Features  →  Logistic Regression  →  Streamlit Dashboard
                                       |              |
                                       Amihud         Walk-forward CV
                                       Corwin-Schultz ROC, PR, lead-time
                                       EDGE
                                       VIX features
                                       Yield curve
                                       Realized vol
```

Three-layer system. Each layer independently testable. Total target: ~800 lines of Python.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Validate the pipeline on synthetic data (no network needed, ~5s)
python scripts/00_offline_smoke.py

# 3. Pull real data from yfinance + FRED (~2 min, requires internet)
python scripts/01_initial_load.py

# 4. Run the test suite
pytest tests/ -v

# 5. Run the dashboard (after Phase 6)
streamlit run src/liquidity_radar/dashboard/app.py
```

## Repository layout

```
liquidity-stress-radar/
├── README.md            # This file
├── PLAN.md              # Phased roadmap — read this next
├── CLAUDE.md            # Conventions for Claude Code
├── pyproject.toml       # Modern Python packaging
├── requirements.txt     # Pinned dependencies
├── .env.example         # FRED API key template (optional)
├── data/                # DuckDB file lives here (gitignored)
├── src/liquidity_radar/
│   ├── config.py        # Tickers, paths, parameters
│   ├── data/            # Ingestion + DuckDB schema
│   ├── features/        # Liquidity, volatility, macro, target
│   ├── models/          # Logistic baseline, walk-forward CV
│   ├── eval/            # Metrics + plots
│   └── dashboard/       # Streamlit app
├── scripts/             # Run-from-terminal entry points
├── tests/               # Pytest smoke tests
└── notebooks/           # Exploratory analysis
```

## What's already built (Phase 0)

- Project skeleton, `pyproject.toml`, `requirements.txt`
- DuckDB schema (`src/liquidity_radar/data/schema.py`)
- Data ingestion for SPY, VIX, VIX9D, VIX3M, 10y/2y yields (`src/liquidity_radar/data/ingest.py`)
- Amihud illiquidity feature (`src/liquidity_radar/features/liquidity.py`)
- Drawdown target labeling (`src/liquidity_radar/features/target.py`)
- Initial-load smoke-test script (`scripts/01_initial_load.py`)

## What's next

See `PLAN.md` for the phased roadmap. Phase 1 is the next thing to do: run the smoke test, validate data, then add Corwin–Schultz and EDGE proxies.

## Data sources

| Source | What | Cost |
|---|---|---|
| yfinance | SPY OHLCV, VIX, VIX9D, VIX3M | Free |
| pandas-datareader → FRED | DGS10, DGS2, DFF | Free |
| Local Parquet (optional) | Existing Nasdaq core US equities tape | Already on disk |

The pipeline is yfinance-only by default. To plug in local Nasdaq data instead, swap `fetch_spy()` in `src/liquidity_radar/data/ingest.py` with a function that reads your Parquet files. The DuckDB schema is identical.

## Conventions

See `CLAUDE.md`. Short version: Python 3.11+, type hints, pandas (not polars), pathlib (not os.path), pytest for tests, ruff for formatting.

## License

MIT. This is academic work.
