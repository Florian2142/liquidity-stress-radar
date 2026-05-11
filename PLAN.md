# Project Plan — Liquidity Stress Radar

This document is the source of truth for what to build, in what order, and what "done" looks like for each step. Read this before writing code.

## Working principle

**MVP first, polish second, stretch last.** Build everything end-to-end ugly by the end of Week 2. Polish from there. No optional feature gets implemented before the MVP works end-to-end.

## Timeline

| Week | Dates | Phase | Output |
|---|---|---|---|
| W1 | May 20–26 | Phase 1 — Data pipeline | DuckDB populated, smoke test passes |
| W2 | May 27–Jun 2 | Phase 2 — MVP model | Logistic regression baseline, walk-forward CV |
| W3 | Jun 3–9 | Phase 3 — Eval + plots | ROC, PR, feature importance, lead-time |
| W4 | Jun 10–16 | Phase 4 — Dashboard | 3-panel Streamlit app + paper draft v1 |
| W5 | Jun 17–23 | Phase 5 — Stretch + writing | XGBoost comparison if time, paper draft v2 |
| W6 | Jun 24–30 | Phase 6 — Polish | Dashboard freeze, paper revisions, deck |
| W7 | Jul 1–5 | Phase 7 — Submit | Paper Jul 5, presentation Jul 9 |

---

## Phase 0 — Setup ✅ DONE

The skeleton in this repository.

**Acceptance:**
- [x] Directory layout exists
- [x] `pyproject.toml` and `requirements.txt` present
- [x] DuckDB schema defined in `src/liquidity_radar/data/schema.py`
- [x] Data ingestion functions in `src/liquidity_radar/data/ingest.py`
- [x] Amihud feature in `src/liquidity_radar/features/liquidity.py`
- [x] Target labeling in `src/liquidity_radar/features/target.py`
- [x] `scripts/01_initial_load.py` runnable

---

## Phase 1 — Data pipeline ⏳ NEXT

Validate that data flows end-to-end from APIs to DuckDB to a clean DataFrame.

**Tasks:**
1. Run `python scripts/01_initial_load.py` and confirm it succeeds.
2. Inspect the loaded DuckDB tables; verify date coverage matches expectations (SPY from 1993, VIX from 1990).
3. If data looks wrong (gaps, duplicates, timezone bugs), fix the ingestion functions before moving on.
4. Add a function in `src/liquidity_radar/data/store.py` named `get_features_panel()` that returns a single pandas DataFrame indexed by date with all base series joined. Forward-fill at most 1 day; do not interpolate.
5. Write a simple test in `tests/test_data.py`: assert no duplicate dates, assert SPY and VIX overlap, assert no NaN in critical columns after forward-fill.

**Acceptance criteria:**
- `python scripts/01_initial_load.py` exits 0 with summary stats printed.
- `pytest tests/` passes.
- The combined DataFrame from `get_features_panel()` has at least 7,500 rows (≥ 30 years).
- DuckDB file at `data/lsr.duckdb` is < 100MB.

**Estimated time:** 1–2 person-days.

**Common pitfalls:**
- yfinance occasionally returns an empty frame on first call (rate limit). Retry once with backoff.
- VIX9D / VIX3M tickers have changed historically. Use `^VIX9D` and `^VIX3M`. If those fail, fall back to `^VXST` and `^VXMT`.
- FRED via `pandas-datareader` doesn't need an API key for most series.

---

## Phase 2 — MVP model

Build the smallest possible working classifier.

**Tasks:**
1. Add Corwin–Schultz proxy to `src/liquidity_radar/features/liquidity.py`.
2. Add EDGE estimator using the `bidask` package: `pip install bidask` then `from bidask import edge`.
3. Add volatility features (5-day VIX change, VIX term structure ratio) in `src/liquidity_radar/features/volatility.py`.
4. Add macro features (yield curve slope, FFR change) in `src/liquidity_radar/features/macro.py`.
5. Add technical features (SPY drawdown from 1Y high, 20-day realized vol) in `src/liquidity_radar/features/technical.py`.
6. Create `src/liquidity_radar/models/walkforward.py` with an expanding-window walk-forward CV class. Use a 6-month purge between train and test.
7. Create `src/liquidity_radar/models/logistic.py` with a thin sklearn `LogisticRegression` wrapper. Use L2 regularisation. Standardise features within each fold (no leakage).
8. Create `src/liquidity_radar/models/baseline.py` with the naïve `VIX > 25` rule.
9. Add `scripts/02_train_logistic.py` that trains the model on the full panel and saves predictions to `data/predictions.parquet`.

**Acceptance criteria:**
- 8 features computed correctly (3 liquidity + 2 volatility + 1 macro + 2 technical).
- Walk-forward CV runs without leakage. Verify by checking that the training set always ends before the test set starts.
- Logistic regression coefficients printed with confidence intervals.
- Predictions DataFrame saved with columns: `date`, `prob`, `actual`, `fold`.
- ROC-AUC > 0.55 on out-of-sample data (sanity check; if below 0.5, something is wrong).

**Estimated time:** 3–4 person-days.

**Reference implementation hint for walk-forward CV:**
```python
# Simplified
for fold_end in pd.date_range(train_min, df.index.max(), freq='6ME'):
    train_end = fold_end
    test_start = train_end + pd.Timedelta(days=180)  # 6-month purge
    test_end = test_start + pd.Timedelta(days=180)
    train = df.loc[:train_end]
    test = df.loc[test_start:test_end]
    yield train, test
```

---

## Phase 3 — Evaluation + plots

**Tasks:**
1. Create `src/liquidity_radar/eval/metrics.py` with: ROC-AUC, PR-AUC, Brier score, lead-time computation.
2. Create `src/liquidity_radar/eval/plots.py` with: ROC curve overlay (3 lines: T0, T1 VIX-only, T1 full), PR curve overlay, feature-importance bar chart, time-series of predicted probability with drawdown shading.
3. Add `scripts/03_evaluate.py` that loads predictions and produces all four plots as PNG files in `figures/`.
4. Add bootstrapped confidence intervals on ROC-AUC (block bootstrap, 1000 reps, 252-day blocks).

**Acceptance criteria:**
- All 4 plots saved as 300-DPI PNG.
- Lead-time computation: for each historical drawdown, find the first day in the previous 30 where `prob > 0.5`; report mean and median over all drawdowns.
- Bootstrap CI for AUC reported.

**Estimated time:** 2 person-days.

---

## Phase 4 — Streamlit dashboard

**Tasks:**
1. Create `src/liquidity_radar/dashboard/app.py` with three panels:
   - Top: probability gauge (today's value) + 90-day sparkline + status pill (Calm / Watch / Elevated / Stress).
   - Bottom-left: feature contribution bar chart (logistic regression coefficient × today's standardised value).
   - Bottom-right: two KPI tiles (out-of-sample ROC-AUC, lead-time vs VIX-only).
2. Add a "Methods" tab with: brief description of features, model, validation; link to the GitHub repo and the paper.
3. Cache data loads with `@st.cache_data(ttl=3600)` to keep the app responsive.
4. Deploy to Streamlit Community Cloud. Note the URL.

**Acceptance criteria:**
- App runs locally with `streamlit run src/liquidity_radar/dashboard/app.py`.
- App loads in under 5 seconds on cold start.
- Public URL works on demo day.

**Estimated time:** 2–3 person-days.

---

## Phase 5 — Stretch goals (only if Phase 4 done by end of W4)

Pick at most TWO from this list:

- **XGBoost comparison + SHAP**: train an XGBoost classifier with the same walk-forward CV; compare PR-AUC; produce a SHAP summary plot. ~1 day.
- **Per-stock screener**: load top 100 S&P stocks, compute per-stock Amihud z-scores, add a sortable table panel to the dashboard. ~1 day.
- **Defensive rotation backtest**: simulate a long/cash strategy that goes to cash when `prob > 0.6`; report Sharpe, max drawdown vs buy-and-hold. ~1.5 days.
- **Robustness**: re-run with drawdown thresholds 3%, 7%, 10% and forecast horizons 5, 10, 40 days; show how AUC changes. ~0.5 days.

**Do not pick more than two.** Each one risks burning the polish/writing budget.

---

## Phase 6 — Paper

Structure (30 pages):
1. Introduction (3 pp) — practitioner motivation (Deutsche Börse framing), academic gap, contribution, structure.
2. Literature review (5 pp) — Amihud (2002), Brunnermeier–Pedersen (2009), Hameed–Kang–Viswanathan (2010), Corwin–Schultz (2012), Ardia et al. (2024) EDGE, Gu–Kelly–Xiu (2020).
3. Data (4 pp) — sources, target variable definition, descriptive statistics.
4. Methodology (5 pp) — features, model, walk-forward CV, evaluation metrics.
5. Results (7 pp) — main table, ROC/PR curves, feature importance, robustness.
6. Dashboard description (2 pp) — screenshots and design rationale.
7. Discussion + Conclusion (3 pp) — limitations, extensions, takeaway.
8. References (1 pp).

---

## Phase 7 — Presentation

15–20 minute slot. Use the existing roadmap deck as a template; slides 1–13 of `Liquidity_Stress_Radar_MVP.pptx` map directly. Replace stylized stats with real numbers. Demo the deployed Streamlit app live.

---

## Cuts (non-negotiable)

These will not be built in this project. Do not let scope creep bring them back.

- World map of global stress (1 week of choropleth work, no research gain).
- Liquidity-aware portfolio optimizer (different research question, doubles scope).
- Granger causality tests (AUC comparison answers the same question more cleanly).
- Tick-level / TAQ data (cleaning cost dominates timeline).
- Meta-stacking / kernel methods (marginal AUC gain rarely > 0.02).

---

## Deliverables checklist

- [ ] `data/lsr.duckdb` populated with 30+ years of base series
- [ ] 8 features computed and stored
- [ ] Walk-forward CV validated (no leakage)
- [ ] Logistic regression trained, coefficients reported with CIs
- [ ] ROC, PR, feature-importance plots saved as PNG
- [ ] Streamlit dashboard deployed with public URL
- [ ] 30-page paper submitted by July 5
- [ ] Presentation deck submitted by July 9
- [ ] GitHub repo public with reproducible pipeline
