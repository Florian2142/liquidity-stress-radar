# Claude Code Conventions

This file is read automatically by Claude Code when working in this repository. Follow these conventions when adding or modifying code.

## Project orientation

- **Read `README.md` first** before making any changes — it describes the finished system, its methodology, and how to run it.
- The deliverable is a finished research product: code, the robustness battery, figures, and the Streamlit dashboard. Keep it that way.

## Python style

- **Python 3.11+**. Use modern syntax: `match` statements, `|` for unions, `list[int]` not `List[int]`.
- **Type hints everywhere**. Functions without return-type hints will be rejected.
- **Pathlib, not `os.path`.** `from pathlib import Path` and use `Path(__file__).parent`.
- **Pandas, not Polars.** The seminar teaches pandas; do not introduce Polars.
- **No global state.** Functions take inputs and return outputs. The single exception is the `data/lsr.duckdb` file.
- **`httpx`, not `requests`.** Already in dependencies.

## Formatting and linting

- Format with `ruff format`.
- Lint with `ruff check`.
- Line length: 100.
- Imports sorted by ruff (stdlib → third-party → local).

## Testing

- Tests live in `tests/` and use pytest.
- Smoke tests are mandatory after Phase 1. Every module gets at least one happy-path test.
- Do not write tests for plotting code — visual QA only.
- Run tests: `pytest tests/ -v`.

## Data handling

- All durable state goes in DuckDB at `data/lsr.duckdb`.
- Never commit `data/` content. The directory has a `.gitkeep` placeholder.
- For pandas DataFrames in memory, always use a `DatetimeIndex` named `date`.
- Treat all dates as US trading-calendar business days. Use `pandas.tseries.offsets.BDay` not raw days.
- Forward-fill missing market data at most 1 trading day. Never interpolate.
- Never look ahead. Walk-forward validation is sacred. If you write code that uses `df.shift(-N)` outside of target-variable construction, justify it in a comment.

## Dependencies

- Adding a new dependency requires editing `requirements.txt` AND `pyproject.toml`.
- Pin to a minor version (`pandas>=2.2,<2.3`).
- **Ask before adding any new dependency.** The seminar values minimal stacks.

## Git hygiene

- Commits are small and atomic. One concept per commit.
- Commit messages in imperative mood: "add Corwin-Schultz proxy" not "added Corwin-Schultz proxy".
- Never commit secrets. `.env` is gitignored.
- Never `git push --force` to a shared branch.

## Anti-patterns to avoid

- Do not generate Markdown reports as deliverables — the deliverable is code, plots, and a paper.
- Do not over-engineer. If a function is 5 lines, do not turn it into a class.
- Do not introduce abstract base classes for "future flexibility." YAGNI applies hard.
- Do not silently swallow exceptions. Log and re-raise, or fix the underlying issue.
- Do not commit Jupyter notebook outputs. Clear outputs before committing.

## When uncertain

- Re-read `README.md` — the methodology and scope usually have the answer.
- If a change would expand scope beyond drawdown-risk monitoring, ask first.

## Performance expectations

- Initial data load: < 3 minutes.
- Full feature recomputation: < 30 seconds.
- Logistic regression training (full panel): < 5 seconds.
- Walk-forward CV with 50 folds: < 2 minutes.
- Streamlit dashboard cold start: < 5 seconds.

If you hit any of these limits, profile before optimising.
