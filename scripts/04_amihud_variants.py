"""Experiment: Amihud feature variant comparison.

Tests four feature sets that differ only in how Amihud illiquidity is represented:

    v0  amihud_level    — raw 20-day rolling mean (current baseline)
    v1  amihud_5d_chg   — 5-day change in the rolling mean
    v2  amihud_zscore   — rolling z-score vs. 252-day trailing window
    v3  zscore+5d_chg   — both z-score AND 5d-change (9 features total)

All other 7 features are held fixed. Walk-forward CV is identical to the main
training script (32 folds, 6-month purge, expanding window).

Run::

    python scripts/04_amihud_variants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from liquidity_radar.data.store import get_connection, get_features_panel  # noqa: E402
from liquidity_radar.features.liquidity import (  # noqa: E402
    amihud_5d_change,
    amihud_illiquidity,
    amihud_zscore,
    corwin_schultz_spread,
    edge_spread,
)
from liquidity_radar.features.macro import yield_curve_slope  # noqa: E402
from liquidity_radar.features.target import forward_drawdown_label  # noqa: E402
from liquidity_radar.features.technical import realized_vol_20d, spy_drawdown_from_high  # noqa: E402
from liquidity_radar.features.volatility import vix_5d_change, vix_term_ratio  # noqa: E402
from liquidity_radar.models.logistic import LogisticModel  # noqa: E402
from liquidity_radar.models.walkforward import WalkForwardCV  # noqa: E402

# ── The 7 features that stay fixed across all variants ───────────────────
FIXED_COLS = [
    "cs_spread",
    "edge",
    "vix_5d_change",
    "vix_term_ratio",
    "yield_curve_slope",
    "spy_drawdown",
    "realized_vol_20d",
]

VARIANTS: dict[str, list[str]] = {
    "v0  amihud_level ":  ["amihud"]              + FIXED_COLS,  # baseline
    "v1  amihud_5d_chg":  ["amihud_5d_change"]    + FIXED_COLS,
    "v2  amihud_zscore":  ["amihud_zscore"]        + FIXED_COLS,
    "v3  zscore+5d_chg":  ["amihud_zscore", "amihud_5d_change"] + FIXED_COLS,
}


def _build_all_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute every feature needed across all variants in one pass."""
    feat = pd.DataFrame(index=panel.index)
    # Amihud variants
    feat["amihud"]           = amihud_illiquidity(panel)
    feat["amihud_zscore"]    = amihud_zscore(panel)
    feat["amihud_5d_change"] = amihud_5d_change(panel)
    # Fixed features
    feat["cs_spread"]        = corwin_schultz_spread(panel)
    feat["edge"]             = edge_spread(panel)
    feat["vix_5d_change"]    = vix_5d_change(panel)
    feat["vix_term_ratio"]   = vix_term_ratio(panel)
    feat["yield_curve_slope"]= yield_curve_slope(panel)
    feat["spy_drawdown"]     = spy_drawdown_from_high(panel)
    feat["realized_vol_20d"] = realized_vol_20d(panel)
    feat.index.name = "date"
    return feat


def _run_variant(combined: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Walk-forward CV for one feature set. Returns AUC + per-fold coefficients."""
    cv = WalkForwardCV()
    all_preds: list[pd.DataFrame] = []
    fold_coefs: list[pd.Series] = []

    for train_df, test_df, _ in cv.split(combined):
        X_train = train_df[feature_cols]
        y_train = train_df["label"]
        X_test  = test_df[feature_cols]
        y_test  = test_df["label"]

        model = LogisticModel()
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)
        fold_coefs.append(model.coef_)

        all_preds.append(
            pd.DataFrame({"prob": probs, "actual": y_test.to_numpy(dtype=float)})
        )

    predictions = pd.concat(all_preds, ignore_index=True)
    auc = roc_auc_score(predictions["actual"], predictions["prob"])
    coef_df = pd.DataFrame(fold_coefs)

    return {
        "auc": auc,
        "n_obs": len(predictions),
        "n_folds": len(fold_coefs),
        "coef_mean": coef_df.mean(),
        "coef_sd": coef_df.std(),
    }


def main() -> int:
    print("\n=== Amihud variant experiment ===\n")

    print("Loading panel from DuckDB…")
    with get_connection() as con:
        panel = get_features_panel(con)
    print(f"  Panel: {len(panel):,} rows  ({panel.index.min().date()} to {panel.index.max().date()})")

    print("\nComputing all feature variants…")
    all_feat = _build_all_features(panel)
    targets   = forward_drawdown_label(panel)

    results: dict[str, dict] = {}
    for name, cols in VARIANTS.items():
        feat_subset = all_feat[cols]
        combined = feat_subset.join(targets[["label"]]).dropna()
        n = len(combined)
        pos = combined["label"].mean()
        print(f"\n  {name}  ({len(cols)} features, {n:,} obs, {pos:.1%} positive)")
        results[name] = _run_variant(combined, cols)
        print(f"    OOS AUC = {results[name]['auc']:.4f}")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    baseline_auc = results["v0  amihud_level "]["auc"]
    print(f"\n  {'Variant':<22}  {'OOS AUC':>8}  {'vs. baseline':>12}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*12}")
    for name, r in results.items():
        delta = r["auc"] - baseline_auc
        marker = "  ← WINNER" if r["auc"] == max(v["auc"] for v in results.values()) else ""
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<22}  {r['auc']:>8.4f}  {sign}{delta:>+11.4f}{marker}")

    # ── Coefficient breakdown for each variant ────────────────────────────
    print("\n\n-- Per-variant mean coefficients (standardised)\n")
    for name, r in results.items():
        print(f"  {name}")
        for feat, mean in r["coef_mean"].items():
            sd = r["coef_sd"][feat]
            print(f"    {feat:<25}  {mean:+.4f}  (±{sd:.4f})")
        print()

    winner = max(results, key=lambda k: results[k]["auc"])
    print("=" * 60)
    print(f"  Recommended feature set: {winner.strip()}")
    print(f"  OOS AUC:  {results[winner]['auc']:.4f}  "
          f"(+{results[winner]['auc'] - baseline_auc:+.4f} vs. amihud_level)")
    print("=" * 60)
    print("\nNext: update FEATURE_COLS in models/logistic.py and re-run 02_train_logistic.py\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
