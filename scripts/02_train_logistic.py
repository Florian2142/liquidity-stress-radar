"""Phase 2 — train logistic regression with walk-forward CV.

Run::

    python scripts/02_train_logistic.py

Outputs:
- Prints fold boundaries (verifying no leakage).
- Prints logistic regression coefficients with 95% CIs (±1.96 SD across folds).
- Prints out-of-sample ROC-AUC.
- Saves ``data/predictions.parquet`` with columns: date, prob, actual, fold.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from liquidity_radar.config import DATA_DIR, ensure_dirs  # noqa: E402
from liquidity_radar.data.store import (  # noqa: E402
    get_connection,
    get_features_panel,
    upsert_dataframe,
)
from liquidity_radar.features.build import build_feature_matrix  # noqa: E402
from liquidity_radar.features.target import forward_drawdown_label  # noqa: E402
from liquidity_radar.models.baseline import vix_threshold_predict  # noqa: E402
from liquidity_radar.models.logistic import FEATURE_COLS, LogisticModel  # noqa: E402
from liquidity_radar.models.walkforward import WalkForwardCV  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("02_train_logistic")


def main() -> int:
    ensure_dirs()

    # ── 1. Load raw panel and compute features ────────────────────────────
    print("\n-- Loading raw panel from DuckDB")
    with get_connection() as con:
        panel = get_features_panel(con)
    print(
        f"   Panel: {len(panel):,} rows x {panel.shape[1]} cols "
        f"({panel.index.min().date()} to {panel.index.max().date()})"
    )

    print("\n-- Computing 9 features (amihud_zscore + amihud_5d_change replace raw amihud level)")
    features_full = build_feature_matrix(panel, include_raw_amihud=True)
    features = features_full[FEATURE_COLS]
    targets = forward_drawdown_label(panel)
    print(f"   Features: {list(features.columns)}")

    # ── 2. Persist updated features to DuckDB + parquet snapshot ─────────
    with get_connection() as con:
        upsert_dataframe(con, features_full, "features")
        upsert_dataframe(con, targets, "targets")
    print("   Persisted features + targets to DuckDB.")

    snapshot_path = DATA_DIR / "panel_snapshot.parquet"
    panel.to_parquet(snapshot_path)
    print(f"   Panel snapshot saved: {snapshot_path}  ({len(panel):,} rows)")

    # ── 3. Build combined DataFrame for walk-forward CV ───────────────────
    combined = features.join(targets[["label"]]).dropna()
    print(
        f"\n-- Combined (no NaN): {len(combined):,} rows, "
        f"positive-label rate: {combined['label'].mean():.2%}"
    )

    # ── 4. Walk-forward CV ────────────────────────────────────────────────
    cv = WalkForwardCV()
    all_preds: list[pd.DataFrame] = []
    fold_coefs: list[pd.Series] = []

    print("\n-- Walk-forward folds (train ends strictly before test starts)")
    print(
        f"   {'Fold':>4}  {'Train start':>12} {'Train end':>12} "
        f"  {'Test start':>12} {'Test end':>12}  {'Gap(d)':>7}  {'N_train':>8}  {'N_test':>7}"
    )

    for train_df, test_df, fold_idx in cv.split(combined):
        X_train = train_df[FEATURE_COLS]
        y_train = train_df["label"]
        X_test = test_df[FEATURE_COLS]
        y_test = test_df["label"]

        # Fit model for this fold
        model = LogisticModel()
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)

        fold_coefs.append(model.coef_)

        # Verify no leakage
        train_end = train_df.index.max()
        test_start = test_df.index.min()
        gap_days = (test_start - train_end).days
        assert train_end < test_start, (
            f"Fold {fold_idx}: leakage! train_end={train_end}, test_start={test_start}"
        )

        print(
            f"   {fold_idx:>4}  {train_df.index.min().date()!s:>12} {train_end.date()!s:>12}"
            f"  {test_start.date()!s:>12} {test_df.index.max().date()!s:>12}"
            f"  {gap_days:>7}  {len(train_df):>8}  {len(test_df):>7}"
        )

        preds = pd.DataFrame(
            {
                "date": test_df.index,
                "prob": probs,
                "actual": y_test.to_numpy(dtype=float),
                "fold": fold_idx,
            }
        )
        all_preds.append(preds)

    if not all_preds:
        logger.error("No folds produced — check data range and TRAIN_MIN_YEARS.")
        return 1

    # ── 5. Aggregate results ──────────────────────────────────────────────
    predictions_df = pd.concat(all_preds, ignore_index=True)

    oos_auc = roc_auc_score(predictions_df["actual"], predictions_df["prob"])
    print(f"\n-- Out-of-sample ROC-AUC: {oos_auc:.4f}  (must be > 0.55)")

    # Baseline: VIX > 25
    baseline_pred = vix_threshold_predict(panel).reindex(predictions_df["date"].values)
    baseline_pred_aligned = baseline_pred.to_numpy(dtype=float)
    actual_aligned = predictions_df["actual"].to_numpy(dtype=float)
    valid_mask = ~np.isnan(baseline_pred_aligned) & ~np.isnan(actual_aligned)
    if valid_mask.sum() > 0:
        baseline_auc = roc_auc_score(
            actual_aligned[valid_mask],
            baseline_pred_aligned[valid_mask],
        )
        print(f"   VIX>25 baseline ROC-AUC:  {baseline_auc:.4f}")

    # ── 6. Coefficients with 95% CI across folds ─────────────────────────
    coef_df = pd.DataFrame(fold_coefs)
    print(
        f"\n-- Logistic Regression Coefficients (mean +/- 1.96 SD across {len(fold_coefs)} folds)"
    )
    print(f"   {'Feature':<25}  {'Mean':>8}  {'95% CI lower':>14}  {'95% CI upper':>14}")
    for col in FEATURE_COLS:
        mean = coef_df[col].mean()
        sd = coef_df[col].std()
        ci_lo = mean - 1.96 * sd
        ci_hi = mean + 1.96 * sd
        print(f"   {col:<25}  {mean:+8.4f}  {ci_lo:+14.4f}  {ci_hi:+14.4f}")

    # ── 7. Save predictions ───────────────────────────────────────────────
    out_path = DATA_DIR / "predictions.parquet"
    predictions_df.to_parquet(out_path, index=False)
    print(f"\n-- Predictions saved: {out_path}")
    print(f"   Rows: {len(predictions_df):,}  |  Folds: {predictions_df['fold'].nunique()}")
    print(f"   Columns: {list(predictions_df.columns)}")

    # ── 8. Save fold coefs for feature-importance plot ────────────────────
    coef_df.to_csv(DATA_DIR / "fold_coefs.csv", index=False)
    print(f"   Fold coefs saved: {DATA_DIR / 'fold_coefs.csv'}")

    # ── 9. Fit final model on ALL data and save weights for dashboard ──────
    print("\n-- Fitting final model on full combined dataset")
    final_model = LogisticModel()
    final_model.fit(combined[FEATURE_COLS], combined["label"])
    np.savez(
        DATA_DIR / "model_params.npz",
        coef=final_model._model.coef_[0],
        intercept=final_model._model.intercept_,
        scaler_mean=final_model._scaler.mean_,
        scaler_scale=final_model._scaler.scale_,
        feature_names=np.array(FEATURE_COLS),
        oos_auc=np.array([oos_auc]),
    )
    print(f"   Model params saved: {DATA_DIR / 'model_params.npz'}")

    if oos_auc <= 0.55:
        print("\nWARNING: ROC-AUC <= 0.55. Check feature computation and label alignment.")
        return 1

    print("\n[OK] Phase 2 complete. Next: run scripts/03_evaluate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
