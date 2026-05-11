"""Phase 3 — produce evaluation metrics and all four publication plots.

Run::

    python scripts/03_evaluate.py

Prerequisites: ``02_train_logistic.py`` must have run first so that
``data/predictions.parquet``, ``data/fold_coefs.csv``, and
``data/model_params.npz`` exist.

Outputs (all written to ``figures/``):
    roc_curves.png          — ROC overlay: logistic vs VIX-score vs random
    pr_curves.png           — PR overlay: logistic vs VIX-score vs no-skill
    feature_importance.png  — Bar chart: mean coef ± 1.96 SD across folds
    probability_ts.png      — Predicted probability time-series + shading
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from liquidity_radar.config import DATA_DIR, FIGURES_DIR, ensure_dirs  # noqa: E402
from liquidity_radar.data.store import get_connection, get_features_panel  # noqa: E402
from liquidity_radar.eval.metrics import (  # noqa: E402
    bootstrap_roc_auc,
    compute_brier_score,
    compute_lead_time,
    compute_pr_auc,
    compute_roc_auc,
)
from liquidity_radar.eval.plots import (  # noqa: E402
    plot_feature_importance,
    plot_pr_curves,
    plot_probability_timeseries,
    plot_roc_curves,
)


def main() -> int:
    ensure_dirs()

    # ── Load artefacts ────────────────────────────────────────────────────
    pred_path = DATA_DIR / "predictions.parquet"
    coef_path = DATA_DIR / "fold_coefs.csv"
    if not pred_path.exists():
        print("ERROR: predictions.parquet not found. Run 02_train_logistic.py first.")
        return 1
    if not coef_path.exists():
        print("ERROR: fold_coefs.csv not found. Run 02_train_logistic.py first.")
        return 1

    predictions = pd.read_parquet(pred_path)
    fold_coefs = pd.read_csv(coef_path)
    print(f"Loaded {len(predictions):,} predictions across {predictions['fold'].nunique()} folds.")

    with get_connection() as con:
        panel = get_features_panel(con)
    print(f"Loaded panel: {len(panel):,} rows.")

    actual = predictions["actual"].to_numpy()
    prob = predictions["prob"].to_numpy()

    # ── Metrics ───────────────────────────────────────────────────────────
    roc_auc = compute_roc_auc(actual, prob)
    pr_auc = compute_pr_auc(actual, prob)
    brier = compute_brier_score(actual, prob)
    auc_point, ci_lo, ci_hi = bootstrap_roc_auc(actual, prob, n_reps=1000, block_size=252)
    lead = compute_lead_time(predictions, lookback=30, threshold=0.5)

    print("\n=== Evaluation Metrics ===")
    print(f"  ROC-AUC:           {roc_auc:.4f}  (bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])")
    print(f"  PR-AUC:            {pr_auc:.4f}")
    print(f"  Brier score:       {brier:.4f}")
    print(f"  Lead-time (mean):  {lead['mean']:.1f} days  (median: {lead['median']:.1f} days)")
    print(f"  Events detected:   {lead.get('n_signalled', 0)} / {lead['n_events']} stress onsets")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n=== Generating plots ===")

    plot_roc_curves(predictions, panel, FIGURES_DIR / "roc_curves.png")
    plot_pr_curves(predictions, panel, FIGURES_DIR / "pr_curves.png")
    plot_feature_importance(fold_coefs, FIGURES_DIR / "feature_importance.png")
    plot_probability_timeseries(predictions, FIGURES_DIR / "probability_ts.png")

    print("\n[OK] All 4 plots saved to figures/ at 300 DPI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
