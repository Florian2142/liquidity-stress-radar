"""Compute the full statistical-robustness battery and persist the artefacts.

Run::

    python scripts/05_robustness.py

Reads the market panel from DuckDB, rebuilds features, and writes a set of
CSV/JSON artefacts to ``data/robustness/`` that the dashboard's
"Robustness Tests" section loads directly. Every number is computed here under
walk-forward cross-validation — none are hard-coded anywhere in the app.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from liquidity_radar.config import DATA_DIR, ensure_dirs  # noqa: E402
from liquidity_radar.data.quality import build_quality_report  # noqa: E402
from liquidity_radar.data.store import get_connection, get_features_panel  # noqa: E402
from liquidity_radar.eval.backtest import (  # noqa: E402
    baseline_score_predictions,
    walk_forward_predict,
)
from liquidity_radar.eval.metrics import (  # noqa: E402
    bootstrap_metric_ci,
    calibration_table,
    classification_report,
)
from liquidity_radar.eval.robustness import (  # noqa: E402
    auc_gain_bootstrap,
    coefficient_stability,
    horizon_sensitivity,
    model_comparison,
    subperiod_metrics,
    threshold_sensitivity,
)
from liquidity_radar.features.build import build_feature_matrix  # noqa: E402
from liquidity_radar.features.target import forward_drawdown_label  # noqa: E402
from liquidity_radar.models.logistic import FEATURE_COLS  # noqa: E402

ROBUST_DIR = DATA_DIR / "robustness"


def main() -> int:
    ensure_dirs()
    ROBUST_DIR.mkdir(parents=True, exist_ok=True)

    print("\n-- Loading panel + building features")
    with get_connection() as con:
        panel = get_features_panel(con)
    features = build_feature_matrix(panel)[FEATURE_COLS]
    target = forward_drawdown_label(panel)
    combined = features.join(target[["label"]]).dropna()
    print(
        f"   Panel {len(panel):,} rows · combined {len(combined):,} rows "
        f"· positive rate {combined['label'].mean():.2%}"
    )

    # 1. Model comparison (baseline / vol / liquidity / full / full-minus-liquidity)
    print("\n-- Model comparison")
    comp = model_comparison(combined, panel)
    comp.to_csv(ROBUST_DIR / "model_comparison.csv", index=False)
    print(comp[["model", "n_features", "roc_auc", "pr_auc", "brier"]].to_string(index=False))

    # 2. Full-model walk-forward predictions + coefficients (recomputed for self-containment)
    full_preds, fold_coefs = walk_forward_predict(combined, FEATURE_COLS)
    full_rep = classification_report(full_preds["actual"].to_numpy(), full_preds["prob"].to_numpy())

    # 3. Subperiod analysis
    print("\n-- Subperiod analysis")
    sub = subperiod_metrics(full_preds)
    sub.to_csv(ROBUST_DIR / "subperiod.csv", index=False)
    print(sub.to_string(index=False))

    # 4. Coefficient stability
    coef_stab = coefficient_stability(fold_coefs)
    coef_stab.to_csv(ROBUST_DIR / "coef_stability.csv", index=False)

    # 5. Threshold sensitivity
    print("\n-- Drawdown-threshold sensitivity")
    thr = threshold_sensitivity(features, panel)
    thr.to_csv(ROBUST_DIR / "threshold_sensitivity.csv", index=False)
    print(thr.to_string(index=False))

    # 6. Horizon sensitivity
    print("\n-- Forecast-horizon sensitivity")
    hor = horizon_sensitivity(features, panel)
    hor.to_csv(ROBUST_DIR / "horizon_sensitivity.csv", index=False)
    print(hor.to_string(index=False))

    # 7. Calibration
    calib = calibration_table(full_preds["actual"].to_numpy(), full_preds["prob"].to_numpy())
    calib.to_csv(ROBUST_DIR / "calibration.csv", index=False)

    # 8. Bootstrap CIs for headline metrics
    print("\n-- Bootstrap confidence intervals")
    ci_rows = []
    for m in ("roc_auc", "pr_auc", "brier"):
        ci = bootstrap_metric_ci(
            full_preds["actual"].to_numpy(), full_preds["prob"].to_numpy(), metric=m
        )
        ci_rows.append({"metric": m, **ci})
        print(f"   {m:8s}: {ci['point']:.4f}  [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")
    pd.DataFrame(ci_rows).to_csv(ROBUST_DIR / "metric_ci.csv", index=False)

    # 9. Paired AUC-gain over VIX baseline
    gain = {
        "gain": float("nan"),
        "ci_lo": float("nan"),
        "ci_hi": float("nan"),
        "prob_positive": float("nan"),
    }
    if "vix" in panel.columns:
        base_preds = baseline_score_predictions(combined, panel["vix"])
        gain = auc_gain_bootstrap(full_preds, base_preds)
        print(
            f"\n-- AUC gain over VIX baseline: {gain['gain']:+.4f}  "
            f"[{gain['ci_lo']:+.4f}, {gain['ci_hi']:+.4f}]  "
            f"P(gain>0)={gain['prob_positive']:.1%}"
        )

    # 10. Data coverage diagnostics
    qr = build_quality_report(panel.join(features))
    qr.summary_frame().to_csv(ROBUST_DIR / "data_coverage.csv")

    # 11. Headline summary JSON
    summary = {
        "asof": str(panel.index.max().date()),
        "n_obs": int(full_rep["n"]),
        "base_rate": full_rep["base_rate"],
        "roc_auc": full_rep["roc_auc"],
        "pr_auc": full_rep["pr_auc"],
        "brier": full_rep["brier"],
        "auc_gain_vs_vix": gain["gain"],
        "auc_gain_ci": [gain["ci_lo"], gain["ci_hi"]],
        "auc_gain_prob_positive": gain["prob_positive"],
        "n_folds": int(fold_coefs.shape[0]),
    }
    (ROBUST_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[OK] Robustness artefacts written to {ROBUST_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
