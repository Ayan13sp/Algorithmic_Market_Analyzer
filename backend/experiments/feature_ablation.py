"""
Feature-Ablation Study (tracked in MLflow)
==========================================
Measures how much each feature family (trend, momentum, volatility, volume)
contributes, by retraining the full model-selection pipeline with:

    * all       — all 16 features (reference)
    * no_<grp>  — every family except <grp>        (leave-one-group-out)
    * only_<grp>— a single family on its own       (--include-single)

Each (ticker, feature set) pair is one MLflow run with the 3 tuned
candidates as nested runs, all tagged with a shared ``study_id`` so they can
be filtered and compared side by side in the MLflow UI. A summary table is
printed and logged as a final ``ablation-summary`` run.

Usage (from backend/):
    python -m experiments.feature_ablation --tickers RELIANCE.NS TCS.NS
    python -m experiments.feature_ablation --tickers INFY.NS --years 8 --include-single
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # then open :5000
"""

from __future__ import annotations

import argparse
import logging
import uuid
from typing import Dict, List

import pandas as pd

from app import tracking
from app.config import settings
from app.ml_engine import (
    FEATURE_COLS,
    FEATURE_GROUPS,
    FEATURE_SET_VERSION,
    backtest_signals,
    download_ohlcv,
    engineer_features,
    train_and_predict,
)

logger = logging.getLogger("feature_ablation")

EXPERIMENT_NAME = "feature-ablation"


def feature_sets(include_single: bool) -> Dict[str, List[str]]:
    """Named feature subsets to evaluate (order preserved for the report)."""
    sets: Dict[str, List[str]] = {"all": list(FEATURE_COLS)}
    for group, cols in FEATURE_GROUPS.items():
        sets[f"no_{group}"] = [c for c in FEATURE_COLS if c not in cols]
    if include_single:
        for group, cols in FEATURE_GROUPS.items():
            sets[f"only_{group}"] = list(cols)
    return sets


def run_study(tickers: List[str], years: int, capital: float, include_single: bool) -> pd.DataFrame:
    study_id = uuid.uuid4().hex[:8]
    rows = []

    for ticker in tickers:
        df = engineer_features(download_ohlcv(ticker, years))
        logger.info("%s: %d rows after feature engineering", ticker, len(df))

        for set_name, cols in feature_sets(include_single).items():
            with tracking.start_run(
                run_name=f"{ticker}-{set_name}",
                experiment_name=EXPERIMENT_NAME,
                tags={
                    "study_id": study_id,
                    "ticker": ticker,
                    "feature_set": set_name,
                    "source": "feature_ablation",
                },
            ) as tracker:
                tracker.log_params({"lookback_years": years, "starting_capital": capital})
                res = train_and_predict(df, ticker, db=None, feature_cols=cols, tracker=tracker)
                signals = res["signals"]
                bt = backtest_signals(df.loc[signals.index], signals, capital)
                tracker.log_backtest(bt)

            rows.append({
                "ticker": ticker,
                "feature_set": set_name,
                "n_features": len(cols),
                "best_model": res["best_model_name"],
                "balanced_accuracy": res["metrics"]["balanced_accuracy"],
                "roc_auc": res["metrics"]["roc_auc"],
                "net_sharpe": bt["sharpe_ratio"],
                "ml_return_pct": bt["ml_return_pct"],
                "bh_return_pct": bt["bh_return_pct"],
                "mlflow_run_id": tracker.run_id,
            })

    summary = pd.DataFrame(rows)
    # Out-of-sample balanced-accuracy change vs. the all-features reference for
    # the same ticker: a large drop when a family is removed means it carries signal.
    ref = summary[summary.feature_set == "all"].set_index("ticker")["balanced_accuracy"]
    summary["delta_bal_acc_vs_all"] = (summary["balanced_accuracy"] - summary["ticker"].map(ref)).round(4)

    with tracking.start_run(
        run_name="ablation-summary",
        experiment_name=EXPERIMENT_NAME,
        tags={"study_id": study_id, "source": "feature_ablation"},
    ) as tracker:
        tracker.log_params({
            "tickers": ",".join(tickers),
            "lookback_years": years,
            "feature_set_version": FEATURE_SET_VERSION,
            "n_configs": len(summary),
        })
        tracker.log_dict(summary.to_dict(orient="records"), "ablation_summary.json")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=["RELIANCE.NS"])
    parser.add_argument("--years", type=int, default=settings.DEFAULT_LOOKBACK_YEARS)
    parser.add_argument("--capital", type=float, default=settings.DEFAULT_STARTING_CAPITAL)
    parser.add_argument("--include-single", action="store_true", help="also train on each family alone")
    parser.add_argument("--register", action="store_true", help="register every model in the Model Registry")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s")
    # Ablation produces many throwaway models; keep the registry for API runs.
    settings.MLFLOW_REGISTER_MODELS = args.register
    if not tracking.is_enabled():
        logger.warning("MLflow tracking is disabled or not installed — results will not be logged.")

    summary = run_study(args.tickers, args.years, args.capital, args.include_single)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("\n" + summary.drop(columns="mlflow_run_id").to_string(index=False))
    print(f"\nTracking URI: {settings.MLFLOW_TRACKING_URI}  |  experiment: {EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()
