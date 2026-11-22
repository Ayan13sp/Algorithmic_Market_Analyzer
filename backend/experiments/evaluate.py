"""
Out-of-Sample Evaluation on Real NSE Data
=========================================
Produces docs/EVALUATION.md. For each ticker:

1. Walk-forward validation over the full history: expanding window,
   quarterly refits, hyperparameter search re-run at every fold.
2. Net-of-cost backtest of the out-of-sample signals against buy-and-hold
   and always-long, with a permutation test, cost sensitivity and threshold
   sensitivity.
3. Drift: today's report, naive vs calibrated flag rates over every past
   quarter, and whether drifted folds are the ones where the model struggles.
4. Feature ablation: the scale-free model features vs the raw rupee indicators.

It then runs the synthetic validation suite (sanity checks and the drift
benchmark) and writes everything to one Markdown report.

    python -m experiments.evaluate --tickers RELIANCE.NS TCS.NS INFY.NS HDFCBANK.NS ICICIBANK.NS --years 10
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from app import tracking
from app.backtest import CostModel, run_backtest
from app.config import settings
from app.drift import DriftConfig, drift_report, feature_drift, fold_drift
from app.ml_engine import (
    FEATURE_COLS,
    FEATURE_SET_VERSION,
    INDICATOR_COLS,
    backtest_signals,
    download_ohlcv,
    engineer_features,
    labelled_rows,
    train_and_predict,
)
from app.walk_forward import WalkForwardConfig
from experiments import backtest_sanity, drift_benchmark

logger = logging.getLogger("evaluate")

TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
EXPERIMENT = "walk-forward-evaluation"
CAPITAL = 100_000.0


def quarterly_flag_rates(df: pd.DataFrame, cfg: DriftConfig) -> Dict[str, Dict[str, float]]:
    """Share of features flagged per quarter by naive vs calibrated rules."""
    n, cw, rw = len(df), cfg.current_window, cfg.reference_window
    out: Dict[str, Dict[str, float]] = {}
    for name, cols in (("scale-free features", FEATURE_COLS), ("raw rupee indicators", INDICATOR_COLS)):
        rows = []
        for end in range(cw + rw, n + 1, cw):
            rep = feature_drift(df.iloc[end - cw - rw:end - cw], df.iloc[end - cw:end], cols, cfg)
            rows.append([
                np.mean([r["psi"] >= 0.25 for r in rep]),
                np.mean([r["ks_p_value"] < 0.05 for r in rep]),
                np.mean([r["drifted"] for r in rep]),
            ])
        mean = np.mean(rows, axis=0)
        out[name] = {"naive_psi": mean[0], "naive_ks": mean[1], "calibrated": mean[2], "quarters": len(rows)}
    return out


def evaluate_ticker(ticker: str, years: int, config: WalkForwardConfig) -> Dict[str, Any]:
    start = time.perf_counter()
    df = engineer_features(download_ohlcv(ticker, years))
    with tracking.start_run(
        run_name=f"{ticker}-evaluation",
        experiment_name=EXPERIMENT,
        tags={"ticker": ticker, "source": "evaluate"},
    ) as tracker:
        res = train_and_predict(df, ticker, wf_config=config, tracker=tracker)
        signals = res["signals"]
        bt = backtest_signals(df.loc[signals.index], signals, CAPITAL)
        tracker.log_backtest(bt)
        drift = drift_report(df, FEATURE_COLS, oos=res["walk_forward"].oos)
        tracker.log_drift(drift)

    wf = res["walk_forward"]
    labelled = labelled_rows(df)
    oos_prices = labelled.loc[wf.oos.index]
    thresholds = []
    for th in (0.50, 0.55, 0.60):
        sig = (wf.oos["proba"] > th).astype(int).to_numpy()
        t_bt = run_backtest(oos_prices, sig, CAPITAL, costs=CostModel())
        thresholds.append({
            "threshold": th,
            "trades": t_bt["trade_stats"]["n_trades"],
            "net_sharpe": t_bt["sharpe_ratio"],
            "edge_bps": t_bt["trade_stats"]["avg_gross_edge_bps"],
        })

    folds = pd.DataFrame(wf.folds).merge(pd.DataFrame(fold_drift(labelled, wf.fold_indices, FEATURE_COLS)), on="fold")
    raw = train_and_predict(df, ticker, feature_cols=INDICATOR_COLS, wf_config=config)

    logger.info("%s done in %.0fs", ticker, time.perf_counter() - start)
    return {
        "ticker": ticker,
        "data_start": df.index[0].strftime("%Y-%m-%d"),
        "data_end": df.index[-1].strftime("%Y-%m-%d"),
        "metrics": res["metrics"],
        "validation": res["validation"],
        "backtest": bt,
        "drift": drift,
        "thresholds": thresholds,
        "folds": folds.assign(ticker=ticker),
        "quarterly_flags": quarterly_flag_rates(df, DriftConfig()),
        "raw_metrics": raw["metrics"],
        "seconds": time.perf_counter() - start,
    }


# ═══════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════

def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _table(header: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def build_report(results: List[Dict[str, Any]], config: WalkForwardConfig, years: int,
                 sanity: List[Any], bench: pd.DataFrame, elapsed: float) -> str:
    costs = CostModel()
    folds = pd.concat([r["folds"] for r in results], ignore_index=True)
    n_sig_baseline = sum(r["metrics"]["p_value_vs_baseline"] < 0.05 for r in results)
    n_sig_perm = sum(r["backtest"]["permutation_p_value"] < 0.05 for r in results)
    n_ci_pos = sum(r["backtest"]["strategy_stats"]["sharpe_ci_low"] > 0 for r in results)
    bal = [r["metrics"]["balanced_accuracy"] for r in results]
    edges = [r["backtest"]["trade_stats"]["avg_gross_edge_bps"] for r in results
             if r["backtest"]["trade_stats"]["avg_gross_edge_bps"] is not None]
    rho_vol = stats.spearmanr(folds["vol_percentile"], folds["balanced_accuracy"])
    rho_psi = stats.spearmanr(folds["mean_psi"], folds["balanced_accuracy"])
    statuses = [r["drift"]["status"] for r in results]
    n = len(results)

    out = [
        "# Out-of-Sample Evaluation",
        "",
        f"Generated by `python -m experiments.evaluate` on {date.today():%Y-%m-%d} "
        f"({elapsed / 60:.0f} min). Every number below is out of sample: each prediction was made by "
        "a model trained, tuned and selected only on data from before that day.",
        "",
        "## Setup",
        "",
        _table(["Setting", "Value"], [
            ["Universe", ", ".join(r["ticker"] for r in results) + f" (daily, up to {years} years)"],
            ["Features", f"16 scale-free technical features (`{FEATURE_SET_VERSION}`)"],
            ["Target", "next session closes above its open (the trade the backtest makes)"],
            ["Validation", f"expanding-window walk-forward: first {config.initial_train_size} days to train, "
                           f"refit every {config.step} days, search re-run every fold"],
            ["Model selection", f"best of LR / RF / SVM by inner TimeSeriesSplit {config.scoring}"],
            ["Execution", "signal at the close of day t → buy the open, sell the close of day t+1"],
            ["Costs", f"{costs.round_trip_bps:.1f} bps per round trip (brokerage, exchange & SEBI fees, GST, "
                      "STT, stamp duty, slippage)"],
            ["Risk-free rate", f"{settings.RISK_FREE_RATE:.1%} (Sharpe uses excess returns; cash earns 0)"],
        ]),
        "",
        "## Headline",
        "",
        f"* Out-of-sample balanced accuracy ranges from **{min(bal):.3f} to {max(bal):.3f}** across "
        f"{n} stocks (chance = 0.500).",
        f"* **{n_sig_baseline}/{n}** stocks beat their majority-class baseline at the 5% level "
        "(exact McNemar test).",
        f"* **{n_sig_perm}/{n}** strategies beat random timing at the 5% level (permutation test, "
        f"same number of trades); **{n_ci_pos}/{n}** have a net Sharpe ratio whose 95% CI is above zero.",
        f"* The median gross edge per trade is **{np.median(edges):.1f} bps** against realistic round-trip "
        f"costs of {costs.round_trip_bps:.1f} bps.",
        f"* Drift monitor right now: **{statuses.count('drift')}** alert(s) and **{statuses.count('warning')}** "
        f"warning(s) across {n} stocks (section 3).",
        "",
        "## 1. Classification (out of sample)",
        "",
        _table(
            ["Ticker", "OOS days", "Folds", "Accuracy", "Balanced acc.", "ROC AUC", "Baseline acc.", "McNemar p"],
            [[r["ticker"], r["validation"]["n_oos"], r["validation"]["n_folds"],
              f"{r['metrics']['accuracy']:.3f}", f"**{r['metrics']['balanced_accuracy']:.3f}**",
              f"{r['metrics']['roc_auc']:.3f}", f"{r['metrics']['baseline_accuracy']:.3f}",
              _p(r["metrics"]["p_value_vs_baseline"])] for r in results],
        ),
        "",
        "Baseline = always predicting the majority class of each training window. Plain accuracy can "
        "look respectable just by matching it; balanced accuracy cannot.",
        "",
        "## 2. Backtest, net of costs",
        "",
        _table(
            ["Ticker", "Net return", "Gross return", "Buy & hold", "Net Sharpe [95% CI]", "Gross Sharpe",
             "B&H Sharpe", "Permutation p", "Max DD", "Trades/yr", "Gross edge"],
            [[r["ticker"], f"{r['backtest']['ml_return_pct']:+.1f}%", f"{r['backtest']['gross_return_pct']:+.1f}%",
              f"{r['backtest']['bh_return_pct']:+.1f}%",
              f"{r['backtest']['sharpe_ratio']:.2f} [{r['backtest']['strategy_stats']['sharpe_ci_low']:.2f}, "
              f"{r['backtest']['strategy_stats']['sharpe_ci_high']:.2f}]",
              f"{r['backtest']['gross_sharpe_ratio']:.2f}", f"{r['backtest']['buy_hold_stats']['sharpe_ratio']:.2f}",
              _p(r["backtest"]["permutation_p_value"]),
              f"{r['backtest']['strategy_stats']['max_drawdown_pct']:.1f}%",
              f"{r['backtest']['trade_stats']['trades_per_year']:.0f}",
              f"{r['backtest']['trade_stats']['avg_gross_edge_bps']:.1f} bps"] for r in results],
        ),
        "",
        "Gross edge = average open-to-close return per trade before costs, i.e. the break-even round-trip cost.",
        "",
        "### Cost sensitivity (net Sharpe)",
        "",
        _table(
            ["Ticker"] + [f"{int(p['round_trip_bps'])} bps" for p in results[0]["backtest"]["cost_sensitivity"]],
            [[r["ticker"]] + [f"{p['sharpe_ratio']:.2f}" for p in r["backtest"]["cost_sensitivity"]]
             for r in results],
        ),
        "",
        "### Decision-threshold sensitivity (reported, not optimised)",
        "",
        _table(
            ["Ticker"] + [f"P(up) > {t['threshold']:.2f}: trades / net Sharpe" for t in results[0]["thresholds"]],
            [[r["ticker"]] + [f"{t['trades']} / {t['net_sharpe']:.2f}" for t in r["thresholds"]] for r in results],
        ),
        "",
        "## 3. Drift monitoring",
        "",
        "### Latest quarter",
        "",
        _table(
            ["Ticker", "Status", "Current window", "Volatility", "Drifted features", "Concept (error rate)",
             "Messages"],
            [[r["ticker"], r["drift"]["status"], f"{r['drift']['current_start']} → {r['drift']['current_end']}",
              f"{r['drift']['volatility']['current_vol_pct']}% vs {r['drift']['volatility']['reference_vol_pct']}% "
              f"({r['drift']['volatility']['percentile']:.0f}th pct)",
              ", ".join(r["drift"]["drifted_features"]) or "none",
              f"{r['drift']['concept']['status']} ({r['drift']['concept'].get('recent_error_rate', 0):.1%} vs "
              f"{r['drift']['concept'].get('reference_error_rate', 0):.1%})",
              " ".join(r["drift"]["alerts"]) or "–"] for r in results],
        ),
        "",
        "### Why calibration: share of features flagged per quarter, all history",
        "",
        _table(
            ["Ticker", "Feature set", "Quarters", "Naive PSI ≥ 0.25", "Naive KS p < 0.05", "Calibrated rule"],
            [[r["ticker"], fs, v["quarters"], f"{v['naive_psi']:.0%}", f"{v['naive_ks']:.0%}",
              f"**{v['calibrated']:.0%}**"]
             for r in results for fs, v in r["quarterly_flags"].items()],
        ),
        "",
        "Naive rules flag most features in almost every quarter because the features are autocorrelated, "
        "so they carry no information. The calibrated rule compares each statistic with its own distribution "
        "over historical windows.",
        "",
        "### Do drifted folds hurt the model?",
        "",
        f"Across {len(folds)} walk-forward folds: Spearman ρ between a fold's volatility percentile and its "
        f"balanced accuracy = **{rho_vol.statistic:.3f}** (p = {_p(rho_vol.pvalue)}); between its mean feature "
        f"PSI and balanced accuracy = **{rho_psi.statistic:.3f}** (p = {_p(rho_psi.pvalue)}).",
        "",
        "## 4. Feature ablation: scale-free vs raw rupee indicators",
        "",
        _table(
            ["Ticker", "Balanced acc. (scale-free)", "Balanced acc. (raw)", "AUC (scale-free)", "AUC (raw)"],
            [[r["ticker"], f"{r['metrics']['balanced_accuracy']:.3f}", f"{r['raw_metrics']['balanced_accuracy']:.3f}",
              f"{r['metrics']['roc_auc']:.3f}", f"{r['raw_metrics']['roc_auc']:.3f}"] for r in results],
        ),
        "",
        "## 5. Synthetic validation (known answers)",
        "",
        "### Backtest sanity checks",
        "",
        backtest_sanity.to_markdown(sanity),
        "",
        "### Drift-detector benchmark",
        "",
        drift_benchmark.to_markdown(bench),
        "",
        "## Reproduce",
        "",
        "```bash",
        "cd backend",
        f"python -m experiments.evaluate --tickers {' '.join(r['ticker'] for r in results)} --years {years}",
        "```",
        "",
        "Results depend on the data available on the day it runs; MLflow logs every run "
        f"(experiment `{EXPERIMENT}`).",
    ]
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--retune-every", type=int, default=1)
    parser.add_argument("--trials", type=int, default=500, help="drift benchmark Monte Carlo trials")
    parser.add_argument("--output", default="../docs/EVALUATION.md")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s")
    settings.MLFLOW_REGISTER_MODELS = False  # evaluation models are not deployment candidates
    config = WalkForwardConfig(retune_every=args.retune_every, n_jobs=settings.ML_N_JOBS)

    start = time.perf_counter()
    results = [evaluate_ticker(t, args.years, config) for t in args.tickers]
    logger.info("Running synthetic validation suite …")
    sanity = backtest_sanity.run_checks()
    bench = drift_benchmark.run_benchmark(args.trials)
    report = build_report(results, config, args.years, sanity, bench, time.perf_counter() - start)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)
    logger.info("Report written to %s", args.output)


if __name__ == "__main__":
    main()
