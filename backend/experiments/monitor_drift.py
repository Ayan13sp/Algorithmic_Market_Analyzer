"""
Scheduled Drift Monitor
=======================
Downloads the latest prices for a watch-list and checks whether each ticker's
latest quarter has drifted away from the two years before it: per-feature
PSI/KS (empirically calibrated) and the volatility regime. No model training
is needed, so it is cheap enough to run on a schedule in GitHub Actions
(.github/workflows/drift-monitor.yml).

Exit code 1 means at least one ticker raised a drift alert: the workflow run
fails and GitHub notifies the repository owner. Download problems are reported
as warnings and never fail the run.

    python -m experiments.monitor_drift --tickers RELIANCE.NS TCS.NS --fail-on-drift
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict

from app.drift import drift_report
from app.ml_engine import FEATURE_COLS, download_ohlcv, engineer_features

DEFAULT_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
ICONS = {"ok": "🟢", "warning": "🟡", "drift": "🔴", "insufficient_data": "⚪"}


def check_ticker(ticker: str, years: int) -> Dict[str, Any]:
    return drift_report(engineer_features(download_ohlcv(ticker, years)), FEATURE_COLS)


def to_markdown(reports: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "| Ticker | Status | Window | Volatility (current vs reference) | Drifted features | Alerts |",
        "|---|---|---|---|---|---|",
    ]
    for ticker, rep in reports.items():
        vol = rep.get("volatility") or {}
        vol_text = (
            f"{vol['current_vol_pct']}% vs {vol['reference_vol_pct']}% "
            f"({vol['percentile']:.0f}th pct, {vol['ratio']}×, {vol['regime']})" if vol else "–"
        )
        drifted = ", ".join(rep.get("drifted_features") or []) or "none"
        lines.append(
            f"| {ticker} | {ICONS.get(rep['status'], '')} {rep['status']} | "
            f"{rep.get('current_start', '–')} → {rep.get('current_end', '–')} | {vol_text} | "
            f"{drifted} | {' '.join(rep.get('alerts') or []) or '–'} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--years", type=int, default=4, help="history to download (≥ 4 covers warm-up + windows)")
    parser.add_argument("--summary", help="append a Markdown report to this file (e.g. $GITHUB_STEP_SUMMARY)")
    parser.add_argument("--fail-on-drift", action="store_true", help="exit 1 if any ticker raises a drift alert")
    args = parser.parse_args()

    reports: Dict[str, Dict[str, Any]] = {}
    for ticker in args.tickers:
        try:
            reports[ticker] = check_ticker(ticker, args.years)
        except Exception as exc:  # noqa: BLE001 - a data outage is not a drift alert
            print(f"::warning title=Drift monitor ({ticker})::could not check {ticker}: {exc}")
            continue
        rep = reports[ticker]
        print(f"{ICONS.get(rep['status'], '')} {ticker:<14} {rep['status']:<17} " + " | ".join(rep["alerts"]))

    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as fh:
            fh.write("## Drift monitor\n\n" + to_markdown(reports) + "\n")

    alerts = {t: r for t, r in reports.items() if r["status"] == "drift"}
    for ticker, rep in alerts.items():
        for alert in rep["alerts"]:
            print(f"::error title=Drift alert ({ticker})::{alert}")
    return 1 if alerts and args.fail_on_drift else 0


if __name__ == "__main__":
    sys.exit(main())
