"""
Backtest Sanity Checks
======================
Scenarios where the correct answer is known in advance. If any of them fails,
the backtester or the validation protocol is broken, whatever the headline
Sharpe ratio says. Runs in CI on every push and needs no network access.

    python -m experiments.backtest_sanity
    python -m experiments.backtest_sanity --summary "$GITHUB_STEP_SUMMARY"

Each check is also a pytest case (tests/test_sanity.py).
"""

from __future__ import annotations

import argparse
import functools
import itertools
import sys
import time
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pandas as pd

from app.backtest import CostModel, run_backtest
from app.config import settings
from app.drift import drift_report, volatility_regime
from app.ml_engine import FEATURE_COLS, backtest_signals, engineer_features, labelled_rows, train_and_predict
from app.synthetic import garch_returns, planted_signal_ohlcv, random_walk_ohlcv
from app.walk_forward import WalkForwardConfig, run_walk_forward

CAPITAL = 100_000.0


def _config(**overrides) -> WalkForwardConfig:
    """Built at call time so the worker count follows settings (1 under pytest)."""
    return WalkForwardConfig(**{"retune_every": 4, "n_jobs": settings.ML_N_JOBS, **overrides})


@dataclass
class CheckResult:
    name: str
    claim: str      # what must hold
    observed: str   # what was measured
    passed: bool
    seconds: float = 0.0


CHECKS: List[Callable[[], CheckResult]] = []


def check(fn: Callable[[], CheckResult]) -> Callable[[], CheckResult]:
    @functools.wraps(fn)
    def timed() -> CheckResult:
        start = time.perf_counter()
        result = fn()
        result.seconds = time.perf_counter() - start
        return result

    CHECKS.append(timed)
    return timed


@functools.lru_cache(maxsize=None)
def _walk_forward(kind: str):
    """Walk-forward + net-of-cost backtest on a 1,500-day synthetic market."""
    raw = (planted_signal_ohlcv if kind == "planted" else random_walk_ohlcv)(n=1500, seed=11)
    df = engineer_features(raw)
    res = train_and_predict(df, f"SYN-{kind}", wf_config=_config())
    signals = res["signals"]
    return res, backtest_signals(df.loc[signals.index], signals, CAPITAL)


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST MECHANICS
# ═══════════════════════════════════════════════════════════════════════════

@check
def all_cash_is_flat() -> CheckResult:
    prices = random_walk_ohlcv(300, seed=1)
    bt = run_backtest(prices, np.zeros(len(prices), dtype=int), CAPITAL)
    trades = bt["trade_stats"]["n_trades"]
    return CheckResult(
        "All-cash strategy is flat",
        "final value = capital, 0 trades, Sharpe 0",
        f"₹{bt['ml_final_value']:,.0f}, {trades} trades, Sharpe {bt['sharpe_ratio']}",
        bt["ml_final_value"] == CAPITAL and trades == 0 and bt["sharpe_ratio"] == 0,
    )


@check
def perfect_foresight_wins() -> CheckResult:
    prices = random_walk_ohlcv(500, seed=2)
    up_next = (prices["close_price"].shift(-1) > prices["open_price"].shift(-1)).astype(int)
    bt = run_backtest(prices, up_next, CAPITAL)
    hit = bt["trade_stats"]["gross_hit_rate"]
    return CheckResult(
        "Perfect foresight of the next session wins",
        "every trade is a gross winner and net Sharpe > 5",
        f"gross hit rate {hit:.0%}, net Sharpe {bt['sharpe_ratio']:.1f}",
        hit == 1.0 and bt["sharpe_ratio"] > 5,
    )


@check
def execution_lag_blocks_lookahead() -> CheckResult:
    """A signal that knows today's own move must not earn today's move."""
    prices = random_walk_ohlcv(500, seed=3)
    same_day = (prices["close_price"] > prices["open_price"]).astype(int)
    lagged = run_backtest(prices, same_day, CAPITAL, costs=CostModel.zero())["gross_sharpe_ratio"]
    intraday = prices["close_price"] / prices["open_price"] - 1
    leaky = (same_day * intraday).iloc[1:]  # what a missing shift(1) would book
    leaky_sharpe = leaky.mean() / leaky.std() * np.sqrt(252)
    return CheckResult(
        "Execution lag blocks look-ahead",
        "|Sharpe| < 2 with the T+1 lag (a missing lag would show > 10)",
        f"Sharpe {lagged:.2f} with lag vs {leaky_sharpe:.1f} without",
        abs(lagged) < 2 and leaky_sharpe > 10,
    )


@check
def costs_only_ever_hurt() -> CheckResult:
    prices = random_walk_ohlcv(500, seed=4)
    signals = np.random.default_rng(4).integers(0, 2, len(prices))
    returns = [
        run_backtest(prices, signals, CAPITAL, costs=CostModel.flat(bps))["ml_return_pct"]
        for bps in (0, 10, 20, 50)
    ]
    return CheckResult(
        "Higher costs strictly lower returns",
        "net return decreases at 0 → 10 → 20 → 50 bps",
        " → ".join(f"{r:+.1f}%" for r in returns),
        all(a > b for a, b in itertools.pairwise(returns)),
    )


@check
def benchmarks_match_closed_form() -> CheckResult:
    prices = random_walk_ohlcv(400, seed=5)
    bt = run_backtest(prices, np.ones(len(prices), dtype=int), CAPITAL, costs=CostModel.zero())
    o, c = prices["open_price"].to_numpy(), prices["close_price"].to_numpy()
    always_long = np.prod(c[1:] / o[1:]) - 1
    buy_hold = c[-1] / o[1] - 1
    err = max(
        abs(bt["always_long_stats"]["total_return_pct"] / 100 - always_long),
        abs(bt["bh_return_pct"] / 100 - buy_hold),
    )
    return CheckResult(
        "Benchmarks match their closed forms",
        "always-long = Π(close/open) − 1 and buy & hold = close_N / open_1 − 1",
        f"max abs error {err:.1e}",
        err < 1e-4,
    )


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════

@check
def no_false_discovery_on_random_walk() -> CheckResult:
    res, bt = _walk_forward("random")
    m = res["metrics"]
    return CheckResult(
        "No edge is found in an efficient (random-walk) market",
        "balanced accuracy within 0.50 ± 0.05; McNemar and permutation p > 0.05",
        f"balanced acc {m['balanced_accuracy']:.3f}, McNemar p {m['p_value_vs_baseline']:.2f}, "
        f"permutation p {bt['permutation_p_value']:.2f}",
        abs(m["balanced_accuracy"] - 0.5) < 0.05
        and m["p_value_vs_baseline"] > 0.05
        and bt["permutation_p_value"] > 0.05,
    )


@check
def planted_edge_is_recovered() -> CheckResult:
    res, bt = _walk_forward("planted")
    m = res["metrics"]
    return CheckResult(
        "A planted edge is recovered out of sample",
        "balanced acc > 0.55, AUC > 0.58, McNemar and permutation p < 0.01, net Sharpe > 0",
        f"balanced acc {m['balanced_accuracy']:.3f}, AUC {m['roc_auc']:.3f}, McNemar p "
        f"{m['p_value_vs_baseline']:.4f}, permutation p {bt['permutation_p_value']:.3f}, "
        f"net Sharpe {bt['sharpe_ratio']:.2f}",
        m["balanced_accuracy"] > 0.55
        and m["roc_auc"] > 0.58
        and m["p_value_vs_baseline"] < 0.01
        and bt["permutation_p_value"] < 0.01
        and bt["sharpe_ratio"] > 0,
    )


def _corrupt_after(raw: pd.DataFrame, day: pd.Timestamp, seed: int = 99) -> pd.DataFrame:
    """Scramble every price and volume strictly after ``day`` (stays positive)."""
    out = raw.copy()
    after = out.index > day
    noise = np.exp(np.random.default_rng(seed).normal(0, 0.05, (int(after.sum()), 5)))
    out.loc[after, :] = out.loc[after, :].to_numpy() * noise
    return out


@check
def features_are_causal() -> CheckResult:
    raw = random_walk_ohlcv(600, seed=6)
    cut = raw.index[450]
    clean = engineer_features(raw.copy())
    dirty = engineer_features(_corrupt_after(raw, cut))
    past = clean.index[clean.index < cut]  # rows strictly before the cut
    cols = FEATURE_COLS + ["target"]
    same = np.allclose(clean.loc[past[:-1], cols], dirty.loc[past[:-1], cols], equal_nan=True)
    return CheckResult(
        "Features never use future prices",
        "scrambling prices after day k leaves every feature/label before k unchanged",
        f"{len(past) - 1} past rows compared: {'identical' if same else 'CHANGED'}",
        bool(same),
    )


@check
def walk_forward_never_sees_the_future() -> CheckResult:
    raw = random_walk_ohlcv(900, seed=8)
    cfg = _config(initial_train_size=300, step=63, retune_every=0)
    clean = labelled_rows(engineer_features(raw.copy()))
    wf_clean = run_walk_forward(clean, FEATURE_COLS, cfg)
    test_idx = wf_clean.fold_indices[0][1]
    fold_days = clean.index[test_idx]
    # Scramble everything after fold 1's test block (+1 day: its last label uses that day).
    nxt = raw.index[raw.index.get_loc(fold_days[-1]) + 1]
    dirty = labelled_rows(engineer_features(_corrupt_after(raw, nxt)))
    wf_dirty = run_walk_forward(dirty, FEATURE_COLS, cfg)
    a = wf_clean.oos.loc[fold_days, "proba"].to_numpy()
    b = wf_dirty.oos.loc[fold_days, "proba"].to_numpy()
    changed_later = not np.allclose(
        wf_clean.oos["proba"].iloc[len(test_idx):], wf_dirty.oos["proba"].iloc[len(test_idx):]
    )
    return CheckResult(
        "Walk-forward predictions never see the future",
        "scrambling data after fold 1 leaves fold-1 predictions bit-identical",
        f"{len(a)} fold-1 predictions {'identical' if np.array_equal(a, b) else 'CHANGED'}; "
        f"later folds {'changed (as expected)' if changed_later else 'unchanged'}",
        bool(np.array_equal(a, b) and changed_later),
    )


# ═══════════════════════════════════════════════════════════════════════════
# DRIFT MONITOR
# ═══════════════════════════════════════════════════════════════════════════

@check
def volatility_shift_raises_alert() -> CheckResult:
    r = garch_returns(567, seed=21, vol_multiplier_after=504, vol_multiplier=2.0)
    vol = volatility_regime(r[:504], r[504:])
    return CheckResult(
        "A doubling of volatility raises an alert",
        "regime = high",
        f"{vol['current_vol_pct']}% vs {vol['reference_vol_pct']}% → {vol['percentile']:.0f}th pct, {vol['regime']}",
        vol["regime"] == "high",
    )


@check
def stationary_market_raises_no_alert() -> CheckResult:
    df = engineer_features(random_walk_ohlcv(900, seed=12))
    rep = drift_report(df, FEATURE_COLS)
    return CheckResult(
        "A stationary market raises no drift alert",
        "status ≠ drift and < 50% of features drifted",
        f"status {rep['status']}, {len(rep['drifted_features'])}/16 drifted, "
        f"volatility {rep['volatility']['percentile']:.0f}th pct",
        rep["status"] != "drift" and rep["share_drifted"] < 0.5,
    )


# ═══════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_checks() -> List[CheckResult]:
    return [fn() for fn in CHECKS]


def to_markdown(results: List[CheckResult]) -> str:
    lines = [
        "| | Check | Must hold | Observed | Time |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {'✅' if r.passed else '❌'} | {r.name} | {r.claim} | {r.observed} | {r.seconds:.1f}s |"
        )
    passed = sum(r.passed for r in results)
    lines.append(f"\n**{passed}/{len(results)} sanity checks passed.**")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", help="append a Markdown report to this file (e.g. $GITHUB_STEP_SUMMARY)")
    args = parser.parse_args()

    results = run_checks()
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"{'PASS' if r.passed else 'FAIL'}  {r.name:<{width}}  {r.observed}  ({r.seconds:.1f}s)")
    passed = sum(r.passed for r in results)
    print(f"\n{passed}/{len(results)} sanity checks passed")

    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as fh:
            fh.write("## Backtest sanity checks\n\n" + to_markdown(results) + "\n")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
