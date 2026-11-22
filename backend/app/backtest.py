"""
Backtesting Engine
==================
Simulates the ML strategy with realistic execution frictions and compares it
against two benchmarks, with the statistics a quant reviewer expects.

Execution model (no look-ahead)
-------------------------------
A signal generated from day t's features (known at t's close) is traded on
day t+1: buy at the open, sell at the close. Positions are never held
overnight, so every long day is one round trip. Cash earns nothing.

Frictions (``CostModel``, basis points of traded notional)
----------------------------------------------------------
    buy fill  = open  × (1 + commission + slippage + buy tax)
    sell fill = close × (1 − commission − slippage − sell tax)

The defaults approximate a discount-broker NSE intraday trade: ~3 bps per side
for brokerage + exchange/SEBI fees + GST, 2 bps per side of slippage, 2.5 bps
STT on the sell side and 0.3 bps stamp duty on the buy side, i.e. about
12.8 bps per round trip. Rates change, so ``cost_sensitivity`` reports how the
result moves across cost levels instead of relying on one number.

Benchmarks
----------
* Buy & hold: buy once at the first tradeable open, sell at the last close.
* Always-long intraday: trade every single day. This isolates timing skill;
  the model has to beat it to show it knows *which* days to trade.

Statistics
----------
* Annualised Sharpe ratio with a 95% confidence interval (Lo, 2002).
* Sortino ratio, max drawdown, Calmar ratio, hit rate, profit factor.
* Permutation test: shuffles the model's long days across the period (same
  number of trades, same costs) to ask whether its timing beats chance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from app.config import settings

BPS = 1e-4
_EPS = 1e-12


# ═══════════════════════════════════════════════════════════════════════════
# COST MODEL
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CostModel:
    """Per-trade frictions in basis points of traded notional."""

    commission_bps: float = 3.0   # per side: brokerage + exchange + SEBI fees + GST
    slippage_bps: float = 2.0     # per side: half-spread + impact at the open/close
    sell_tax_bps: float = 2.5     # STT on the intraday sell leg (0.025%)
    buy_tax_bps: float = 0.3      # stamp duty on the intraday buy leg (0.003%)

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative number, got {value}")

    @property
    def buy_cost(self) -> float:
        """Fractional cost added to the buy price."""
        return (self.commission_bps + self.slippage_bps + self.buy_tax_bps) * BPS

    @property
    def sell_cost(self) -> float:
        """Fractional cost deducted from the sell price."""
        return (self.commission_bps + self.slippage_bps + self.sell_tax_bps) * BPS

    @property
    def round_trip_bps(self) -> float:
        return 2 * (self.commission_bps + self.slippage_bps) + self.sell_tax_bps + self.buy_tax_bps

    @classmethod
    def zero(cls) -> "CostModel":
        return cls(0.0, 0.0, 0.0, 0.0)

    @classmethod
    def flat(cls, round_trip_bps: float) -> "CostModel":
        """Charge ``round_trip_bps`` per round trip, split evenly across both sides."""
        return cls(commission_bps=round_trip_bps / 2, slippage_bps=0.0, sell_tax_bps=0.0, buy_tax_bps=0.0)

    def as_dict(self) -> Dict[str, float]:
        return {**asdict(self), "round_trip_bps": round(self.round_trip_bps, 4)}


def intraday_returns(prices: pd.DataFrame, costs: CostModel) -> tuple[pd.Series, pd.Series]:
    """Open-to-close return of each day, before and after round-trip costs."""
    open_, close = prices["open_price"], prices["close_price"]
    gross = close / open_ - 1
    net = (close * (1 - costs.sell_cost)) / (open_ * (1 + costs.buy_cost)) - 1
    return gross, net


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def _daily_rf(rf_annual: float, periods: int) -> float:
    return (1 + rf_annual) ** (1 / periods) - 1


def performance_stats(
    returns: Iterable[float],
    rf_annual: Optional[float] = None,
    periods: Optional[int] = None,
) -> Dict[str, float]:
    """
    Risk/return statistics of a daily return series.

    Sharpe = mean(r − rf) / std(r − rf) × √periods, with its 95% confidence
    interval from the iid standard error SE = √((1 + SR²/2) / n) (Lo, 2002),
    where SR is the per-period Sharpe. A strategy with no variability (e.g.
    always in cash) gets Sharpe and Sortino of 0.
    """
    rf_annual = settings.RISK_FREE_RATE if rf_annual is None else rf_annual
    periods = periods or settings.TRADING_DAYS_PER_YEAR
    r = pd.Series(list(returns), dtype=float).dropna()
    n = len(r)
    if n < 2:
        raise ValueError(f"Need at least 2 daily returns, got {n}.")

    equity = (1 + r).cumprod()
    final = float(equity.iloc[-1])
    cagr = final ** (periods / n) - 1 if final > 0 else -1.0

    excess = r - _daily_rf(rf_annual, periods)
    sd = float(excess.std(ddof=1))
    if sd > _EPS:
        sr = float(excess.mean()) / sd
        downside = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
        sortino = float(excess.mean()) / downside * np.sqrt(periods) if downside > _EPS else 0.0
    else:
        sr, sortino = 0.0, 0.0
    sharpe = sr * np.sqrt(periods)
    se = np.sqrt((1 + 0.5 * sr**2) / n) * np.sqrt(periods)

    # Peak includes the starting capital, so a loss on day 1 is a drawdown.
    drawdown = equity / equity.cummax().clip(lower=1.0) - 1
    max_dd = float(drawdown.min())

    return {
        "total_return_pct": round((final - 1) * 100, 4),
        "cagr_pct": round(cagr * 100, 4),
        "volatility_pct": round(float(r.std(ddof=1)) * np.sqrt(periods) * 100, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sharpe_ci_low": round(sharpe - 1.96 * se, 4),
        "sharpe_ci_high": round(sharpe + 1.96 * se, 4),
        "sortino_ratio": round(float(sortino), 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "calmar_ratio": round(cagr / abs(max_dd), 4) if max_dd < 0 else 0.0,
        "n_days": n,
    }


def trade_stats(
    position: pd.Series,
    gross: pd.Series,
    net: pd.Series,
    periods: Optional[int] = None,
) -> Dict[str, Optional[float]]:
    """
    Per-trade statistics. Each long day is one round trip.

    ``avg_gross_edge_bps`` is the average open-to-close return captured per
    trade before costs, which is also the break-even round-trip cost: if
    trading costs more than this, the strategy loses money on average.
    """
    periods = periods or settings.TRADING_DAYS_PER_YEAR
    traded = position.to_numpy() > 0
    n_days, n_trades = len(position), int(traded.sum())
    if n_trades == 0:
        return {
            "n_trades": 0, "exposure_pct": 0.0, "trades_per_year": 0.0,
            "hit_rate": None, "gross_hit_rate": None,
            "avg_gross_edge_bps": None, "avg_net_trade_bps": None, "profit_factor": None,
        }
    g, nt = gross.to_numpy()[traded], net.to_numpy()[traded]
    gains, losses = nt[nt > 0].sum(), -nt[nt < 0].sum()
    return {
        "n_trades": n_trades,
        "exposure_pct": round(100 * n_trades / n_days, 4),
        "trades_per_year": round(n_trades / (n_days / periods), 2),
        "hit_rate": round(float((nt > 0).mean()), 4),
        "gross_hit_rate": round(float((g > 0).mean()), 4),
        "avg_gross_edge_bps": round(float(g.mean()) / BPS, 4),
        "avg_net_trade_bps": round(float(nt.mean()) / BPS, 4),
        "profit_factor": round(float(gains / losses), 4) if losses > 0 else None,
    }


def _sharpe_rows(strategy: np.ndarray, daily_rf: float, periods: int) -> np.ndarray:
    """Annualised Sharpe of each row of a (n_series, n_days) return matrix."""
    excess = strategy - daily_rf
    sd = excess.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(sd > _EPS, excess.mean(axis=1) / sd, 0.0)
    return sharpe * np.sqrt(periods)


def permutation_p_value(
    position: pd.Series,
    net: pd.Series,
    n_permutations: int = 1000,
    seed: int = 42,
    rf_annual: Optional[float] = None,
    periods: Optional[int] = None,
) -> float:
    """
    One-sided p-value of the strategy's net Sharpe under "no timing skill".

    The null keeps the number of long days (so the same cost drag) but places
    them at random dates: p = (1 + #{shuffled Sharpe ≥ observed}) / (1 + N).
    """
    rf_annual = settings.RISK_FREE_RATE if rf_annual is None else rf_annual
    periods = periods or settings.TRADING_DAYS_PER_YEAR
    daily_rf = _daily_rf(rf_annual, periods)
    pos, r = position.to_numpy(dtype=float), net.to_numpy(dtype=float)
    observed = _sharpe_rows((pos * r)[None, :], daily_rf, periods)[0]
    rng = np.random.default_rng(seed)
    shuffled = rng.permuted(np.tile(pos, (n_permutations, 1)), axis=1)
    null = _sharpe_rows(shuffled * r, daily_rf, periods)
    return float((1 + np.sum(null >= observed - 1e-9)) / (1 + n_permutations))


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════════════════════════

def run_backtest(
    test_df: pd.DataFrame,
    signals: Iterable[int],
    capital: float,
    costs: Optional[CostModel] = None,
    n_permutations: int = 0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Backtest 0/1 signals on out-of-sample days with realistic costs.

    Args:
        test_df: Out-of-sample days with ``open_price`` and ``close_price``.
        signals: One 0/1 signal per row, generated at that row's close.
        capital: Starting capital (INR).
        costs: Friction model; defaults to ``CostModel()`` (~12.8 bps round trip).
        n_permutations: If > 0, also run the permutation significance test.

    The first row can never trade (no earlier signal), so all strategies and
    benchmarks are measured on rows 1..N-1. Headline fields (``ml_*``,
    ``sharpe_ratio``) are net of costs.
    """
    if capital <= 0:
        raise ValueError("capital must be positive")
    costs = costs or CostModel()
    prices = test_df[["open_price", "close_price"]].astype(float)
    if len(prices) < 3:
        raise ValueError(f"Backtest needs at least 3 days, got {len(prices)}.")
    values = np.asarray(list(signals), dtype=float)
    if len(values) != len(prices) or not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("signals must contain exactly one 0/1 value per row")
    sig = pd.Series(values, index=prices.index)

    # Yesterday's signal drives today's open→close trade (execution lag).
    position = sig.shift(1).fillna(0.0)
    gross, net_long = intraday_returns(prices, costs)
    strat_gross = position * gross
    strat_net = position * net_long

    # Buy & hold over the same tradeable window, paying one round trip.
    close, open_ = prices["close_price"], prices["open_price"]
    bh = close.pct_change().fillna(0.0)
    bh.iloc[1] = close.iloc[1] / (open_.iloc[1] * (1 + costs.buy_cost)) - 1
    bh.iloc[-1] = (1 + bh.iloc[-1]) * (1 - costs.sell_cost) - 1

    window = slice(1, None)
    strategy = performance_stats(strat_net.iloc[window])
    gross_stats = performance_stats(strat_gross.iloc[window])
    buy_hold = performance_stats(bh.iloc[window])
    always_long = performance_stats(net_long.iloc[window])
    trades = trade_stats(position.iloc[window], gross.iloc[window], net_long.iloc[window])

    ml_curve = (1 + strat_net).cumprod()
    ml_gross_curve = (1 + strat_gross).cumprod()
    bh_curve = (1 + bh).cumprod()
    daily_cumulative = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "ml_cumulative": round(float(ml), 6),
            "bh_cumulative": round(float(b), 6),
            "ml_gross_cumulative": round(float(mg), 6),
        }
        for idx, ml, b, mg in zip(prices.index, ml_curve, bh_curve, ml_gross_curve, strict=True)
    ]

    result: Dict[str, Any] = {
        "starting_capital": capital,
        "ml_final_value": round(capital * float(ml_curve.iloc[-1]), 2),
        "bh_final_value": round(capital * float(bh_curve.iloc[-1]), 2),
        "ml_return_pct": round(strategy["total_return_pct"], 2),
        "bh_return_pct": round(buy_hold["total_return_pct"], 2),
        "sharpe_ratio": strategy["sharpe_ratio"],
        "gross_return_pct": round(gross_stats["total_return_pct"], 2),
        "gross_sharpe_ratio": gross_stats["sharpe_ratio"],
        "cost_drag_pct": round(gross_stats["total_return_pct"] - strategy["total_return_pct"], 2),
        "costs": costs.as_dict(),
        "strategy_stats": strategy,
        "buy_hold_stats": buy_hold,
        "always_long_stats": always_long,
        "trade_stats": trades,
        "permutation_p_value": None,
        "daily_cumulative": daily_cumulative,
    }
    if n_permutations > 0:
        result["permutation_p_value"] = round(
            permutation_p_value(position.iloc[window], net_long.iloc[window], n_permutations, seed), 4
        )
    return result


def cost_sensitivity(
    test_df: pd.DataFrame,
    signals: Iterable[int],
    round_trip_bps: Iterable[float] = (0, 5, 10, 15, 20, 30, 50),
) -> List[Dict[str, float]]:
    """Net return and Sharpe of the same signals across round-trip cost levels."""
    signals = list(signals)
    rows = []
    for bps in round_trip_bps:
        bt = run_backtest(test_df, signals, capital=1.0, costs=CostModel.flat(float(bps)))
        rows.append({
            "round_trip_bps": float(bps),
            "net_return_pct": bt["ml_return_pct"],
            "sharpe_ratio": bt["sharpe_ratio"],
        })
    return rows
