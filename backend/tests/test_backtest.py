"""Tests for the backtesting engine (app.backtest): cost model, statistics,
execution timing, benchmarks, significance testing and invariants."""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.backtest import (
    CostModel,
    cost_sensitivity,
    intraday_returns,
    performance_stats,
    permutation_p_value,
    run_backtest,
    trade_stats,
)
from app.synthetic import random_walk_ohlcv


def _prices(opens, closes) -> pd.DataFrame:
    return pd.DataFrame(
        {"open_price": opens, "close_price": closes},
        index=pd.bdate_range("2024-01-01", periods=len(opens)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# COST MODEL
# ═══════════════════════════════════════════════════════════════════════════

class TestCostModel:
    def test_default_round_trip_is_nse_intraday_estimate(self):
        costs = CostModel()
        assert costs.round_trip_bps == pytest.approx(2 * (3.0 + 2.0) + 2.5 + 0.3)  # 12.8 bps
        assert costs.buy_cost == pytest.approx((3.0 + 2.0 + 0.3) * 1e-4)
        assert costs.sell_cost == pytest.approx((3.0 + 2.0 + 2.5) * 1e-4)

    def test_flat_and_zero(self):
        assert CostModel.flat(20).round_trip_bps == pytest.approx(20)
        assert CostModel.zero().round_trip_bps == 0

    @pytest.mark.parametrize("field", ["commission_bps", "slippage_bps", "sell_tax_bps", "buy_tax_bps"])
    def test_rejects_negative_costs(self, field):
        with pytest.raises(ValueError, match=field):
            CostModel(**{field: -1.0})

    def test_fills_apply_costs_on_both_legs(self):
        prices = _prices([100.0], [101.0])
        costs = CostModel()
        gross, net = intraday_returns(prices, costs)
        assert gross.iloc[0] == pytest.approx(0.01)
        expected = 101 * (1 - costs.sell_cost) / (100 * (1 + costs.buy_cost)) - 1
        assert net.iloc[0] == pytest.approx(expected)
        assert net.iloc[0] < gross.iloc[0]


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

class TestPerformanceStats:
    def test_matches_hand_computation(self):
        r = pd.Series([0.01, -0.02, 0.03, 0.0])
        stats = performance_stats(r, rf_annual=0.0)
        assert stats["total_return_pct"] == pytest.approx((np.prod(1 + r) - 1) * 100, abs=1e-3)
        assert stats["sharpe_ratio"] == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(252), abs=1e-3)
        # equity 1.01 → 0.9898 → ...: drawdown from the 1.01 peak = −2%.
        assert stats["max_drawdown_pct"] == pytest.approx(-2.0, abs=1e-3)
        assert stats["n_days"] == 4

    def test_drawdown_counts_a_loss_from_starting_capital(self):
        assert performance_stats([-0.10, 0.05], rf_annual=0.0)["max_drawdown_pct"] == pytest.approx(-10.0)

    def test_zero_variance_strategy_has_zero_sharpe_and_sortino(self):
        stats = performance_stats(np.zeros(50))
        assert stats["sharpe_ratio"] == 0 and stats["sortino_ratio"] == 0
        assert stats["volatility_pct"] == 0

    def test_confidence_interval_brackets_sharpe_and_shrinks_with_data(self):
        rng = np.random.default_rng(0)
        short = performance_stats(rng.normal(0.001, 0.01, 250))
        long = performance_stats(rng.normal(0.001, 0.01, 2500))
        for s in (short, long):
            assert s["sharpe_ci_low"] < s["sharpe_ratio"] < s["sharpe_ci_high"]
        width = lambda s: s["sharpe_ci_high"] - s["sharpe_ci_low"]  # noqa: E731
        assert width(long) < width(short) / 2.5  # SE ∝ 1/√n

    def test_risk_free_rate_lowers_sharpe(self):
        r = np.random.default_rng(1).normal(0.0005, 0.01, 500)
        with_rf, without_rf = performance_stats(r, rf_annual=0.10), performance_stats(r, rf_annual=0.0)
        assert with_rf["sharpe_ratio"] < without_rf["sharpe_ratio"]

    def test_needs_two_returns(self):
        with pytest.raises(ValueError):
            performance_stats([0.01])


class TestTradeStats:
    def test_hand_example(self):
        position = pd.Series([1, 0, 1, 1])
        gross = pd.Series([0.010, 0.050, -0.004, 0.002])
        net = gross - 0.001
        stats = trade_stats(position, gross, net)
        assert stats["n_trades"] == 3
        assert stats["exposure_pct"] == pytest.approx(75.0)
        assert stats["avg_gross_edge_bps"] == pytest.approx((0.010 - 0.004 + 0.002) / 3 * 1e4, abs=1e-3)
        assert stats["hit_rate"] == pytest.approx(2 / 3, abs=1e-4)  # net: +0.009, −0.005, +0.001
        assert stats["profit_factor"] == pytest.approx((0.009 + 0.001) / 0.005)

    def test_no_trades(self):
        stats = trade_stats(pd.Series([0, 0, 0]), pd.Series([0.1, 0.2, 0.3]), pd.Series([0.1, 0.2, 0.3]))
        assert stats["n_trades"] == 0 and stats["hit_rate"] is None and stats["profit_factor"] is None


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════════════════════════

class TestRunBacktest:
    def test_signal_trades_the_next_session_only(self):
        # A single buy signal on day 1 must capture day 2's open→close move, nothing else.
        prices = _prices([100, 100, 100, 100, 100], [99, 101, 110, 95, 100])
        signals = [0, 1, 0, 0, 0]
        bt = run_backtest(prices, signals, 1_000.0, costs=CostModel.zero())
        assert bt["ml_return_pct"] == pytest.approx(10.0)
        assert bt["trade_stats"]["n_trades"] == 1

    def test_costs_are_charged_on_every_trade(self):
        prices = _prices([100.0] * 6, [100.0] * 6)  # flat market: every trade only pays costs
        costs = CostModel.flat(20)
        bt = run_backtest(prices, [1] * 6, 1_000.0, costs=costs)
        per_trade = (1 - costs.sell_cost) / (1 + costs.buy_cost)
        assert bt["ml_final_value"] == pytest.approx(1_000 * per_trade**5, abs=0.01)
        assert bt["gross_return_pct"] == 0
        assert bt["cost_drag_pct"] > 0

    def test_buy_and_hold_pays_one_round_trip(self):
        prices = random_walk_ohlcv(100, seed=3)
        costs = CostModel()
        bt = run_backtest(prices, np.zeros(100, dtype=int), 1.0, costs=costs)
        o1, cn = prices["open_price"].iloc[1], prices["close_price"].iloc[-1]
        expected = cn * (1 - costs.sell_cost) / (o1 * (1 + costs.buy_cost)) - 1
        assert bt["bh_return_pct"] == pytest.approx(expected * 100, abs=0.01)

    def test_equity_curve_starts_at_one_and_covers_every_day(self):
        prices = random_walk_ohlcv(60, seed=4)
        bt = run_backtest(prices, np.ones(60, dtype=int), 1.0)
        curve = bt["daily_cumulative"]
        assert len(curve) == 60
        assert curve[0]["ml_cumulative"] == 1.0 and curve[0]["bh_cumulative"] == 1.0
        assert curve[-1]["ml_cumulative"] < curve[-1]["ml_gross_cumulative"]

    @pytest.mark.parametrize(
        "signals, capital, error",
        [([0, 1], 1.0, "one 0/1"), ([0, 2, 0, 1, 0], 1.0, "0/1"), ([0] * 5, 0.0, "capital")],
    )
    def test_rejects_invalid_inputs(self, signals, capital, error):
        with pytest.raises(ValueError, match=error):
            run_backtest(_prices([1.0] * 5, [1.0] * 5), signals, capital)

    def test_needs_three_days(self):
        with pytest.raises(ValueError, match="at least 3"):
            run_backtest(_prices([1.0, 1.0], [1.0, 1.0]), [0, 0], 1.0)

    def test_permutation_test_separates_skill_from_luck(self):
        prices = random_walk_ohlcv(400, seed=5)
        oracle = (prices["close_price"].shift(-1) > prices["open_price"].shift(-1)).astype(int)
        random_signals = np.random.default_rng(5).integers(0, 2, 400)
        skilled = run_backtest(prices, oracle, 1.0, n_permutations=500)
        lucky = run_backtest(prices, random_signals, 1.0, n_permutations=500)
        assert skilled["permutation_p_value"] <= 0.01
        assert lucky["permutation_p_value"] > 0.05

    def test_permutation_p_value_is_one_without_trades(self):
        position = pd.Series(np.zeros(50))
        net = pd.Series(np.random.default_rng(0).normal(0, 0.01, 50))
        assert permutation_p_value(position, net, n_permutations=100) == 1.0

    def test_cost_sensitivity_is_monotone_and_starts_at_gross(self):
        prices = random_walk_ohlcv(300, seed=6)
        signals = np.random.default_rng(6).integers(0, 2, 300)
        rows = cost_sensitivity(prices, signals, round_trip_bps=(0, 10, 25, 50))
        returns = [r["net_return_pct"] for r in rows]
        assert returns == sorted(returns, reverse=True) and len(set(returns)) == 4
        gross = run_backtest(prices, signals, 1.0, costs=CostModel.zero())["ml_return_pct"]
        assert returns[0] == pytest.approx(gross)


# ═══════════════════════════════════════════════════════════════════════════
# PROPERTY-BASED INVARIANTS (Hypothesis)
# ═══════════════════════════════════════════════════════════════════════════

market = st.tuples(
    st.integers(min_value=5, max_value=60),   # days
    st.integers(min_value=0, max_value=10_000),  # seed
    st.floats(min_value=0.0, max_value=60.0),  # round-trip cost (bps)
)


@settings(max_examples=60, deadline=None)
@given(market)
def test_costs_never_improve_returns(params):
    days, seed, bps = params
    prices = random_walk_ohlcv(days, seed=seed)
    signals = np.random.default_rng(seed).integers(0, 2, days)
    bt = run_backtest(prices, signals, 1.0, costs=CostModel.flat(bps))
    assert bt["ml_return_pct"] <= bt["gross_return_pct"] + 1e-9
    assert -100.0 <= bt["strategy_stats"]["max_drawdown_pct"] <= 0.0


@settings(max_examples=40, deadline=None)
@given(market, st.floats(min_value=1.0, max_value=1e7))
def test_capital_scales_value_but_not_returns(params, capital):
    days, seed, bps = params
    prices = random_walk_ohlcv(days, seed=seed)
    signals = np.random.default_rng(seed).integers(0, 2, days)
    small = run_backtest(prices, signals, 1.0, costs=CostModel.flat(bps))
    big = run_backtest(prices, signals, capital, costs=CostModel.flat(bps))
    assert big["ml_return_pct"] == small["ml_return_pct"]
    assert big["sharpe_ratio"] == small["sharpe_ratio"]
    growth = small["daily_cumulative"][-1]["ml_cumulative"]
    assert big["ml_final_value"] == pytest.approx(capital * growth, rel=1e-5, abs=0.01)
