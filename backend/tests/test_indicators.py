"""Unit tests for the pure indicator and backtest functions in app.ml_engine."""

import numpy as np
import pandas as pd
import pytest

from app.ml_engine import (
    FEATURE_COLS,
    compute_ema,
    compute_rsi,
    compute_sma,
    engineer_features,
    run_backtest,
)


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open_price": close + rng.normal(0, 0.3, n),
            "high_price": close + 1.0,
            "low_price": close - 1.0,
            "close_price": close,
            "volume": rng.integers(1_000, 5_000, n).astype(float),
        },
        index=idx,
    )


def test_sma_matches_manual_mean():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = compute_sma(s, 3)
    assert out.isna().sum() == 2
    assert out.iloc[-1] == pytest.approx(4.0)


def test_ema_first_value_equals_first_price():
    s = pd.Series([10.0, 11.0, 12.0])
    assert compute_ema(s, 3).iloc[0] == pytest.approx(10.0)


def test_rsi_is_bounded(ohlcv):
    rsi = compute_rsi(ohlcv["close_price"], 14).dropna()
    assert rsi.between(0, 100).all()


def test_engineer_features_outputs_all_columns_and_no_nans(ohlcv):
    out = engineer_features(ohlcv.copy())
    assert set(FEATURE_COLS).issubset(out.columns)
    assert not out[FEATURE_COLS + ["target"]].isna().any().any()
    assert set(out["target"].unique()) <= {0, 1}


def test_target_is_next_day_direction(ohlcv):
    out = engineer_features(ohlcv.copy())
    close = ohlcv["close_price"]
    day = out.index[10]
    nxt = close.index[close.index.get_loc(day) + 1]
    assert out.loc[day, "target"] == int(close[nxt] > close[day])


def test_backtest_all_cash_returns_flat_capital(ohlcv):
    test = ohlcv.iloc[:50]
    result = run_backtest(test, np.zeros(len(test), dtype=int), capital=100_000.0)
    assert result["ml_final_value"] == pytest.approx(100_000.0)
    assert result["ml_return_pct"] == pytest.approx(0.0)
    assert len(result["daily_cumulative"]) == len(test)
