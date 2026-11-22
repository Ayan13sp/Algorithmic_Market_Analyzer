"""
Synthetic Markets
=================
Deterministic generators for validating the pipeline in settings where the
right answer is known in advance. Used by the test suite, the CI backtest
sanity check, and the drift-detector benchmark.

* ``random_walk_ohlcv``: efficient market. Intraday returns are pure noise,
  so any model should score about 50% balanced accuracy and earn nothing.
* ``planted_signal_ohlcv``: a known, learnable edge (short-term mean
  reversion), so the pipeline must find it.
* ``garch_returns`` / ``ar1_series``: realistic nulls for drift detectors,
  with volatility clustering and autocorrelation but no regime change.
"""

from typing import Optional

import numpy as np
import pandas as pd


def _frame(opens: np.ndarray, closes: np.ndarray, rng: np.random.Generator,
           start: str, intraday_vol: float) -> pd.DataFrame:
    n = len(opens)
    wicks = np.abs(rng.normal(0.0, intraday_vol / 2, size=(2, n)))
    return pd.DataFrame(
        {
            "open_price": opens,
            "high_price": np.maximum(opens, closes) * np.exp(wicks[0]),
            "low_price": np.minimum(opens, closes) * np.exp(-wicks[1]),
            "close_price": closes,
            "volume": np.round(rng.lognormal(mean=13.8, sigma=0.35, size=n)),
        },
        index=pd.bdate_range(start, periods=n),
    )


def random_walk_ohlcv(
    n: int = 1000,
    seed: int = 0,
    intraday_vol: float = 0.012,
    overnight_vol: float = 0.006,
    start_price: float = 1000.0,
    start: str = "2018-01-01",
) -> pd.DataFrame:
    """Daily OHLCV where overnight gaps and intraday moves are iid Gaussian."""
    rng = np.random.default_rng(seed)
    gaps = rng.normal(0.0, overnight_vol, n)
    moves = rng.normal(0.0, intraday_vol, n)
    # open_t = close_{t-1} · e^{gap_t};  close_t = open_t · e^{move_t}
    log_open = np.log(start_price) + np.cumsum(gaps) + np.concatenate([[0.0], np.cumsum(moves)[:-1]])
    opens = np.exp(log_open)
    return _frame(opens, opens * np.exp(moves), rng, start, intraday_vol)


def planted_signal_ohlcv(
    n: int = 1500,
    seed: int = 0,
    edge: float = 0.004,
    intraday_vol: float = 0.012,
    overnight_vol: float = 0.006,
    lookback: int = 14,
    start_price: float = 1000.0,
    start: str = "2018-01-01",
) -> pd.DataFrame:
    """
    OHLCV with a planted mean-reversion edge.

    When yesterday's close is below its ``lookback``-day SMA, today's
    open-to-close return has mean +edge, otherwise −edge. The model's
    ``dist_sma_14`` feature encodes exactly this condition, so a working
    pipeline must learn it (P(up) ≈ Φ(edge / vol) ≈ 63% at the defaults).
    """
    rng = np.random.default_rng(seed)
    gaps = rng.normal(0.0, overnight_vol, n)
    noise = rng.normal(0.0, intraday_vol, n)
    opens, closes = np.empty(n), np.empty(n)
    prev_close = start_price
    for t in range(n):
        opens[t] = prev_close * np.exp(gaps[t])
        mean = 0.0
        if t >= lookback:
            mean = edge if closes[t - 1] < closes[t - lookback:t].mean() else -edge
        closes[t] = opens[t] * np.exp(mean + noise[t])
        prev_close = closes[t]
    return _frame(opens, closes, rng, start, intraday_vol)


def garch_returns(
    n: int,
    seed: int = 0,
    omega: float = 2e-6,
    alpha: float = 0.08,
    beta: float = 0.90,
    burn_in: int = 500,
    vol_multiplier_after: Optional[int] = None,
    vol_multiplier: float = 1.0,
) -> np.ndarray:
    """
    GARCH(1,1) daily returns (long-run vol = √(ω / (1 − α − β)) ≈ 1%/day).

    If ``vol_multiplier_after`` is set, returns from that index onward are
    scaled by ``vol_multiplier``: a genuine volatility regime shift.
    """
    rng = np.random.default_rng(seed)
    total = n + burn_in
    z = rng.standard_normal(total)
    r = np.empty(total)
    var = omega / (1 - alpha - beta)
    for t in range(total):
        r[t] = np.sqrt(var) * z[t]
        var = omega + alpha * r[t] ** 2 + beta * var
    out = r[burn_in:]
    if vol_multiplier_after is not None:
        out = out.copy()
        out[vol_multiplier_after:] *= vol_multiplier
    return out


def ar1_series(n: int, phi: float = 0.95, seed: int = 0) -> np.ndarray:
    """Stationary AR(1) with unit variance: x_t = φ·x_{t−1} + √(1 − φ²)·ε_t."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = eps[0]
    scale = np.sqrt(1 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + scale * eps[t]
    return x
