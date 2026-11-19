"""Shared fixtures. Tracking is off by default so tests never touch ./mlflow.db."""

import numpy as np
import pandas as pd
import pytest

from app.config import settings


@pytest.fixture(autouse=True)
def _tracking_disabled_by_default(monkeypatch):
    monkeypatch.setattr(settings, "MLFLOW_ENABLED", False)


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
