"""Service plumbing: Yahoo Finance parsing, app start-up, settings and DB sessions."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect

from app import database, main, ml_engine
from app.config import Settings


def _yahoo_frame() -> pd.DataFrame:
    """What yfinance returns: MultiIndex columns, unsorted dates, gaps."""
    idx = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04"])
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["TCS.NS"]])
    data = np.array([
        [101, 103, 100, 102, 2_000],
        [100, 102, 99, 101, 1_000],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
    ], dtype=float)
    return pd.DataFrame(data, index=idx, columns=cols)


def test_download_ohlcv_flattens_sorts_and_fills(monkeypatch):
    monkeypatch.setattr(ml_engine.yf, "download", lambda *args, **kwargs: _yahoo_frame())
    out = ml_engine.download_ohlcv("TCS.NS", 2)
    assert list(out.columns) == ["open_price", "high_price", "low_price", "close_price", "volume"]
    assert out.index.is_monotonic_increasing
    assert out.loc["2024-01-04", "close_price"] == 102  # forward-filled gap
    assert not out.isna().any().any()


def test_download_ohlcv_unknown_symbol(monkeypatch):
    monkeypatch.setattr(ml_engine.yf, "download", lambda *args, **kwargs: pd.DataFrame())
    with pytest.raises(ml_engine.DataNotFoundError, match=r"\.NS"):
        ml_engine.download_ohlcv("NOPE", 2)


def test_startup_creates_tables(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'startup.db'}")
    monkeypatch.setattr(main, "engine", engine)
    with TestClient(main.app) as client:  # runs the lifespan
        assert client.get("/api/health").status_code == 200
    assert {"market_data", "model_predictions"} <= set(inspect(engine).get_table_names())


def test_extra_cors_origins_are_appended():
    s = Settings(EXTRA_CORS_ORIGINS="https://a.example, https://b.example ,")
    assert s.all_cors_origins[-2:] == ["https://a.example", "https://b.example"]


def test_get_db_closes_the_session(monkeypatch):
    closed = []

    class FakeSession:
        def close(self):
            closed.append(True)

    monkeypatch.setattr(database, "SessionLocal", FakeSession)
    gen = database.get_db()
    assert isinstance(next(gen), FakeSession)
    gen.close()
    assert closed == [True]
