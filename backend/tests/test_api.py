"""End-to-end tests of every REST endpoint with FastAPI's TestClient.

Yahoo Finance is replaced by a synthetic market (with a planted edge) and the
database by a temporary SQLite file, so the full pipeline runs offline and
deterministically: download → features → walk-forward → backtest → drift.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import ml_engine
from app.config import settings
from app.database import Base, get_db
from app.main import app
from app.synthetic import planted_signal_ohlcv

TICKER = "SYN.NS"
UNKNOWN = "NOPE.NS"
SYNTHETIC = planted_signal_ohlcv(n=1100, seed=5)
NO_COSTS = {"commission_bps": 0, "slippage_bps": 0, "sell_tax_bps": 0, "buy_tax_bps": 0}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    engine = create_engine(
        f"sqlite:///{tmp_path_factory.mktemp('db') / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def override_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    def fake_download(ticker, years):
        if ticker == UNKNOWN:
            raise ml_engine.DataNotFoundError(f"No price data found for '{ticker}'.")
        return SYNTHETIC.copy()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(settings, "MLFLOW_ENABLED", False)
        mp.setattr(settings, "ML_N_JOBS", 1)
        mp.setattr(ml_engine, "download_ohlcv", fake_download)
        app.dependency_overrides[get_db] = override_db
        # No `with`: skips the lifespan, so the real dev database is never touched.
        yield TestClient(app)
        app.dependency_overrides.clear()
    engine.dispose()


@pytest.fixture(scope="module")
def analysis(client):
    res = client.post("/api/analyze", json={"ticker": TICKER, "years": 5})
    assert res.status_code == 200, res.text
    return res.json()


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok", "service": "algorithmic-market-analyzer"}


# ═══════════════════════════════════════════════════════════════════════════
# POST /api/analyze
# ═══════════════════════════════════════════════════════════════════════════

class TestAnalyze:
    def test_walk_forward_summary(self, analysis):
        v = analysis["validation"]
        assert v["method"] == "walk_forward" and v["n_folds"] >= 3
        assert analysis["test_size"] == v["n_oos"] == sum(f["n_test"] for f in v["folds"])
        assert analysis["train_size"] == v["initial_train_size"] == 504
        assert analysis["best_model"] in v["model_selection_counts"]

    def test_metrics_include_baseline_and_significance(self, analysis):
        m = analysis["metrics"]
        for key in ("accuracy", "balanced_accuracy", "roc_auc", "baseline_accuracy", "p_value_vs_baseline"):
            assert m[key] is not None, key
        assert set(analysis["all_model_metrics"]) == {"Logistic Regression", "Random Forest", "SVM"}

    def test_planted_edge_is_visible_through_the_api(self, analysis):
        assert analysis["metrics"]["balanced_accuracy"] > 0.55
        assert analysis["metrics"]["p_value_vs_baseline"] < 0.05

    def test_backtest_is_net_of_costs_with_benchmarks(self, analysis):
        bt = analysis["backtest"]
        assert bt["costs"]["round_trip_bps"] == pytest.approx(12.8)
        assert bt["ml_return_pct"] <= bt["gross_return_pct"]
        for key in ("strategy_stats", "buy_hold_stats", "always_long_stats", "trade_stats"):
            assert bt[key], key
        assert 0 < bt["permutation_p_value"] <= 1
        assert [p["round_trip_bps"] for p in bt["cost_sensitivity"]] == [0, 5, 10, 15, 20, 30, 50]
        # Every out-of-sample day plus the latest day (traded on yesterday's signal).
        assert len(bt["daily_cumulative"]) == analysis["test_size"] + 1

    def test_latest_prediction_is_for_the_next_session(self, analysis):
        latest_bar = analysis["market_data"][-1]
        assert analysis["latest_prediction"]["date"] == latest_bar["date"]
        assert latest_bar["target"] is None  # its outcome is not known yet
        assert 0.5 <= analysis["latest_prediction"]["confidence"] <= 1.0

    def test_drift_report(self, analysis):
        drift = analysis["drift"]
        assert drift["status"] in {"ok", "warning", "drift"}
        assert len(drift["features"]) == 16
        assert drift["volatility"]["regime"] in {"low", "normal", "high"}
        assert drift["concept"]["status"] in {"stable", "degraded"}


# ═══════════════════════════════════════════════════════════════════════════
# STORED DATA ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

class TestStoredData:
    def test_market_data(self, client, analysis):
        rows = client.get(f"/api/market-data/{TICKER}").json()
        assert len(rows) == analysis["data_points"]
        assert rows[-1]["target"] is None and rows[0]["target"] in (0, 1)

    def test_predictions_include_the_live_signal_and_accept_any_case(self, client, analysis):
        preds = client.get("/api/predictions/syn.ns").json()
        assert len(preds) == analysis["test_size"] + 1
        assert preds[-1]["date"] == analysis["latest_prediction"]["date"]

    def test_sql_window_functions_match_pandas(self, client, analysis):
        """Dual verification: SQL AVG() OVER (…) reproduces the pandas SMA-14."""
        sql = client.get(f"/api/sql-analytics/{TICKER}").json()
        pandas_sma = {row["date"]: row["sma_14"] for row in analysis["market_data"]}
        assert len(sql) == 100
        for row in sql:
            assert row["sql_sma_14"] == pytest.approx(pandas_sma[row["date"]], abs=0.011)

    def test_drift_endpoint_reproduces_the_analysis(self, client, analysis):
        drift = client.get(f"/api/drift/{TICKER}").json()
        for key in ("status", "share_drifted", "volatility", "concept"):
            assert drift[key] == analysis["drift"][key], key


# ═══════════════════════════════════════════════════════════════════════════
# POST /api/backtest
# ═══════════════════════════════════════════════════════════════════════════

class TestBacktestEndpoint:
    def test_reproduces_the_analysis_backtest_exactly(self, client, analysis):
        bt = client.post("/api/backtest", json={"ticker": TICKER, "capital": 100_000}).json()
        for key in ("ml_final_value", "ml_return_pct", "sharpe_ratio", "permutation_p_value", "bh_return_pct"):
            assert bt[key] == analysis["backtest"][key], key

    def test_does_not_download_or_retrain(self, client, analysis, monkeypatch):
        def forbidden(*args, **kwargs):
            raise AssertionError("/api/backtest must reuse stored signals")

        monkeypatch.setattr(ml_engine, "download_ohlcv", forbidden)
        monkeypatch.setattr(ml_engine, "train_and_predict", forbidden)
        assert client.post("/api/backtest", json={"ticker": TICKER}).status_code == 200

    def test_capital_and_costs_are_applied(self, client, analysis):
        base = client.post("/api/backtest", json={"ticker": TICKER, "capital": 100_000}).json()
        double = client.post("/api/backtest", json={"ticker": TICKER, "capital": 200_000}).json()
        free = client.post(
            "/api/backtest", json={"ticker": TICKER, "capital": 100_000, "costs": NO_COSTS},
        ).json()
        assert double["ml_final_value"] == pytest.approx(2 * base["ml_final_value"], abs=0.02)
        assert double["ml_return_pct"] == base["ml_return_pct"]
        assert free["costs"]["round_trip_bps"] == 0
        assert free["ml_return_pct"] == base["gross_return_pct"]


# ═══════════════════════════════════════════════════════════════════════════
# ERRORS
# ═══════════════════════════════════════════════════════════════════════════

class TestErrors:
    def test_unknown_ticker_is_404(self, client):
        res = client.post("/api/analyze", json={"ticker": UNKNOWN})
        assert res.status_code == 404 and UNKNOWN in res.json()["detail"]

    @pytest.mark.parametrize("path", ["market-data", "predictions", "drift", "sql-analytics"])
    def test_ticker_without_stored_data_is_404(self, client, path):
        assert client.get(f"/api/{path}/{UNKNOWN}").status_code == 404

    def test_backtest_before_analysis_is_404(self, client):
        assert client.post("/api/backtest", json={"ticker": UNKNOWN}).status_code == 404

    @pytest.mark.parametrize(
        "body",
        [
            {"ticker": "bad ticker!"},
            {"ticker": TICKER, "years": 1},
            {"ticker": TICKER, "costs": {"slippage_bps": -1}},
        ],
    )
    def test_invalid_requests_are_422(self, client, body):
        assert client.post("/api/analyze", json=body).status_code == 422

    def test_invalid_ticker_in_path_is_422(self, client):
        assert client.get("/api/market-data/bad%20ticker!").status_code == 422

    def test_too_little_history_is_422(self, client, monkeypatch):
        monkeypatch.setattr(ml_engine, "download_ohlcv", lambda ticker, years: SYNTHETIC.iloc[:300].copy())
        res = client.post("/api/analyze", json={"ticker": "SHORT.NS", "years": 2})
        assert res.status_code == 422 and "at least" in res.json()["detail"]

    def test_unexpected_errors_do_not_leak_internals(self, client, monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError("secret connection string")

        monkeypatch.setattr(ml_engine, "train_and_predict", explode)
        res = client.post("/api/analyze", json={"ticker": "BOOM.NS"})
        assert res.status_code == 500
        assert "secret" not in res.json()["detail"]
