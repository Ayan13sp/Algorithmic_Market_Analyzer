"""
API Routes
==========
REST endpoints for the Algorithmic Market Analyzer.

| Method | Endpoint                  | Description                                                   |
|--------|---------------------------|---------------------------------------------------------------|
| POST   | /api/analyze              | Full pipeline: fetch → walk-forward → backtest → drift report |
| GET    | /api/market-data/{tkr}    | Stored OHLCV + indicators for charting                        |
| GET    | /api/predictions/{tkr}    | Out-of-sample predictions + the live next-session signal      |
| GET    | /api/sql-analytics/{tkr}  | SQL window functions demo                                     |
| POST   | /api/backtest             | Re-run the backtest with custom capital/costs (no retraining) |
| GET    | /api/drift/{tkr}          | Data, volatility and concept drift of the latest quarter      |
| GET    | /api/health               | Service health check                                          |

Errors: 404 when a ticker has no data, 422 for invalid input or too little
history, 500 (without internal details) for anything unexpected.
"""

import logging
from typing import List, NoReturn, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.backtest import CostModel
from app.config import settings
from app.database import get_db
from app.ml_engine import DataNotFoundError, backtest_from_db, drift_from_db, run_full_analysis
from app.models import MarketData, ModelPrediction
from app.schemas import (
    AnalysisResponse,
    AnalyzeRequest,
    BacktestRequest,
    BacktestResult,
    CostAssumptions,
    DriftReport,
    HealthResponse,
    MarketDataPoint,
    PredictionDetail,
    normalise_ticker,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Market Analysis"])


def _ticker(raw: str) -> str:
    """Validate a ticker from the URL path (422 if malformed)."""
    try:
        return normalise_ticker(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _cost_model(costs: Optional[CostAssumptions]) -> CostModel:
    if costs is None:
        return CostModel()
    return CostModel(
        commission_bps=costs.commission_bps,
        slippage_bps=costs.slippage_bps,
        sell_tax_bps=costs.sell_tax_bps,
        buy_tax_bps=costs.buy_tax_bps,
    )


def _raise_http(exc: Exception, ticker: str, action: str) -> NoReturn:
    """Map pipeline exceptions to HTTP errors without leaking internals."""
    if isinstance(exc, DataNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, ValueError):  # bad input or not enough history
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    logger.exception("%s failed for %s", action, ticker)
    raise HTTPException(
        status_code=500, detail=f"{action} failed for {ticker}. Check the server logs.",
    ) from exc


# ═══════════════════════════════════════════════════════════════════════════
# POST /api/analyze — Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/analyze", response_model=AnalysisResponse)
def analyze_ticker(request: AnalyzeRequest, db: Session = Depends(get_db)):
    """
    Run the complete analysis pipeline for a given ticker:
    data acquisition → feature engineering → walk-forward training →
    net-of-cost backtest → drift report.

    Returns all data the frontend needs to render the dashboard.
    """
    try:
        result = run_full_analysis(
            ticker=request.ticker,
            years=request.years,
            capital=settings.DEFAULT_STARTING_CAPITAL,
            db=db,
            costs=_cost_model(request.costs),
        )
    except Exception as exc:
        _raise_http(exc, request.ticker, "Analysis")
    return AnalysisResponse(**result)


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/market-data/{ticker} — Historical Data
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/market-data/{ticker}", response_model=List[MarketDataPoint])
def get_market_data(ticker: str, db: Session = Depends(get_db)):
    """
    Retrieve stored OHLCV + engineered indicators for a ticker.
    Used by the frontend charting components.
    """
    ticker = _ticker(ticker)
    rows = (
        db.query(MarketData)
        .filter(MarketData.ticker == ticker)
        .order_by(MarketData.date)
        .all()
    )
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for ticker '{ticker}'. Run /api/analyze first.",
        )

    return [
        MarketDataPoint(
            date=r.date,
            open=r.open_price,
            high=r.high_price,
            low=r.low_price,
            close=r.close_price,
            volume=r.volume,
            sma_14=r.sma_14,
            sma_50=r.sma_50,
            sma_200=r.sma_200,
            ema_14=r.ema_14,
            ema_20=r.ema_20,
            ema_50=r.ema_50,
            rsi_14=r.rsi_14,
            macd_line=r.macd_line,
            macd_signal=r.macd_signal,
            macd_hist=r.macd_hist,
            bb_upper=r.bb_upper,
            bb_lower=r.bb_lower,
            atr_14=r.atr_14,
            daily_return_vol=r.daily_return_vol,
            volume_ratio=r.volume_ratio,
            volume_change=r.volume_change,
            target=r.target,
        )
        for r in rows
    ]


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/predictions/{ticker} — Predictions
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/predictions/{ticker}", response_model=List[PredictionDetail])
def get_predictions(ticker: str, db: Session = Depends(get_db)):
    """
    Out-of-sample walk-forward predictions for a ticker, ordered by date.
    The last entry is the live signal for the next session.
    """
    ticker = _ticker(ticker)
    rows = (
        db.query(ModelPrediction)
        .filter(ModelPrediction.ticker == ticker)
        .order_by(ModelPrediction.prediction_date)
        .all()
    )
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for '{ticker}'. Run /api/analyze first.",
        )

    return [
        PredictionDetail(
            date=r.prediction_date,
            signal=r.predicted_signal,
            confidence=r.confidence_score,
        )
        for r in rows
    ]


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/sql-analytics/{ticker} — Complex SQL Demo
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/sql-analytics/{ticker}")
def get_sql_analytics(ticker: str, db: Session = Depends(get_db)):
    """
    Computes 14- and 50-day moving averages directly in the database with SQL
    window functions (AVG ... OVER), as a cross-check against the pandas-computed
    indicators stored by the analysis pipeline.
    """
    ticker = _ticker(ticker)
    query = text(
        "SELECT "
        "  date, "
        "  close_price, "
        "  AVG(close_price) OVER ("
        "    PARTITION BY ticker "
        "    ORDER BY date "
        "    ROWS BETWEEN 13 PRECEDING AND CURRENT ROW"
        "  ) AS sql_sma_14, "
        "  AVG(close_price) OVER ("
        "    PARTITION BY ticker "
        "    ORDER BY date "
        "    ROWS BETWEEN 49 PRECEDING AND CURRENT ROW"
        "  ) AS sql_sma_50 "
        "FROM market_data "
        "WHERE ticker = :ticker "
        "ORDER BY date DESC "
        "LIMIT 100"
    )
    result = db.execute(query, {"ticker": ticker}).fetchall()

    if not result:
        raise HTTPException(status_code=404, detail="No data found for the ticker.")

    return [
        {
            "date": row.date,
            "close_price": row.close_price,
            "sql_sma_14": round(row.sql_sma_14, 2) if row.sql_sma_14 else None,
            "sql_sma_50": round(row.sql_sma_50, 2) if row.sql_sma_50 else None,
        }
        for row in result
    ]


# ═══════════════════════════════════════════════════════════════════════════
# POST /api/backtest — Re-run the backtest
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/backtest", response_model=BacktestResult)
def run_backtest_endpoint(
    request: BacktestRequest,
    db: Session = Depends(get_db),
):
    """
    Re-run the backtest on the stored out-of-sample signals with a custom
    starting capital and/or cost model. Requires /api/analyze first; nothing
    is downloaded or retrained, so it returns in milliseconds.
    """
    try:
        result = backtest_from_db(request.ticker, request.capital, db, _cost_model(request.costs))
    except Exception as exc:
        _raise_http(exc, request.ticker, "Backtest")
    return BacktestResult(**result)


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/drift/{ticker} — Drift Monitoring
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/drift/{ticker}", response_model=DriftReport)
def get_drift(ticker: str, db: Session = Depends(get_db)):
    """
    Has the market moved away from the model's training regime? Compares the
    latest quarter of stored data with the two years before it: per-feature
    PSI/KS drift, the volatility regime, and the out-of-sample error rate.
    """
    ticker = _ticker(ticker)
    try:
        report = drift_from_db(ticker, db)
    except Exception as exc:
        _raise_http(exc, ticker, "Drift check")
    return DriftReport(**report)


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/health — Health Check
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/health", response_model=HealthResponse)
def health_check():
    """Simple health check for monitoring and load balancer probes."""
    return HealthResponse()
