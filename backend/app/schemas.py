"""
Pydantic Schemas (Request / Response)
=====================================
These models define the API contract between the FastAPI backend and the
Next.js frontend.  Pydantic v2 is used for serialisation and validation.
"""

import re
from datetime import date
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Yahoo Finance symbols: RELIANCE.NS, BAJAJ-AUTO.NS, M&M.NS, ^NSEI, EURUSD=X …
TICKER_PATTERN = re.compile(r"[A-Z0-9^][A-Z0-9.\-=&^]{0,24}")


def normalise_ticker(value: str) -> str:
    """Upper-case and validate a Yahoo Finance symbol."""
    ticker = value.strip().upper()
    if not TICKER_PATTERN.fullmatch(ticker):
        raise ValueError(
            f"Invalid ticker symbol '{value}'. Use a Yahoo Finance symbol such as RELIANCE.NS."
        )
    return ticker


# ═══════════════════════════════════════════════════════════════════════════
# REQUEST SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════

class CostAssumptions(BaseModel):
    """Transaction costs in basis points of traded notional (1 bp = 0.01%)."""
    commission_bps: float = Field(3.0, ge=0, le=100, description="Per side: brokerage, exchange/SEBI fees, GST")
    slippage_bps: float = Field(2.0, ge=0, le=100, description="Per side: spread and market impact")
    sell_tax_bps: float = Field(2.5, ge=0, le=100, description="STT on the sell leg")
    buy_tax_bps: float = Field(0.3, ge=0, le=100, description="Stamp duty on the buy leg")
    round_trip_bps: Optional[float] = Field(None, description="Total per round trip (filled in responses)")


class AnalyzeRequest(BaseModel):
    """Request body for the /api/analyze endpoint."""
    ticker: str = Field(
        ...,
        examples=["RELIANCE.NS"],
        description="Yahoo Finance ticker symbol",
    )
    years: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Years of history to fetch (≥ 2: walk-forward needs a 200-day warm-up plus training data)",
    )
    costs: Optional[CostAssumptions] = Field(
        default=None, description="Override the default NSE intraday cost model",
    )

    @field_validator("ticker")
    @classmethod
    def _valid_ticker(cls, value: str) -> str:
        return normalise_ticker(value)


class BacktestRequest(BaseModel):
    """Request body for the /api/backtest endpoint."""
    ticker: str = Field(..., examples=["RELIANCE.NS"])
    capital: float = Field(
        default=100_000.0,
        gt=0,
        description="Starting capital in INR",
    )
    costs: Optional[CostAssumptions] = Field(
        default=None, description="Override the default NSE intraday cost model",
    )

    @field_validator("ticker")
    @classmethod
    def _valid_ticker(cls, value: str) -> str:
        return normalise_ticker(value)


# ═══════════════════════════════════════════════════════════════════════════
# RESPONSE SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════

class MarketDataPoint(BaseModel):
    """A single day's OHLCV + all 16 engineered indicators."""
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    sma_14: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_14: Optional[float] = None
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    rsi_14: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    atr_14: Optional[float] = None
    daily_return_vol: Optional[float] = None
    volume_ratio: Optional[float] = None
    volume_change: Optional[float] = None
    target: Optional[int] = None


class PredictionDetail(BaseModel):
    """ML prediction made at the close of `date` for the next session."""
    date: date
    signal: int = Field(description="1 = Buy, 0 = Sell/Hold")
    confidence: float = Field(description="Probability of the predicted class (0.5 – 1.0)")


class FeatureImportance(BaseModel):
    """Random Forest importance of one feature, averaged over walk-forward folds."""
    feature: str
    importance: float
    std: Optional[float] = Field(None, description="Standard deviation across folds")


class ClassificationMetrics(BaseModel):
    """Out-of-sample classification metrics of the walk-forward strategy."""
    accuracy: float
    precision: float
    recall: float
    balanced_accuracy: Optional[float] = Field(None, description="Mean per-class recall; chance = 0.5")
    roc_auc: Optional[float] = None
    baseline_accuracy: Optional[float] = Field(
        None, description="Always predicting each training window's majority class",
    )
    p_value_vs_baseline: Optional[float] = Field(
        None, description="Exact one-sided McNemar test of the model against that baseline",
    )


class ModelMetrics(BaseModel):
    """Out-of-sample metrics for one model family in the comparison."""
    accuracy: float
    precision: float
    recall: float
    balanced_accuracy: Optional[float] = None
    roc_auc: Optional[float] = None


class DailyCumulative(BaseModel):
    """A single day's cumulative return data point for charting."""
    date: str
    ml_cumulative: float
    bh_cumulative: float
    ml_gross_cumulative: Optional[float] = None


class PerformanceStats(BaseModel):
    """Risk/return statistics of a daily return series."""
    total_return_pct: float
    cagr_pct: float
    volatility_pct: float
    sharpe_ratio: float
    sharpe_ci_low: float = Field(description="95% confidence interval (Lo, 2002)")
    sharpe_ci_high: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    n_days: int


class TradeStats(BaseModel):
    """Per-trade statistics. Each long day is one intraday round trip."""
    n_trades: int
    exposure_pct: float
    trades_per_year: float
    hit_rate: Optional[float] = None
    gross_hit_rate: Optional[float] = None
    avg_gross_edge_bps: Optional[float] = Field(
        None, description="Average gross return per trade, i.e. the break-even round-trip cost",
    )
    avg_net_trade_bps: Optional[float] = None
    profit_factor: Optional[float] = None


class CostSensitivityPoint(BaseModel):
    round_trip_bps: float
    net_return_pct: float
    sharpe_ratio: float


class BacktestResult(BaseModel):
    """Results of the backtesting simulation (headline fields are net of costs)."""
    starting_capital: float
    ml_final_value: float
    bh_final_value: float
    ml_return_pct: float
    bh_return_pct: float
    sharpe_ratio: float
    daily_cumulative: List[DailyCumulative] = Field(
        description="List of daily cumulative return data points for charting",
    )
    gross_return_pct: Optional[float] = None
    gross_sharpe_ratio: Optional[float] = None
    cost_drag_pct: Optional[float] = None
    costs: Optional[CostAssumptions] = None
    strategy_stats: Optional[PerformanceStats] = None
    buy_hold_stats: Optional[PerformanceStats] = None
    always_long_stats: Optional[PerformanceStats] = None
    trade_stats: Optional[TradeStats] = None
    permutation_p_value: Optional[float] = Field(
        None, description="P(random timing with the same number of trades beats the strategy's Sharpe)",
    )
    cost_sensitivity: List[CostSensitivityPoint] = Field(default_factory=list)


class FoldSummary(BaseModel):
    """One walk-forward fold: trained on the past, evaluated on the next block."""
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    model: str
    retuned: bool
    cv_score: Optional[float] = None
    accuracy: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    roc_auc: Optional[float] = None
    positive_rate: Optional[float] = None


class ValidationSummary(BaseModel):
    """How the out-of-sample predictions were produced."""
    method: str = "walk_forward"
    window: str
    n_folds: int
    initial_train_size: int
    step: int
    retune_every: int
    gap: int
    selection_metric: str
    oos_start: str
    oos_end: str
    n_oos: int
    model_selection_counts: Dict[str, int]
    folds: List[FoldSummary]


class FeatureDrift(BaseModel):
    feature: str
    psi: float
    psi_p_value: Optional[float] = Field(None, description="Empirical p-value vs historical windows")
    ks_statistic: float
    ks_p_value: float = Field(description="Naive iid KS p-value, for comparison only")
    drifted: bool


class VolatilityDrift(BaseModel):
    current_vol_pct: float
    reference_vol_pct: float
    ratio: Optional[float] = None
    percentile: float = Field(description="Percentile among historical same-length windows")
    levene_p_value: Optional[float] = None
    regime: Literal["low", "normal", "high"]


class ConceptDrift(BaseModel):
    status: Literal["stable", "degraded", "insufficient_data"]
    window: int
    n_oos: int
    recent_error_rate: Optional[float] = None
    reference_error_rate: Optional[float] = None
    error_p_value: Optional[float] = None
    recent_up_rate: Optional[float] = None
    reference_up_rate: Optional[float] = None
    label_shift_p_value: Optional[float] = None


class DriftReport(BaseModel):
    """Is the latest quarter still like the regime the model was trained on?"""
    status: Literal["ok", "warning", "drift", "insufficient_data"]
    alerts: List[str] = Field(default_factory=list)
    reference_start: Optional[str] = None
    reference_end: Optional[str] = None
    current_start: Optional[str] = None
    current_end: Optional[str] = None
    n_reference: Optional[int] = None
    n_current: Optional[int] = None
    share_drifted: Optional[float] = None
    drifted_features: List[str] = Field(default_factory=list)
    features: List[FeatureDrift] = Field(default_factory=list)
    volatility: Optional[VolatilityDrift] = None
    concept: Optional[ConceptDrift] = None
    thresholds: Optional[Dict[str, Any]] = None


class AnalysisResponse(BaseModel):
    """
    Complete response from the /api/analyze endpoint.
    Bundles market data, predictions, metrics, backtest and drift results
    into a single payload for the frontend dashboard.
    """
    ticker: str
    data_points: int
    train_size: int = Field(description="Rows in the first walk-forward training window")
    test_size: int = Field(description="Out-of-sample rows across all folds")
    date_range: Dict[str, str]  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    market_data: List[MarketDataPoint]
    predictions: List[PredictionDetail]
    latest_prediction: Optional[PredictionDetail] = None
    metrics: ClassificationMetrics
    feature_importances: List[FeatureImportance]
    backtest: BacktestResult
    best_model: str = Field(
        description="Model family selected at the most recent walk-forward tuning",
    )
    all_model_metrics: Dict[str, ModelMetrics] = Field(
        default_factory=dict,
        description="Out-of-sample metrics for each of the 3 compared models",
    )
    validation: Optional[ValidationSummary] = None
    drift: Optional[DriftReport] = None
    mlflow_run_id: Optional[str] = Field(
        default=None,
        description="MLflow run that recorded this analysis (null when tracking is off)",
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    service: str = "algorithmic-market-analyzer"
