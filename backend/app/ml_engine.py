"""
ML Engine — Core Quantitative Logic
====================================
Refactored from the standalone market_analyzer.py into a service layer
that the FastAPI routes call.  All functions return structured data
instead of printing to stdout.

Pipeline:
    1. fetch_and_store()   – downloads OHLCV via yfinance, engineers features,
                             stores everything in the MarketData table.
    2. train_and_predict() – walk-forward validation (app.walk_forward): every
                             quarter, re-tunes Logistic Regression, Random
                             Forest and SVM on past data only, selects one by
                             cross-validated balanced accuracy and predicts
                             the next quarter out of sample; then refits a
                             production model for the next-session signal.
    3. run_backtest()      – (app.backtest) ML strategy vs buy-and-hold and
                             always-long benchmarks, net of transaction costs.
    4. drift_report()      – (app.drift) feature, volatility-regime and
                             concept drift of the latest quarter.
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

from app import tracking
from app.backtest import CostModel, cost_sensitivity, run_backtest
from app.drift import drift_report
from app.estimators import build_models  # noqa: F401  (re-exported for callers and tests)
from app.models import MarketData, ModelPrediction
from app.tracking import ExperimentTracker
from app.walk_forward import WalkForwardConfig, fit_production_model, run_walk_forward

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Raw indicators: stored in the database and drawn on the dashboard charts.
INDICATOR_COLS = [
    "sma_14", "sma_50", "sma_200",
    "ema_14", "ema_20", "ema_50",
    "rsi_14",
    "macd_line", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "atr_14", "daily_return_vol",
    "volume_ratio", "volume_change",
]

# Model features (16): scale-free versions of the indicators, see
# add_model_features(). Rupee-denominated levels drift with the price, so a
# model trained at ₹1,500 would be extrapolating at ₹3,000.
FEATURE_COLS = [
    "dist_sma_14", "dist_sma_50", "dist_sma_200",
    "dist_ema_14", "dist_ema_20", "dist_ema_50",
    "rsi_14",
    "macd_line_pct", "macd_signal_pct", "macd_hist_pct",
    "bb_width", "bb_pct_b", "atr_pct", "daily_return_vol",
    "volume_ratio", "volume_change",
]

# Logical feature families — used by the feature-ablation experiments
# (experiments/feature_ablation.py) to measure each family's contribution.
FEATURE_GROUPS: Dict[str, List[str]] = {
    "trend": ["dist_sma_14", "dist_sma_50", "dist_sma_200", "dist_ema_14", "dist_ema_20", "dist_ema_50"],
    "momentum": ["rsi_14", "macd_line_pct", "macd_signal_pct", "macd_hist_pct"],
    "volatility": ["bb_width", "bb_pct_b", "atr_pct", "daily_return_vol"],
    "volume": ["volume_ratio", "volume_change"],
}

# Bump when the feature engineering logic changes, so tracked experiments
# from different feature definitions are never silently compared.
FEATURE_SET_VERSION = "v2-16-stationary"

TRAIN_RATIO = 0.80
N_PERMUTATIONS = 1000


class DataNotFoundError(LookupError):
    """No market data or predictions exist for the requested ticker (HTTP 404)."""


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL INDICATOR FUNCTIONS (unchanged from market_analyzer.py)
# ═══════════════════════════════════════════════════════════════════════════

def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average — unweighted mean of the last `window` prices."""
    return series.rolling(window=window).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """
    Exponential Moving Average — gives exponentially decreasing weights
    to older observations, more responsive to recent changes than SMA.
    """
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI) — momentum oscillator (0–100).

    RSI = 100 - (100 / (1 + RS))
    RS  = avg_gain / avg_loss  over `window` periods (Wilder method).

    RSI > 70 → overbought (sell signal)
    RSI < 30 → oversold   (buy signal)
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_bollinger_bands(
    series: pd.Series, window: int = 20, num_std: int = 2
) -> Tuple[pd.Series, pd.Series]:
    """
    Bollinger Bands — volatility indicator.
        Upper = SMA(window) + num_std × σ(window)
        Lower = SMA(window) - num_std × σ(window)
    """
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return sma + num_std * std, sma - num_std * std


def compute_macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD, Signal, and Histogram."""
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (Volatility)."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the 16 scale-free model features from the raw indicators.

    Moving averages, Bollinger bands, MACD and ATR are measured in rupees, so
    their distribution moves with the price level (they are non-stationary).
    Each is re-expressed relative to the current close: a distance from the
    average, a band width or position, or a share of price. RSI, return
    volatility and the volume ratios are already scale-free.
    """
    close = df["close_price"]
    for col in ("sma_14", "sma_50", "sma_200", "ema_14", "ema_20", "ema_50"):
        df[f"dist_{col}"] = close / df[col] - 1
    for col in ("macd_line", "macd_signal", "macd_hist"):
        df[f"{col}_pct"] = df[col] / close
    band = df["bb_upper"] - df["bb_lower"]
    df["bb_width"] = band / close
    df["bb_pct_b"] = (close - df["bb_lower"]) / band
    df["atr_pct"] = df["atr_14"] / close
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the 16 raw indicators, the 16 model features and the binary target.

    Every feature at row t uses prices up to t's close only (causal). The
    target is whether the *next* session closes above its open, exactly the
    trade the backtest makes (buy at the T+1 open, sell at the T+1 close).
    The latest row has no next session yet, so its target is NaN: the row is
    kept for live inference but never used for training or evaluation.
    """
    close = df["close_price"]

    df["sma_14"] = compute_sma(close, 14)
    df["sma_50"] = compute_sma(close, 50)
    df["sma_200"] = compute_sma(close, 200)

    df["ema_14"] = compute_ema(close, 14)
    df["ema_20"] = compute_ema(close, 20)
    df["ema_50"] = compute_ema(close, 50)

    df["rsi_14"] = compute_rsi(close, 14)
    df["macd_line"], df["macd_signal"], df["macd_hist"] = compute_macd(close)

    df["bb_upper"], df["bb_lower"] = compute_bollinger_bands(close, 20, 2)
    df["atr_14"] = compute_atr(df["high_price"], df["low_price"], close, 14)
    df["daily_return_vol"] = close.pct_change().rolling(20).std()

    vol_sma_20 = compute_sma(df["volume"], 20)
    df["volume_ratio"] = df["volume"] / vol_sma_20
    df["volume_change"] = df["volume"].pct_change()

    add_model_features(df)

    # Target: 1 (Buy) if the next session's intraday return is positive.
    # Prediction is made after day T's close → trade executed on day T+1.
    next_open, next_close = df["open_price"].shift(-1), close.shift(-1)
    df["target"] = np.where(next_close.notna(), (next_close > next_open).astype(float), np.nan)

    # Replace infinite values (divide-by-zero in volume ratio, RSI or band
    # position) with NaN, then drop the indicator warm-up rows.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=INDICATOR_COLS + FEATURE_COLS, inplace=True)

    return df


def labelled_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose next-session outcome is known (every row except the latest)."""
    out = df[df["target"].notna()].copy()
    out["target"] = out["target"].astype(int)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATA ACQUISITION & STORAGE
# ═══════════════════════════════════════════════════════════════════════════

def download_ohlcv(ticker: str, years: int) -> pd.DataFrame:
    """Download and clean daily OHLCV data from Yahoo Finance (no features)."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)

    logger.info("Downloading %dY data for %s", years, ticker)
    raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if raw is None or raw.empty:
        raise DataNotFoundError(
            f"No price data found for '{ticker}'. Check the symbol "
            "(NSE tickers end in .NS, BSE tickers in .BO)."
        )

    # Flatten multi-level columns if present.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Standardise column names to match our ORM fields.
    raw = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    raw.columns = ["open_price", "high_price", "low_price", "close_price", "volume"]
    raw.index = pd.to_datetime(raw.index)
    raw.sort_index(inplace=True)

    # Handle missing data.
    raw.ffill(inplace=True)
    raw.bfill(inplace=True)
    raw.dropna(inplace=True)
    return raw


def fetch_and_store(
    ticker: str,
    years: int,
    db: Session,
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance, engineer features, and
    upsert all rows into the MarketData table.

    Returns the cleaned DataFrame (with features) for downstream use.
    """
    df = engineer_features(download_ohlcv(ticker, years))

    # ── Bulk upsert into database ─────────────────────────────────────────
    # Delete existing rows for this ticker, then insert fresh data.
    # This is simpler and safer than per-row upsert for a batch pipeline.
    db.query(MarketData).filter(MarketData.ticker == ticker).delete()

    records = [
        MarketData(
            ticker=ticker,
            date=idx.date(),
            open_price=float(row["open_price"]),
            high_price=float(row["high_price"]),
            low_price=float(row["low_price"]),
            close_price=float(row["close_price"]),
            volume=float(row["volume"]),
            **{col: float(row[col]) for col in INDICATOR_COLS},
            # The latest row's outcome is not known yet.
            target=None if pd.isna(row["target"]) else int(row["target"]),
        )
        for idx, row in df.iterrows()
    ]
    db.bulk_save_objects(records)
    db.commit()

    logger.info("Stored %d rows for %s in database", len(records), ticker)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — INPUT VALIDATION & DATA PREPARATION
# (the model registry lives in app/estimators.py)
# ═══════════════════════════════════════════════════════════════════════════

def validate_training_data(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> None:
    """
    Fail-fast validation of a DataFrame before model training.

    Checks:
        1. DataFrame is not empty.
        2. All feature columns (default: the 16 FEATURE_COLS) are present.
        3. The 'target' column exists.
        4. No NaN or infinite values in features + target.
        5. Target values are binary (0 or 1).
        6. Minimum row count for meaningful training (≥ 50).

    Raises:
        ValueError: With a descriptive message if any check fails.
    """
    feature_cols = feature_cols or FEATURE_COLS
    if df.empty:
        raise ValueError("Training DataFrame is empty — no rows to train on.")

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing {len(missing_features)} feature column(s): {missing_features}"
        )

    if "target" not in df.columns:
        raise ValueError("Missing 'target' column in DataFrame.")

    check_cols = feature_cols + ["target"]
    nan_counts = df[check_cols].isna().sum()
    cols_with_nans = nan_counts[nan_counts > 0]
    if not cols_with_nans.empty:
        raise ValueError(
            f"NaN values found in {len(cols_with_nans)} column(s): "
            f"{dict(cols_with_nans)}"
        )

    # Check for ±inf in numeric feature columns.
    inf_mask = np.isinf(df[feature_cols].select_dtypes(include=[np.number]))
    cols_with_infs = inf_mask.any()
    cols_with_infs = cols_with_infs[cols_with_infs].index.tolist()
    if cols_with_infs:
        raise ValueError(
            f"Infinite values found in column(s): {cols_with_infs}"
        )

    unique_targets = set(df["target"].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"Target column must be binary (0/1), found values: {unique_targets}"
        )

    MIN_ROWS = 50
    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Need at least {MIN_ROWS} rows for training, got {len(df)}."
        )


def prepare_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Prepare chronological train/test split with defensive data cleaning.

    Steps:
        1. Verify the index is monotonically increasing (chronological order).
        2. Replace any residual ±inf with NaN, then drop affected rows.
        3. Split at ``train_ratio`` boundary.
        4. Extract X (features) and y (target) arrays.

    Returns:
        Dict with keys: X_train, X_test, y_train, y_test, train_df, test_df.

    Raises:
        ValueError: If the index is not in chronological order.
    """
    feature_cols = feature_cols or FEATURE_COLS

    # ── Chronological ordering guard ──────────────────────────────────────
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "DataFrame index is not in chronological order. "
            "Sort by date before calling this function to prevent "
            "future data from leaking into the training set."
        )

    # ── Defensive NaN / inf cleanup ───────────────────────────────────────
    clean = df.copy()
    clean = clean.replace([np.inf, -np.inf], np.nan)
    rows_before = len(clean)
    clean = clean.dropna(subset=feature_cols + ["target"])
    rows_dropped = rows_before - len(clean)
    if rows_dropped > 0:
        logger.warning(
            "Dropped %d rows with NaN/inf during train/test preparation",
            rows_dropped,
        )

    # ── Time-series split ─────────────────────────────────────────────────
    split_idx = int(len(clean) * train_ratio)
    train_df = clean.iloc[:split_idx].copy()
    test_df = clean.iloc[split_idx:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    logger.info(
        "Train/test split: %d train rows, %d test rows (ratio=%.2f)",
        len(train_df), len(test_df), train_ratio,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — WALK-FORWARD TRAINING & PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

METRIC_KEYS = ("accuracy", "precision", "recall", "balanced_accuracy", "roc_auc")


def _prediction(day: pd.Timestamp, p_up: float, threshold: float) -> Dict[str, Any]:
    """Signal plus confidence (the probability of the predicted class)."""
    signal = int(p_up > threshold)
    return {"date": day.date(), "signal": signal, "confidence": round(p_up if signal else 1 - p_up, 4)}


def _rounded(metrics: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Optional[float]]:
    return {k: None if metrics.get(k) is None else round(float(metrics[k]), 4) for k in keys}


def train_and_predict(
    df: pd.DataFrame,
    ticker: str,
    db: Optional[Session] = None,
    feature_cols: Optional[List[str]] = None,
    tracker: Optional[ExperimentTracker] = None,
    wf_config: Optional[WalkForwardConfig] = None,
) -> Dict[str, Any]:
    """
    Walk-forward training and out-of-sample prediction.

    1. Validate the labelled rows (every row except the latest).
    2. Walk forward (see app.walk_forward): each quarter, tune all three
       families on past data only, select one by cross-validated balanced
       accuracy, and predict the next quarter.
    3. Refit the most recently selected configuration on all labelled rows
       (the production model) and predict the next session from the latest
       row, whose outcome is not known yet.

    Args:
        df: Engineered DataFrame (see ``engineer_features``).
        ticker: Ticker symbol, used for persistence and tracking.
        db: Optional session. When given, the out-of-sample predictions and
            the live signal are stored in the model_predictions table.
        feature_cols: Feature subset to train on (default: all 16).
        tracker: Optional MLflow tracker (folds, candidates, model, datasets).
        wf_config: Walk-forward settings (default: from app settings).

    Returns a dict with OOS metrics, the per-family comparison, fold-averaged
    feature importances, OOS predictions, the signal series to backtest
    (``signals``: every OOS day plus the latest row) and the walk-forward
    summary.
    """
    feature_cols = list(feature_cols or FEATURE_COLS)
    tracker = tracker or tracking.NULL_TRACKER
    wf_config = wf_config or WalkForwardConfig.from_settings()

    # ── Validate, then walk forward on labelled rows only ────────────────
    labelled = labelled_rows(df)
    validate_training_data(labelled, feature_cols)
    wf = run_walk_forward(labelled, feature_cols, wf_config)
    oos = wf.oos

    # ── Production model + live next-session signal ──────────────────────
    production = fit_production_model(labelled, feature_cols, wf)
    live = df[df["target"].isna()].dropna(subset=feature_cols).tail(1)
    latest = None
    if not live.empty:
        p_up = float(production.predict_proba(live[feature_cols])[:, 1][0])
        latest = _prediction(live.index[-1], p_up, wf_config.threshold)

    predictions = [_prediction(day, p, wf_config.threshold) for day, p in zip(oos.index, oos["proba"], strict=True)]

    # Signals to backtest: every OOS day, plus the latest row. That row trades
    # on yesterday's OOS signal; its own signal is for the next session.
    signals = oos["signal"].astype(int)
    if latest is not None:
        signals = pd.concat([signals, pd.Series([latest["signal"]], index=live.index)])

    metrics = {
        **_rounded(wf.metrics, METRIC_KEYS + ("baseline_accuracy",)),
        "p_value_vs_baseline": round(float(wf.metrics["p_value_vs_baseline"]), 6),
    }
    all_model_metrics = {name: _rounded(m, METRIC_KEYS) for name, m in wf.family_metrics.items()}

    # Random Forest importances, averaged over folds (± std across folds).
    mean_imp, std_imp = wf.importances.mean(), wf.importances.std(ddof=0)
    importances = [
        {"feature": f, "importance": float(mean_imp[f]), "std": float(std_imp[f])}
        for f in mean_imp.sort_values(ascending=False).index
    ]

    # ── Experiment tracking ───────────────────────────────────────────────
    summary = wf.summary()
    tracker.log_params({
        "ticker": ticker,
        "feature_set_version": FEATURE_SET_VERSION,
        "n_features": len(feature_cols),
        "features": ",".join(feature_cols),
        "validation": "walk_forward",
        "wf_initial_train_size": wf.initial_train_size,
        "wf_step": wf_config.step,
        "wf_retune_every": wf_config.retune_every,
        "wf_gap": wf_config.gap,
        "wf_n_folds": len(wf.folds),
        "selection_metric": wf_config.scoring,
        "decision_threshold": wf_config.threshold,
        "random_state": wf_config.random_state,
        "data_start": labelled.index[0].strftime("%Y-%m-%d"),
        "oos_start": summary["oos_start"],
        "oos_end": summary["oos_end"],
        "n_labelled": len(labelled),
        "n_oos": len(oos),
    })
    tracker.log_metrics({
        **{f"oos_{k}": metrics[k] for k in METRIC_KEYS},
        "oos_positive_rate": wf.metrics["positive_rate"],
        "train_positive_rate": float(labelled["target"].mean()),
        "baseline_majority_accuracy": wf.metrics["baseline_accuracy"],
        "accuracy_lift_over_baseline": wf.metrics["accuracy_lift"],
        "p_value_vs_baseline": wf.metrics["p_value_vs_baseline"],
    })
    tracker.log_dataset(labelled, feature_cols, name=f"{ticker}-train", context="training")
    tracker.log_dataset(labelled.loc[oos.index], feature_cols, name=f"{ticker}-oos", context="testing")
    tracker.log_walk_forward(wf.folds)
    # The candidates considered at the most recent tuning: the deployed model's selection.
    for name, search in wf.last_searches.items():
        tracker.log_candidate(name, search, wf.family_metrics[name], is_best=(name == wf.selected_model))
    tracker.set_tags({"best_model": wf.selected_model})
    tracker.log_feature_importances(importances)
    tracker.log_dict(all_model_metrics, "model_comparison.json")
    tracker.log_model(production, labelled[feature_cols], ticker=ticker, model_name=wf.selected_model)

    # ── Persist predictions ───────────────────────────────────────────────
    if db is not None:
        records = [
            ModelPrediction(
                ticker=ticker,
                prediction_date=p["date"],
                predicted_signal=p["signal"],
                confidence_score=p["confidence"],
                actual_outcome=int(outcome),
            )
            for p, outcome in zip(predictions, oos["target"], strict=True)
        ]
        if latest is not None:
            records.append(ModelPrediction(
                ticker=ticker,
                prediction_date=latest["date"],
                predicted_signal=latest["signal"],
                confidence_score=latest["confidence"],
                actual_outcome=None,
            ))
        db.add_all(records)
        db.commit()

    logger.info(
        "Walk-forward %s: %d folds, %d OOS days, balanced acc=%.4f, acc=%.4f (baseline %.4f), selected %s",
        ticker, len(wf.folds), len(oos), metrics["balanced_accuracy"], metrics["accuracy"],
        metrics["baseline_accuracy"], wf.selected_model,
    )

    return {
        "train_size": wf.initial_train_size,
        "test_size": len(oos),
        "metrics": metrics,
        "feature_importances": importances,
        "predictions": predictions,
        "latest_prediction": latest or (predictions[-1] if predictions else None),
        "test_df": labelled.loc[oos.index],
        "y_pred": oos["signal"].to_numpy(),
        "signals": signals,
        "best_model_name": wf.selected_model,
        "all_model_metrics": all_model_metrics,
        "validation": summary,
        "walk_forward": wf,
        "production_model": production,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4 — BACKTESTING (engine: app.backtest)
# ═══════════════════════════════════════════════════════════════════════════

def backtest_signals(
    prices: pd.DataFrame,
    signals: Iterable[int],
    capital: float,
    costs: Optional[CostModel] = None,
) -> Dict[str, Any]:
    """Net-of-cost backtest, permutation test and cost sensitivity for one signal series."""
    signals = [int(s) for s in signals]
    result = run_backtest(prices, signals, capital, costs=costs or CostModel(), n_permutations=N_PERMUTATIONS)
    result["cost_sensitivity"] = cost_sensitivity(prices, signals)
    return result


def backtest_from_db(
    ticker: str,
    capital: float,
    db: Session,
    costs: Optional[CostModel] = None,
) -> Dict[str, Any]:
    """
    Re-run the backtest on the stored out-of-sample signals with another
    capital or cost model. No download and no retraining: just the simulation.
    """
    preds = (
        db.query(ModelPrediction)
        .filter(ModelPrediction.ticker == ticker)
        .order_by(ModelPrediction.prediction_date)
        .all()
    )
    if not preds:
        raise DataNotFoundError(f"No predictions found for '{ticker}'. Run /api/analyze first.")
    rows = (
        db.query(MarketData)
        .filter(MarketData.ticker == ticker, MarketData.date >= preds[0].prediction_date)
        .order_by(MarketData.date)
        .all()
    )
    prices = pd.DataFrame(
        {"open_price": [r.open_price for r in rows], "close_price": [r.close_price for r in rows]},
        index=pd.to_datetime([r.date for r in rows]),
    )
    by_date = {pd.Timestamp(p.prediction_date): p.predicted_signal for p in preds}
    signals = [int(by_date.get(day, 0)) for day in prices.index]
    return backtest_signals(prices, signals, capital, costs)


def drift_from_db(ticker: str, db: Session) -> Dict[str, Any]:
    """Drift report from stored market data and predictions (no download)."""
    rows = db.query(MarketData).filter(MarketData.ticker == ticker).order_by(MarketData.date).all()
    if not rows:
        raise DataNotFoundError(f"No data found for '{ticker}'. Run /api/analyze first.")
    df = pd.DataFrame(
        [{"close_price": r.close_price, **{c: getattr(r, c) for c in INDICATOR_COLS}} for r in rows],
        index=pd.to_datetime([r.date for r in rows]),
    )
    add_model_features(df)
    preds = (
        db.query(ModelPrediction)
        .filter(ModelPrediction.ticker == ticker)
        .order_by(ModelPrediction.prediction_date)
        .all()
    )
    oos = None
    if preds:
        oos = pd.DataFrame(
            {"signal": [p.predicted_signal for p in preds], "target": [p.actual_outcome for p in preds]},
            index=pd.to_datetime([p.prediction_date for p in preds]),
            dtype=float,
        )
    return drift_report(df, FEATURE_COLS, oos=oos)


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — runs the full pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_full_analysis(
    ticker: str,
    years: int,
    capital: float,
    db: Session,
    costs: Optional[CostModel] = None,
) -> Dict[str, Any]:
    """
    Execute the complete pipeline: fetch → features → walk-forward training →
    net-of-cost backtest → drift report.

    This is the main entry point called by the /api/analyze endpoint.
    Returns a fully structured dict ready for AnalysisResponse serialisation.
    """
    costs = costs or CostModel()
    # Every analysis is one MLflow run; it is a no-op when tracking is off.
    with tracking.start_run(
        run_name=f"{ticker}-analysis",
        tags={"ticker": ticker, "source": "api"},
    ) as tracker:
        tracker.log_params({
            "lookback_years": years,
            "starting_capital": capital,
            **{f"cost_{k}": v for k, v in costs.as_dict().items()},
        })

        # Phase 1+2: Fetch data, engineer features, store in DB.
        df = fetch_and_store(ticker, years, db)

        # Clear old predictions for this ticker.
        db.query(ModelPrediction).filter(ModelPrediction.ticker == ticker).delete()
        db.commit()

        # Phase 3: Walk-forward training and out-of-sample predictions.
        ml_results = train_and_predict(df, ticker, db, tracker=tracker)

        # Phase 4: Backtest the out-of-sample signals net of costs.
        signals = ml_results["signals"]
        backtest_results = backtest_signals(df.loc[signals.index], signals, capital, costs)
        tracker.log_backtest(backtest_results)

        # Phase 5: Has the market moved away from the training regime?
        drift = drift_report(df, FEATURE_COLS, oos=ml_results["walk_forward"].oos)
        tracker.log_drift(drift)

    if drift["status"] in ("drift", "warning"):
        logger.warning("Drift %s for %s: %s", drift["status"], ticker, " ".join(drift["alerts"]))

    # ── Assemble market_data for the frontend ─────────────────────────────
    market_data = [
        {
            "date": idx.date(),
            "open": round(float(row["open_price"]), 2),
            "high": round(float(row["high_price"]), 2),
            "low": round(float(row["low_price"]), 2),
            "close": round(float(row["close_price"]), 2),
            "volume": float(row["volume"]),
            **{
                col: round(float(row[col]), 4 if col == "daily_return_vol" else 2)
                for col in INDICATOR_COLS
            },
            "target": None if pd.isna(row["target"]) else int(row["target"]),
        }
        for idx, row in df.iterrows()
    ]

    # Date range metadata.
    date_range = {
        "start": df.index[0].strftime("%Y-%m-%d"),
        "end": df.index[-1].strftime("%Y-%m-%d"),
    }

    return {
        "ticker": ticker,
        "data_points": len(df),
        "train_size": ml_results["train_size"],
        "test_size": ml_results["test_size"],
        "date_range": date_range,
        "market_data": market_data,
        "predictions": ml_results["predictions"],
        "latest_prediction": ml_results["latest_prediction"],
        "metrics": ml_results["metrics"],
        "feature_importances": ml_results["feature_importances"],
        "backtest": backtest_results,
        "best_model": ml_results["best_model_name"],
        "all_model_metrics": ml_results["all_model_metrics"],
        "validation": ml_results["validation"],
        "drift": drift,
        "mlflow_run_id": tracker.run_id,
    }
