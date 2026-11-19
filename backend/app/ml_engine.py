"""
ML Engine — Core Quantitative Logic
====================================
Refactored from the standalone market_analyzer.py into a service layer
that the FastAPI routes call.  All functions return structured data
instead of printing to stdout.

Pipeline:
    1. fetch_and_store()  – downloads OHLCV via yfinance, engineers features,
                            bulk-upserts into the MarketData table.
    2. train_and_predict() – loads data from DB, trains & tunes 3 classifiers
                             (Random Forest, Logistic Regression, SVM) using
                             TimeSeriesSplit for cross-validation, selects
                             the best, returns predictions + metrics + importances.
    3. run_backtest()      – simulates ML strategy vs Buy-and-Hold.
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sqlalchemy.orm import Session

from app import tracking
from app.config import settings
from app.models import MarketData, ModelPrediction
from app.tracking import ExperimentTracker

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Features used by the classifiers (16 total).
FEATURE_COLS = [
    "sma_14", "sma_50", "sma_200",
    "ema_14", "ema_20", "ema_50",
    "rsi_14",
    "macd_line", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "atr_14", "daily_return_vol",
    "volume_ratio", "volume_change",
]

# Logical feature families — used by the feature-ablation experiments
# (experiments/feature_ablation.py) to measure each family's contribution.
FEATURE_GROUPS: Dict[str, List[str]] = {
    "trend": ["sma_14", "sma_50", "sma_200", "ema_14", "ema_20", "ema_50"],
    "momentum": ["rsi_14", "macd_line", "macd_signal", "macd_hist"],
    "volatility": ["bb_upper", "bb_lower", "atr_14", "daily_return_vol"],
    "volume": ["volume_ratio", "volume_change"],
}

# Bump when the feature engineering logic changes, so tracked experiments
# from different feature definitions are never silently compared.
FEATURE_SET_VERSION = "v1-16-technical"

TRAIN_RATIO = 0.80
CV_SPLITS = 3
RANDOM_STATE = 42


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

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all 16 statistical indicators and the binary target variable."""
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

    # Target: 1 (Buy) if tomorrow's intraday return is positive, else 0.
    # This aligns the target with what the backtest actually captures:
    # buy at next-day Open, sell at next-day Close.
    # Prediction is made after day T's close → trade executed on day T+1.
    df["target"] = (close.shift(-1) > df["open_price"].shift(-1)).astype(int)

    # Replace infinite values (caused by divide by zero in volume ratio or RSI) with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATA ACQUISITION & STORAGE
# ═══════════════════════════════════════════════════════════════════════════

def download_ohlcv(ticker: str, years: int) -> pd.DataFrame:
    """Download and clean daily OHLCV data from Yahoo Finance (no features)."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)

    logger.info("Downloading %dY data for %s", years, ticker)
    raw = yf.download(ticker, start=start_date, end=end_date, progress=False)

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

    records = []
    for idx, row in df.iterrows():
        records.append(
            MarketData(
                ticker=ticker,
                date=idx.date(),
                open_price=float(row["open_price"]),
                high_price=float(row["high_price"]),
                low_price=float(row["low_price"]),
                close_price=float(row["close_price"]),
                volume=float(row["volume"]),
                sma_14=float(row["sma_14"]),
                sma_50=float(row["sma_50"]),
                sma_200=float(row["sma_200"]),
                ema_14=float(row["ema_14"]),
                ema_20=float(row["ema_20"]),
                ema_50=float(row["ema_50"]),
                rsi_14=float(row["rsi_14"]),
                macd_line=float(row["macd_line"]),
                macd_signal=float(row["macd_signal"]),
                macd_hist=float(row["macd_hist"]),
                bb_upper=float(row["bb_upper"]),
                bb_lower=float(row["bb_lower"]),
                atr_14=float(row["atr_14"]),
                daily_return_vol=float(row["daily_return_vol"]),
                volume_ratio=float(row["volume_ratio"]),
                volume_change=float(row["volume_change"]),
                target=int(row["target"]),
            )
        )
    db.bulk_save_objects(records)
    db.commit()

    logger.info("Stored %d rows for %s in database", len(records), ticker)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — MODEL REGISTRY, VALIDATION & DATA PREPARATION
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


def build_models() -> Dict[str, Pipeline]:
    """
    Model registry — returns the three candidate pipelines with sensible
    default hyperparameters.

    Each pipeline encapsulates its own preprocessing (StandardScaler for
    models that require it), so all pipelines accept identical raw feature
    matrices.  No hyperparameter tuning is applied here; that is the
    responsibility of the caller (e.g. ``train_and_predict`` wraps these
    in ``RandomizedSearchCV``).

    Returns:
        Dict mapping model name → sklearn Pipeline.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=RANDOM_STATE,
            )),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                probability=True, kernel="rbf", random_state=RANDOM_STATE,
            )),
        ]),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — MODEL TRAINING & PREDICTION (uses Stage 1 components)
# ═══════════════════════════════════════════════════════════════════════════

# Hyperparameter search spaces (clf__-prefixed for Pipeline compatibility).
PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "Logistic Regression": {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
    },
    "Random Forest": {
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [5, 10, None],
        "clf__min_samples_split": [2, 5],
    },
    "SVM": {
        "clf__C": [0.1, 1.0],
        "clf__gamma": ["scale", "auto"],
    },
}
# Kept small so API responses stay fast on free-tier hosting.
N_ITER: Dict[str, int] = {
    "Logistic Regression": 2,
    "Random Forest": 3,
    "SVM": 2,
}


def train_and_predict(
    df: pd.DataFrame,
    ticker: str,
    db: Optional[Session] = None,
    feature_cols: Optional[List[str]] = None,
    tracker: Optional[ExperimentTracker] = None,
) -> Dict[str, Any]:
    """
    Train and tune 3 classifiers (Random Forest, Logistic Regression, SVM)
    on the first 80% of data (time-series split), predict on the last 20%,
    and return structured results.

    Cross-validation uses sklearn's TimeSeriesSplit to prevent future
    data leakage during hyperparameter tuning.

    Args:
        df: Engineered feature DataFrame (see ``engineer_features``).
        ticker: Ticker symbol, used for persistence and tracking.
        db: Optional session. When given, predictions are stored in the
            model_predictions table; offline experiments pass ``None``.
        feature_cols: Feature subset to train on (default: all 16).
        tracker: Optional MLflow tracker. Each candidate model is logged
            as a nested run; the selected model is logged as an artifact.

    Returns a dict with keys:
        train_size, test_size, metrics, feature_importances,
        predictions, test_df, best_model_name
    """
    feature_cols = list(feature_cols or FEATURE_COLS)
    tracker = tracker or tracking.NULL_TRACKER

    # ── Validate and prepare data ─────────────────────────────────────────
    validate_training_data(df, feature_cols)
    split = prepare_train_test_split(df, TRAIN_RATIO, feature_cols)
    X_train, X_test = split["X_train"], split["X_test"]
    y_train, y_test = split["y_train"], split["y_test"]
    train, test = split["train_df"], split["test_df"]

    # Naive baseline: always predict the majority class seen in training.
    # Any model that cannot beat this has learned nothing useful.
    majority_class = int(y_train.mean() >= 0.5)
    baseline_acc = float((y_test == majority_class).mean())

    tracker.log_params({
        "ticker": ticker,
        "feature_set_version": FEATURE_SET_VERSION,
        "n_features": len(feature_cols),
        "features": ",".join(feature_cols),
        "train_ratio": TRAIN_RATIO,
        "cv_strategy": f"TimeSeriesSplit(n_splits={CV_SPLITS})",
        "random_state": RANDOM_STATE,
        "train_start": train.index[0].strftime("%Y-%m-%d"),
        "train_end": train.index[-1].strftime("%Y-%m-%d"),
        "test_start": test.index[0].strftime("%Y-%m-%d"),
        "test_end": test.index[-1].strftime("%Y-%m-%d"),
        "train_size": len(train),
        "test_size": len(test),
    })
    tracker.log_metrics({
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "baseline_majority_accuracy": baseline_acc,
    })
    tracker.log_dataset(train, feature_cols, name=f"{ticker}-train", context="training")
    tracker.log_dataset(test, feature_cols, name=f"{ticker}-test", context="testing")

    # ── Build model pipelines ─────────────────────────────────────────────
    pipelines = build_models()
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    # ── Tune each pipeline (all receive identical X_train / y_train) ─────
    searches: Dict[str, RandomizedSearchCV] = {}
    tuned_models: Dict[str, Pipeline] = {}
    for name, pipeline in pipelines.items():
        search = RandomizedSearchCV(
            pipeline,
            PARAM_GRIDS[name],
            n_iter=N_ITER[name],
            cv=tscv,
            scoring="accuracy",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        search.fit(X_train, y_train)
        searches[name] = search
        tuned_models[name] = search.best_estimator_
        logger.info("  %s — best params: %s", name, search.best_params_)

    # ── Compare all 3 models on the held-out test set ────────────────────
    # All pipelines receive identical X_test — internal scalers handle
    # any necessary transformations.
    all_model_metrics = {}
    best_acc = -1.0
    best_model_name = ""
    best_model = None

    for name, model in tuned_models.items():
        y_pred_temp = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred_temp)
        prec = precision_score(y_test, y_pred_temp, zero_division=0)
        rec = recall_score(y_test, y_pred_temp, zero_division=0)
        all_model_metrics[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
        }
        logger.info("  %s — Acc=%.4f  Prec=%.4f  Rec=%.4f", name, acc, prec, rec)
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model = model

    logger.info("[Selection] Best model: %s (accuracy=%.4f)", best_model_name, best_acc)

    # One nested run per candidate: search space, best params, CV + test scores.
    for name, search in searches.items():
        tracker.log_candidate(
            name,
            search,
            all_model_metrics[name],
            is_best=(name == best_model_name),
        )

    # For feature importances we need a model with feature_importances_.
    # Always use RF's importances for interpretability — access the
    # classifier step inside its Pipeline via named_steps.
    importance_model = tuned_models["Random Forest"].named_steps["clf"]

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

    # ── Classification metrics (from the best model) ─────────────────────
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    }

    # ── Feature importances (always from RF for interpretability) ────────
    importances = [
        {"feature": feat, "importance": float(imp)}
        for feat, imp in sorted(
            zip(feature_cols, importance_model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
    ]

    # ── Track the selected model ──────────────────────────────────────────
    tracker.set_tags({"best_model": best_model_name})
    tracker.log_metrics({
        f"test_{k}": v for k, v in metrics.items()
    } | {"accuracy_lift_over_baseline": metrics["accuracy"] - baseline_acc})
    tracker.log_feature_importances(importances)
    tracker.log_dict(all_model_metrics, "model_comparison.json")
    tracker.log_model(best_model, X_train, ticker=ticker, model_name=best_model_name)

    # ── Build prediction records ──────────────────────────────────────────
    predictions = []
    for i, (idx, row) in enumerate(test.iterrows()):
        signal = int(y_pred[i])
        # Confidence = probability of the predicted class.
        confidence = float(y_proba[i][signal])

        predictions.append({
            "date": idx.date(),
            "signal": signal,
            "confidence": confidence,
        })

        # Store in model_predictions table.
        if db is not None:
            db.add(
                ModelPrediction(
                    ticker=ticker,
                    prediction_date=idx.date(),
                    predicted_signal=signal,
                    confidence_score=confidence,
                    actual_outcome=int(row["target"]),
                )
            )

    if db is not None:
        db.commit()
    logger.info(
        "Best model: %s — accuracy=%.4f, precision=%.4f, recall=%.4f",
        best_model_name, metrics["accuracy"], metrics["precision"], metrics["recall"],
    )

    return {
        "train_size": len(train),
        "test_size": len(test),
        "metrics": metrics,
        "feature_importances": importances,
        "predictions": predictions,
        "test_df": test,
        "y_pred": y_pred,
        "best_model_name": best_model_name,
        "all_model_metrics": all_model_metrics,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4 — BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════

def run_backtest(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    capital: float,
) -> Dict[str, Any]:
    """
    Simulate the ML strategy vs Buy-and-Hold on the test period.

    ML Strategy:
        Prediction made after day T's close → trade on day T+1.
        On 'Buy' signal, capture next day's intraday return = (Close-Open)/Open.
        On 'Sell/Hold' signal, stay in cash (return = 0).

    The prediction is shifted forward by 1 day so that a signal generated
    from day T's features drives day T+1's trade, eliminating look-ahead bias.

    Sharpe Ratio (annualised):
        Sharpe = (mean(excess_return) / std(strategy_return)) × √252
        excess_return = strategy_return - daily_risk_free_rate
    """
    test = test_df.copy()
    test["prediction"] = y_pred

    # Daily intraday return for each day.
    test["daily_return"] = (
        (test["close_price"] - test["open_price"]) / test["open_price"]
    )

    # ── Align prediction with execution ───────────────────────────────────
    # Prediction[i] is generated using features known at day i's close.
    # Therefore the trade happens on day i+1 (buy at open, sell at close).
    # shift(1) moves each prediction forward so it drives the NEXT day's
    # return.  The first day has no prior prediction → stays in cash (0).
    test["strategy_return"] = test["daily_return"] * test["prediction"].shift(1).fillna(0)

    # ── Cumulative returns ────────────────────────────────────────────────
    # Start cumulative tracking from the first day a trade can actually
    # be executed (the second row, since the first row has no prediction).
    ml_cumulative = (1 + test["strategy_return"]).cumprod()
    ml_final_value = float(capital * ml_cumulative.iloc[-1])

    # Buy-and-Hold: fully invested from Day 2 open (first tradeable day)
    # to last day close — aligned with the ML strategy's tradeable window.
    tradeable_start_idx = 1  # first day a prediction can drive a trade
    bh_return = float(
        test["close_price"].iloc[-1] / test["open_price"].iloc[tradeable_start_idx]
    )
    bh_final_value = float(capital * bh_return)

    # Buy-and-Hold cumulative series for charting.
    bh_cumulative = test["close_price"] / float(test["open_price"].iloc[tradeable_start_idx])

    # ── Sharpe Ratio ──────────────────────────────────────────────────────
    daily_rf = (
        (1 + settings.RISK_FREE_RATE)
        ** (1 / settings.TRADING_DAYS_PER_YEAR)
        - 1
    )
    excess = test["strategy_return"] - daily_rf
    std = test["strategy_return"].std()
    sharpe = float(
        (excess.mean() / std) * np.sqrt(settings.TRADING_DAYS_PER_YEAR)
        if std > 0
        else 0.0
    )

    # ── Build daily cumulative series for frontend charting ───────────────
    daily_cumulative = []
    for idx, ml_val, bh_val in zip(
        test.index, ml_cumulative, bh_cumulative
    ):
        daily_cumulative.append({
            "date": idx.strftime("%Y-%m-%d"),
            "ml_cumulative": round(float(ml_val), 6),
            "bh_cumulative": round(float(bh_val), 6),
        })

    ml_return_pct = round((ml_final_value / capital - 1) * 100, 2)
    bh_return_pct = round((bh_final_value / capital - 1) * 100, 2)

    logger.info(
        "Backtest complete: ML=₹%.2f (%+.2f%%), B&H=₹%.2f (%+.2f%%), Sharpe=%.4f",
        ml_final_value, ml_return_pct, bh_final_value, bh_return_pct, sharpe,
    )

    return {
        "starting_capital": capital,
        "ml_final_value": round(ml_final_value, 2),
        "bh_final_value": round(bh_final_value, 2),
        "ml_return_pct": ml_return_pct,
        "bh_return_pct": bh_return_pct,
        "sharpe_ratio": round(sharpe, 4),
        "daily_cumulative": daily_cumulative,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — runs the full pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_full_analysis(
    ticker: str,
    years: int,
    capital: float,
    db: Session,
) -> Dict[str, Any]:
    """
    Execute the complete pipeline: fetch → features → train → backtest.

    This is the main entry point called by the /api/analyze endpoint.
    Returns a fully structured dict ready for AnalysisResponse serialisation.
    """
    # Every analysis is one MLflow run; it is a no-op when tracking is off.
    with tracking.start_run(
        run_name=f"{ticker}-analysis",
        tags={"ticker": ticker, "source": "api"},
    ) as tracker:
        tracker.log_params({"lookback_years": years, "starting_capital": capital})

        # Phase 1+2: Fetch data, engineer features, store in DB.
        df = fetch_and_store(ticker, years, db)

        # Clear old predictions for this ticker.
        db.query(ModelPrediction).filter(ModelPrediction.ticker == ticker).delete()
        db.commit()

        # Phase 3: Train model, generate predictions.
        ml_results = train_and_predict(df, ticker, db, tracker=tracker)

        # Phase 4: Backtest the strategy.
        backtest_results = run_backtest(
            ml_results["test_df"],
            ml_results["y_pred"],
            capital,
        )
        tracker.log_backtest(backtest_results)

    # ── Assemble market_data for the frontend ─────────────────────────────
    market_data = []
    for idx, row in df.iterrows():
        market_data.append({
            "date": idx.date(),
            "open": round(float(row["open_price"]), 2),
            "high": round(float(row["high_price"]), 2),
            "low": round(float(row["low_price"]), 2),
            "close": round(float(row["close_price"]), 2),
            "volume": float(row["volume"]),
            "sma_14": round(float(row["sma_14"]), 2),
            "sma_50": round(float(row["sma_50"]), 2),
            "sma_200": round(float(row["sma_200"]), 2),
            "ema_14": round(float(row["ema_14"]), 2),
            "ema_20": round(float(row["ema_20"]), 2),
            "ema_50": round(float(row["ema_50"]), 2),
            "rsi_14": round(float(row["rsi_14"]), 2),
            "macd_line": round(float(row["macd_line"]), 2),
            "macd_signal": round(float(row["macd_signal"]), 2),
            "macd_hist": round(float(row["macd_hist"]), 2),
            "bb_upper": round(float(row["bb_upper"]), 2),
            "bb_lower": round(float(row["bb_lower"]), 2),
            "atr_14": round(float(row["atr_14"]), 2),
            "daily_return_vol": round(float(row["daily_return_vol"]), 4),
            "volume_ratio": round(float(row["volume_ratio"]), 2),
            "volume_change": round(float(row["volume_change"]), 2),
            "target": int(row["target"]),
        })

    # Date range metadata.
    date_range = {
        "start": df.index[0].strftime("%Y-%m-%d"),
        "end": df.index[-1].strftime("%Y-%m-%d"),
    }

    # Latest prediction (most recent test date).
    latest = ml_results["predictions"][-1] if ml_results["predictions"] else None

    return {
        "ticker": ticker,
        "data_points": len(df),
        "train_size": ml_results["train_size"],
        "test_size": ml_results["test_size"],
        "date_range": date_range,
        "market_data": market_data,
        "predictions": ml_results["predictions"],
        "latest_prediction": latest,
        "metrics": ml_results["metrics"],
        "feature_importances": ml_results["feature_importances"],
        "backtest": backtest_results,
        "best_model": ml_results["best_model_name"],
        "all_model_metrics": ml_results["all_model_metrics"],
        "mlflow_run_id": tracker.run_id,
    }
