"""Unit tests for the pure indicator and backtest functions in app.ml_engine.

Stage 1 model tests verify that build_models(), validate_training_data(),
and prepare_train_test_split() work correctly in isolation — before any
hyperparameter tuning is involved.
"""

import numpy as np
import pandas as pd
import pytest

from app.ml_engine import (
    FEATURE_COLS,
    build_models,
    compute_ema,
    compute_rsi,
    compute_sma,
    engineer_features,
    prepare_train_test_split,
    run_backtest,
    validate_training_data,
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


@pytest.fixture
def engineered_df(ohlcv) -> pd.DataFrame:
    """Fully engineered DataFrame ready for model training."""
    return engineer_features(ohlcv.copy())


# ═══════════════════════════════════════════════════════════════════════════
# ORIGINAL INDICATOR TESTS
# ═══════════════════════════════════════════════════════════════════════════

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


def test_target_is_next_day_intraday_direction(ohlcv):
    out = engineer_features(ohlcv.copy())
    close = ohlcv["close_price"]
    open_price = ohlcv["open_price"]
    day = out.index[10]
    nxt_loc = close.index.get_loc(day) + 1
    nxt = close.index[nxt_loc]
    # Target now reflects next day's intraday direction (close > open on T+1)
    assert out.loc[day, "target"] == int(close[nxt] > open_price[nxt])


def test_backtest_all_cash_returns_flat_capital(ohlcv):
    test = ohlcv.iloc[:50]
    result = run_backtest(test, np.zeros(len(test), dtype=int), capital=100_000.0)
    assert result["ml_final_value"] == pytest.approx(100_000.0)
    assert result["ml_return_pct"] == pytest.approx(0.0)
    assert len(result["daily_cumulative"]) == len(test)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — MODEL REGISTRY TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildModels:
    """Tests for the build_models() pipeline registry."""

    def test_returns_three_named_pipelines(self):
        models = build_models()
        assert len(models) == 3
        assert set(models.keys()) == {
            "Logistic Regression", "Random Forest", "SVM",
        }

    def test_each_entry_is_a_sklearn_pipeline(self):
        from sklearn.pipeline import Pipeline
        for name, pipeline in build_models().items():
            assert isinstance(pipeline, Pipeline), f"{name} is not a Pipeline"

    def test_lr_pipeline_has_scaler(self):
        lr = build_models()["Logistic Regression"]
        assert "scaler" in lr.named_steps
        assert "clf" in lr.named_steps

    def test_rf_pipeline_has_no_scaler(self):
        rf = build_models()["Random Forest"]
        assert "scaler" not in rf.named_steps
        assert "clf" in rf.named_steps

    def test_svm_pipeline_has_scaler(self):
        svm = build_models()["SVM"]
        assert "scaler" in svm.named_steps
        assert "clf" in svm.named_steps

    def test_all_pipelines_fit_and_predict(self, engineered_df):
        """Verify all 3 models can fit and predict on real engineered data."""
        split = prepare_train_test_split(engineered_df)
        X_train, X_test = split["X_train"], split["X_test"]
        y_train, y_test = split["y_train"], split["y_test"]

        for name, pipeline in build_models().items():
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            assert len(preds) == len(y_test), f"{name} prediction length mismatch"
            assert set(np.unique(preds)).issubset({0, 1}), (
                f"{name} produced non-binary predictions"
            )

    def test_all_models_receive_identical_training_data(self, engineered_df):
        """Ensure every model sees the exact same raw feature matrix."""
        split = prepare_train_test_split(engineered_df)
        X_train = split["X_train"]

        # Fit all models on the same X_train — if any pipeline mutated
        # the input, subsequent models would see corrupted data.
        x_train_original = X_train.copy()
        for name, pipeline in build_models().items():
            pipeline.fit(X_train, split["y_train"])
            pd.testing.assert_frame_equal(
                X_train, x_train_original,
                obj=f"X_train after fitting {name}",
            )


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — INPUT VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestValidateTrainingData:
    """Tests for validate_training_data()."""

    def test_accepts_valid_engineered_dataframe(self, engineered_df):
        # Should not raise.
        validate_training_data(engineered_df)

    def test_rejects_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            validate_training_data(pd.DataFrame())

    def test_rejects_missing_feature_columns(self, engineered_df):
        df = engineered_df.drop(columns=["sma_14", "rsi_14"])
        with pytest.raises(ValueError, match="Missing.*feature column"):
            validate_training_data(df)

    def test_rejects_missing_target_column(self, engineered_df):
        df = engineered_df.drop(columns=["target"])
        with pytest.raises(ValueError, match="target"):
            validate_training_data(df)

    def test_rejects_nan_in_features(self, engineered_df):
        df = engineered_df.copy()
        df.iloc[0, df.columns.get_loc("sma_14")] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            validate_training_data(df)

    def test_rejects_inf_in_features(self, engineered_df):
        df = engineered_df.copy()
        df.iloc[0, df.columns.get_loc("volume_ratio")] = np.inf
        with pytest.raises(ValueError, match="Infinite"):
            validate_training_data(df)

    def test_rejects_non_binary_target(self, engineered_df):
        df = engineered_df.copy()
        df.iloc[0, df.columns.get_loc("target")] = 2
        with pytest.raises(ValueError, match="binary"):
            validate_training_data(df)

    def test_rejects_too_few_rows(self):
        """A DataFrame with < 50 rows should fail validation."""
        rng = np.random.default_rng(42)
        n = 30
        df = pd.DataFrame(
            {col: rng.random(n) for col in FEATURE_COLS},
            index=pd.date_range("2023-01-01", periods=n, freq="B"),
        )
        df["target"] = rng.choice([0, 1], n)
        with pytest.raises(ValueError, match="at least 50"):
            validate_training_data(df)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — TRAIN / TEST SPLIT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestPrepareTrainTestSplit:
    """Tests for prepare_train_test_split()."""

    def test_preserves_chronological_order(self, engineered_df):
        split = prepare_train_test_split(engineered_df)
        train_end = split["train_df"].index[-1]
        test_start = split["test_df"].index[0]
        assert train_end < test_start, "Train period must end before test begins"

    def test_split_ratio(self, engineered_df):
        split = prepare_train_test_split(engineered_df, train_ratio=0.80)
        total = len(split["train_df"]) + len(split["test_df"])
        train_ratio = len(split["train_df"]) / total
        assert 0.79 <= train_ratio <= 0.81

    def test_rejects_unsorted_index(self, engineered_df):
        shuffled = engineered_df.sample(frac=1.0, random_state=0)
        with pytest.raises(ValueError, match="chronological"):
            prepare_train_test_split(shuffled)

    def test_drops_residual_inf_values(self, engineered_df):
        df = engineered_df.copy()
        # Inject an inf — should be silently dropped.
        df.iloc[5, df.columns.get_loc("volume_ratio")] = np.inf
        split = prepare_train_test_split(df)
        combined = pd.concat([split["train_df"], split["test_df"]])
        assert not np.isinf(combined[FEATURE_COLS].values).any()

    def test_feature_and_target_shapes_match(self, engineered_df):
        split = prepare_train_test_split(engineered_df)
        assert len(split["X_train"]) == len(split["y_train"])
        assert len(split["X_test"]) == len(split["y_test"])
        assert split["X_train"].shape[1] == len(FEATURE_COLS)
