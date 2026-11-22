"""Tests for walk-forward validation (app.walk_forward): fold geometry, the
retuning schedule, out-of-sample bookkeeping, metrics and determinism."""

import json

import numpy as np
import pytest

from app.estimators import MODEL_FAMILIES
from app.ml_engine import FEATURE_COLS, engineer_features, labelled_rows
from app.synthetic import planted_signal_ohlcv
from app.walk_forward import (
    WalkForwardConfig,
    classification_metrics,
    fit_production_model,
    mcnemar_p_value,
    run_walk_forward,
    walk_forward_splits,
)

FAST = dict(n_jobs=1)


@pytest.fixture(scope="module")
def planted():
    """~600 labelled rows with a learnable edge."""
    return labelled_rows(engineer_features(planted_signal_ohlcv(n=800, seed=3)))


@pytest.fixture(scope="module")
def walk(planted):
    cfg = WalkForwardConfig(initial_train_size=300, step=63, retune_every=2, **FAST)
    return run_walk_forward(planted, FEATURE_COLS, cfg)


# ═══════════════════════════════════════════════════════════════════════════
# FOLD GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════

class TestSplits:
    def test_expanding_window_covers_every_oos_row_once(self):
        folds = walk_forward_splits(1000, WalkForwardConfig())
        test_rows = np.concatenate([te for _, te in folds])
        assert test_rows[0] == 504 and test_rows[-1] == 999
        assert np.array_equal(test_rows, np.arange(504, 1000))  # contiguous, no overlap
        for train, test in folds:
            assert train[0] == 0 and train[-1] == test[0] - 1  # all past data, nothing after
            assert len(test) <= 63 + 63 // 4

    def test_gap_purges_rows_between_train_and_test(self):
        for train, test in walk_forward_splits(800, WalkForwardConfig(gap=5)):
            assert test[0] - train[-1] == 6

    def test_rolling_window_caps_training_length(self):
        folds = walk_forward_splits(1200, WalkForwardConfig(max_train_size=400))
        assert all(len(train) <= 400 for train, _ in folds)
        assert folds[-1][0][0] > 0

    def test_short_history_shrinks_the_initial_window(self):
        folds = walk_forward_splits(300, WalkForwardConfig())
        assert len(folds[0][0]) == 150
        assert sum(len(te) for _, te in folds) == 150

    def test_short_final_block_is_merged(self):
        # 504 + 63·3 = 693 → a 7-row tail is too short to be its own fold.
        folds = walk_forward_splits(700, WalkForwardConfig())
        assert len(folds) == 3 and len(folds[-1][1]) == 63 + 7

    def test_too_little_data_is_rejected(self):
        with pytest.raises(ValueError, match="at least"):
            walk_forward_splits(100, WalkForwardConfig())

    @pytest.mark.parametrize(
        "kwargs", [{"step": 0}, {"gap": -1}, {"retune_every": -1}, {"threshold": 1.0},
                   {"initial_train_size": 50}],
    )
    def test_invalid_config_is_rejected(self, kwargs):
        with pytest.raises(ValueError):
            WalkForwardConfig(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# WALK-FORWARD LOOP
# ═══════════════════════════════════════════════════════════════════════════

class TestRunWalkForward:
    def test_oos_series_is_continuous_after_the_initial_window(self, planted, walk):
        assert walk.initial_train_size == 300
        assert list(walk.oos.index) == list(planted.index[300:])
        assert set(walk.oos["model"]) <= set(MODEL_FAMILIES)
        assert walk.oos["signal"].isin([0, 1]).all()
        assert ((walk.oos["proba"] > 0.5).astype(int) == walk.oos["signal"]).all()

    def test_every_fold_trains_strictly_before_it_tests(self, walk):
        for fold in walk.folds:
            assert fold["train_end"] < fold["test_start"]
        assert sum(f["n_test"] for f in walk.folds) == len(walk.oos)

    def test_retune_schedule(self, walk):
        assert [f["retuned"] for f in walk.folds] == [k % 2 == 0 for k in range(len(walk.folds))]
        cfg = WalkForwardConfig(initial_train_size=300, retune_every=0, **FAST)
        assert [f["retuned"] for f in run_walk_forward_cached(cfg)] == [True] + [False] * 4

    def test_baseline_is_each_training_windows_majority_class(self, planted, walk):
        for (train, test), fold in zip(walk.fold_indices, walk.folds, strict=True):
            majority = int(planted["target"].iloc[train].mean() >= 0.5)
            assert (walk.oos["baseline"].iloc[test - 300] == majority).all(), fold["fold"]

    def test_finds_the_planted_edge(self, walk):
        assert walk.metrics["balanced_accuracy"] > 0.55
        assert walk.metrics["roc_auc"] > 0.58
        assert walk.metrics["p_value_vs_baseline"] < 0.05

    def test_family_comparison_and_importances(self, walk):
        assert list(walk.family_proba.columns) == MODEL_FAMILIES
        assert set(walk.family_metrics) == set(MODEL_FAMILIES)
        assert walk.importances.shape == (len(walk.folds), len(FEATURE_COLS))
        # Impurity importances are non-negative and sum to 1 within each fold.
        assert (walk.importances >= 0).all().all()
        assert np.allclose(walk.importances.sum(axis=1), 1.0)

    def test_summary_is_json_serialisable(self, walk):
        summary = json.loads(json.dumps(walk.summary()))
        assert summary["n_folds"] == len(walk.folds)
        assert sum(summary["model_selection_counts"].values()) == len(walk.folds)

    def test_is_deterministic(self, planted, walk):
        cfg = WalkForwardConfig(initial_train_size=300, step=63, retune_every=2, **FAST)
        again = run_walk_forward(planted, FEATURE_COLS, cfg)
        assert np.array_equal(again.oos["proba"].to_numpy(), walk.oos["proba"].to_numpy())

    def test_rejects_unsorted_rows_and_unlabelled_rows(self, planted):
        with pytest.raises(ValueError, match="chronological"):
            run_walk_forward(planted.iloc[::-1], FEATURE_COLS, WalkForwardConfig(**FAST))
        unlabelled = planted.copy()
        unlabelled.iloc[-1, unlabelled.columns.get_loc("target")] = np.nan
        with pytest.raises(ValueError, match="labelled"):
            run_walk_forward(unlabelled, FEATURE_COLS, WalkForwardConfig(**FAST))

    def test_production_model_uses_the_latest_selection(self, planted, walk):
        model = fit_production_model(planted, FEATURE_COLS, walk)
        assert model.n_features_in_ == len(FEATURE_COLS)
        assert type(model.named_steps["clf"]).__name__ in {
            "LogisticRegression", "RandomForestClassifier", "SVC",
        }
        assert model.predict_proba(planted[FEATURE_COLS].tail(3)).shape == (3, 2)


_CACHE = {}


def run_walk_forward_cached(cfg):
    """Small helper so the retune-schedule test does not re-run the fixture data."""
    if cfg not in _CACHE:
        data = labelled_rows(engineer_features(planted_signal_ohlcv(n=800, seed=3)))
        _CACHE[cfg] = run_walk_forward(data, FEATURE_COLS, cfg).folds
    return _CACHE[cfg]


# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════

class TestMetrics:
    def test_degenerate_majority_predictor_scores_half_balanced_accuracy(self):
        y = np.array([0] * 60 + [1] * 40)
        m = classification_metrics(y, np.zeros(100, dtype=int), np.full(100, 0.3))
        assert m["accuracy"] == pytest.approx(0.6)
        assert m["balanced_accuracy"] == pytest.approx(0.5)
        assert m["roc_auc"] == pytest.approx(0.5)

    def test_auc_is_undefined_with_one_class(self):
        assert classification_metrics([1, 1, 1], [1, 0, 1], [0.9, 0.2, 0.8])["roc_auc"] is None

    def test_mcnemar(self):
        y = np.array([1, 0] * 50)
        assert mcnemar_p_value(y, y, y) == 1.0  # identical predictions: no evidence
        always_right, always_wrong = y, 1 - y
        assert mcnemar_p_value(y, always_right, always_wrong) < 1e-20
        assert mcnemar_p_value(y, always_wrong, always_right) == pytest.approx(1.0)
