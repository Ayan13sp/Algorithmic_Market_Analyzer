"""Tests for MLflow experiment tracking (app.tracking + train_and_predict hooks)."""

import pytest

mlflow = pytest.importorskip("mlflow")

from app import tracking  # noqa: E402
from app.config import settings  # noqa: E402
from app.ml_engine import (  # noqa: E402
    FEATURE_COLS,
    FEATURE_GROUPS,
    engineer_features,
    run_backtest,
    train_and_predict,
)

TICKER = "TEST.NS"


@pytest.fixture
def engineered_df(ohlcv):
    return engineer_features(ohlcv.copy())


@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    """Enable tracking against an isolated SQLite store + artifact dir."""
    monkeypatch.chdir(tmp_path)  # default artifact root (./mlruns) lands here
    monkeypatch.setattr(settings, "MLFLOW_ENABLED", True)
    monkeypatch.setattr(settings, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(settings, "MLFLOW_EXPERIMENT_NAME", "test-experiment")
    yield mlflow.MlflowClient(settings.MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(None)


def _child_runs(client, parent_id):
    parent = client.get_run(parent_id)
    return client.search_runs(
        [parent.info.experiment_id],
        filter_string=f"tags.`mlflow.parentRunId` = '{parent_id}'",
    )


def test_feature_groups_partition_feature_cols():
    grouped = [f for cols in FEATURE_GROUPS.values() for f in cols]
    assert sorted(grouped) == sorted(FEATURE_COLS)
    assert len(grouped) == len(set(grouped))


def test_disabled_tracking_yields_inactive_tracker():
    with tracking.start_run("disabled") as tracker:
        assert not tracker.active
        assert tracker.run_id is None
        assert tracker.log_metrics({"x": 1.0}) is None


def test_training_logs_parent_and_candidate_runs(mlflow_store, engineered_df):
    with tracking.start_run("unit-test", tags={"ticker": TICKER}) as tracker:
        results = train_and_predict(engineered_df, TICKER, db=None, tracker=tracker)
        tracker.log_backtest(run_backtest(results["test_df"], results["y_pred"], 100_000.0))

    run = mlflow_store.get_run(tracker.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["n_features"] == str(len(FEATURE_COLS))
    assert run.data.tags["best_model"] == results["best_model_name"]
    assert run.data.metrics["test_accuracy"] == pytest.approx(results["metrics"]["accuracy"])
    for key in ("baseline_majority_accuracy", "backtest_sharpe_ratio", "accuracy_lift_over_baseline"):
        assert key in run.data.metrics

    # Train/test datasets are recorded as run inputs.
    contexts = {
        tag.value
        for ds in run.inputs.dataset_inputs
        for tag in ds.tags
        if tag.key == "mlflow.data.context"
    }
    assert contexts == {"training", "testing"}

    artifacts = {a.path for a in mlflow_store.list_artifacts(tracker.run_id)}
    assert {"feature_importances.json", "model_comparison.json", "plots"} <= artifacts

    # One nested run per candidate, exactly one flagged as best.
    children = _child_runs(mlflow_store, tracker.run_id)
    assert {c.info.run_name for c in children} == {"Logistic Regression", "Random Forest", "SVM"}
    assert [c.data.tags["is_best"] for c in children].count("true") == 1
    assert all("cv_accuracy_mean" in c.data.metrics for c in children)

    # The selected model is logged, loadable, and registered.
    model_uri = f"models:/{tracking._registered_model_name(TICKER)}/1"
    loaded = mlflow.pyfunc.load_model(model_uri)
    preds = loaded.predict(results["test_df"][FEATURE_COLS].head(3))
    assert len(preds) == 3


def test_feature_subset_is_tracked(mlflow_store, engineered_df):
    cols = FEATURE_GROUPS["momentum"]
    with tracking.start_run("subset") as tracker:
        results = train_and_predict(engineered_df, TICKER, feature_cols=cols, tracker=tracker)

    assert [d["feature"] for d in results["feature_importances"]] != []
    assert {d["feature"] for d in results["feature_importances"]} == set(cols)
    assert mlflow_store.get_run(tracker.run_id).data.params["features"] == ",".join(cols)


def test_pipeline_error_marks_run_failed_and_propagates(mlflow_store):
    with pytest.raises(RuntimeError):
        with tracking.start_run("boom") as tracker:
            raise RuntimeError("pipeline failed")
    assert mlflow_store.get_run(tracker.run_id).info.status == "FAILED"


def test_unreachable_tracking_backend_does_not_break_training(monkeypatch, engineered_df):
    monkeypatch.setattr(settings, "MLFLOW_ENABLED", True)
    monkeypatch.setattr(settings, "MLFLOW_TRACKING_URI", "unsupported-scheme://nowhere")
    try:
        with tracking.start_run("offline") as tracker:
            results = train_and_predict(engineered_df, TICKER, tracker=tracker)
    finally:
        mlflow.set_tracking_uri(None)
    assert not tracker.active
    assert results["best_model_name"]


@pytest.mark.parametrize("family", ["Logistic Regression", "Random Forest", "SVM"])
def test_every_model_family_can_be_logged_and_reloaded(mlflow_store, engineered_df, family):
    """Regression: skops rejected RandomForest trees, silently skipping the artifact."""
    from app.ml_engine import build_models

    X, y = engineered_df[FEATURE_COLS], engineered_df["target"]
    model = build_models()[family].fit(X, y)
    with tracking.start_run(f"log-{family}") as tracker:
        tracker.log_model(model, X, ticker=TICKER, model_name=family)

    logged = mlflow_store.search_logged_models(
        experiment_ids=[mlflow_store.get_run(tracker.run_id).info.experiment_id]
    )
    assert len(logged) == 1
    reloaded = mlflow.sklearn.load_model(logged[0].model_uri)
    assert (reloaded.predict(X.head(5)) == model.predict(X.head(5))).all()
