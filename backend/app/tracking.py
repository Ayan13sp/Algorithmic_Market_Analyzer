"""
Experiment Tracking (MLflow)
============================
Records every training run so experiments are reproducible and comparable:

    Parent run  (one per analysis / experiment config)
    ├── params     ticker, feature list, feature-set version, split dates, CV setup
    ├── inputs     train / test datasets (MLflow dataset digests + schema)
    ├── metrics    test accuracy / precision / recall, majority-class baseline,
    │              backtest return, Sharpe ratio
    ├── artifacts  feature importances (json + png), model comparison,
    │              equity curve, the selected sklearn Pipeline (+ signature)
    └── child runs (one per candidate model)
        ├── params     search space + best hyperparameters
        ├── metrics    CV accuracy mean/std, fit time, test metrics
        └── artifacts  full RandomizedSearchCV cv_results_ table

Design rules
------------
* **Optional dependency.** If ``mlflow`` is not installed or
  ``MLFLOW_ENABLED=false``, every call is a no-op (the lean Render image
  ships without MLflow).
* **Fail-soft.** Tracking must never break a prediction request: any MLflow
  error is logged as a warning and swallowed. Exceptions raised by the
  pipeline itself still propagate, and the run is marked FAILED.
* **Thread-safe.** MLflow's active-run stack is thread-local and each
  request opens its own run, so concurrent FastAPI workers do not mix runs.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)

try:  # MLflow is an optional dependency (see requirements-tracking.txt).
    import mlflow
    from mlflow.models import infer_signature

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without mlflow
    mlflow = None
    MLFLOW_AVAILABLE = False


def is_enabled() -> bool:
    """Tracking is active only when configured on *and* mlflow is importable."""
    return settings.MLFLOW_ENABLED and MLFLOW_AVAILABLE


def _safe(method):
    """Decorator: no-op when the tracker is inactive; never raise MLflow errors."""

    @functools.wraps(method)
    def wrapper(self: "ExperimentTracker", *args, **kwargs):
        if not self.active:
            return None
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - tracking is best-effort
            logger.warning("MLflow %s failed: %s", method.__name__, exc)
            return None

    return wrapper


SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


def _registered_model_name(ticker: str) -> str:
    """'RELIANCE.NS' → 'market-direction-RELIANCE-NS'."""
    return "market-direction-" + re.sub(r"[^A-Za-z0-9]+", "-", ticker).strip("-")


@functools.lru_cache(maxsize=1)
def _git_commit() -> Optional[str]:
    """Commit SHA of the running code, for reproducibility (best-effort)."""
    for var in ("GIT_SHA", "RENDER_GIT_COMMIT"):
        if os.getenv(var):
            return os.environ[var]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=2, check=True,
        ).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


class ExperimentTracker:
    """Thin, fail-soft facade over the MLflow fluent API for one run."""

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id

    @property
    def active(self) -> bool:
        return self.run_id is not None

    # ── Generic logging ──────────────────────────────────────────────────
    @_safe
    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

    @_safe
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

    @_safe
    def set_tags(self, tags: Dict[str, Any]) -> None:
        mlflow.set_tags(tags)

    @_safe
    def log_dict(self, obj: Any, artifact_file: str) -> None:
        mlflow.log_dict(obj, artifact_file)

    # ── Feature / dataset lineage ────────────────────────────────────────
    @_safe
    def log_dataset(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        name: str,
        context: str,
    ) -> None:
        """Log the exact feature matrix used, so runs are comparable by digest."""
        data = df[feature_cols + ["target"]].reset_index(names="date")
        dataset = mlflow.data.from_pandas(data, name=name, targets="target")
        mlflow.log_input(dataset, context=context)

    @_safe
    def log_feature_importances(self, importances: List[Dict[str, Any]]) -> None:
        mlflow.log_dict(importances, "feature_importances.json")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [d["feature"] for d in importances][::-1]
        values = [d["importance"] for d in importances][::-1]
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(names) + 1.2))
        ax.barh(names, values, color="#4C78A8")
        ax.set_xlabel("Mean decrease in impurity (Random Forest)")
        ax.set_title("Feature importances")
        fig.tight_layout()
        mlflow.log_figure(fig, "plots/feature_importances.png")
        plt.close(fig)

    # ── Model candidates (nested runs) ───────────────────────────────────
    @_safe
    def log_candidate(
        self,
        name: str,
        search: Any,
        test_metrics: Dict[str, float],
        is_best: bool,
    ) -> None:
        """Log one tuned candidate (a fitted RandomizedSearchCV) as a child run."""
        cv = search.cv_results_
        best = search.best_index_
        with mlflow.start_run(
            # Without this, nested runs land in the "Default" experiment.
            experiment_id=mlflow.active_run().info.experiment_id,
            run_name=name,
            nested=True,
            tags={"model_family": name, "is_best": str(is_best).lower()},
        ):
            mlflow.log_params({
                **{k.removeprefix("clf__"): v for k, v in search.best_params_.items()},
                "search_n_iter": search.n_iter,
                "search_scoring": search.scoring,
            })
            mlflow.log_dict(
                {k: [str(v) for v in vals] for k, vals in search.param_distributions.items()},
                "search_space.json",
            )
            mlflow.log_metrics({
                "cv_accuracy_mean": float(search.best_score_),
                "cv_accuracy_std": float(cv["std_test_score"][best]),
                "mean_fit_time_s": float(cv["mean_fit_time"][best]),
                **{f"test_{k}": float(v) for k, v in test_metrics.items()},
            })
            mlflow.log_text(
                pd.DataFrame(cv).drop(columns="params").to_csv(index=False),
                "cv_results.csv",
            )

    # ── Model artifact ───────────────────────────────────────────────────
    @_safe
    def log_model(self, model: Any, X_example: pd.DataFrame, ticker: str, model_name: str) -> None:
        """Log the selected Pipeline with its signature; optionally register it."""
        import sklearn

        example = X_example.head(5)
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(example, model.predict_proba(example)),
            input_example=example,
            # Pin requirements explicitly: automatic inference spawns a
            # subprocess and adds seconds to every API request.
            pip_requirements=[f"scikit-learn=={sklearn.__version__}", "pandas", "numpy"],
            # MLflow serialises with skops (no arbitrary code on load) and
            # rejects tree internals unless explicitly trusted. These are
            # models we just trained ourselves, so trusting them is safe.
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
            registered_model_name=(
                _registered_model_name(ticker) if settings.MLFLOW_REGISTER_MODELS else None
            ),
            tags={"model_family": model_name},
        )

    # ── Backtest ─────────────────────────────────────────────────────────
    @_safe
    def log_backtest(self, backtest: Dict[str, Any]) -> None:
        mlflow.log_metrics({
            "backtest_ml_return_pct": backtest["ml_return_pct"],
            "backtest_bh_return_pct": backtest["bh_return_pct"],
            "backtest_excess_return_pct": backtest["ml_return_pct"] - backtest["bh_return_pct"],
            "backtest_sharpe_ratio": backtest["sharpe_ratio"],
        })

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        curve = pd.DataFrame(backtest["daily_cumulative"])
        curve["date"] = pd.to_datetime(curve["date"])
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(curve["date"], curve["ml_cumulative"], label="ML strategy")
        ax.plot(curve["date"], curve["bh_cumulative"], label="Buy & hold")
        ax.set_ylabel("Growth of 1 unit")
        ax.set_title("Out-of-sample equity curve")
        ax.legend()
        fig.tight_layout()
        mlflow.log_figure(fig, "plots/equity_curve.png")
        plt.close(fig)


NULL_TRACKER = ExperimentTracker(None)


def _experiment_id(name: str) -> str:
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    try:
        return client.create_experiment(name)
    except Exception:  # noqa: BLE001 - lost a creation race with another worker
        return client.get_experiment_by_name(name).experiment_id


@contextlib.contextmanager
def start_run(
    run_name: str,
    tags: Optional[Dict[str, str]] = None,
    experiment_name: Optional[str] = None,
) -> Iterator[ExperimentTracker]:
    """
    Open an MLflow run (nested if one is already active on this thread).

    Yields an inactive tracker when tracking is disabled or the tracking
    server is unreachable, so callers never need to branch on it.
    """
    tracker = NULL_TRACKER
    if is_enabled():
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            run = mlflow.start_run(
                experiment_id=_experiment_id(experiment_name or settings.MLFLOW_EXPERIMENT_NAME),
                run_name=run_name,
                nested=mlflow.active_run() is not None,
                tags={
                    **(tags or {}),
                    **({"mlflow.source.git.commit": _git_commit()} if _git_commit() else {}),
                },
            )
            tracker = ExperimentTracker(run.info.run_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow tracking unavailable, continuing without it: %s", exc)

    status = "FINISHED"
    try:
        yield tracker
    except BaseException:
        status = "FAILED"
        raise
    finally:
        if tracker.active:
            try:
                mlflow.end_run(status=status)
            except Exception as exc:  # noqa: BLE001
                logger.warning("MLflow end_run failed: %s", exc)
