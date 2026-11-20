"""
Walk-Forward Validation
=======================
Replaces a single 80/20 split with the protocol a live trading model actually
follows (expanding window, refit every ``step`` days):

    fold 1: train rows [0, 504)  → predict [504, 567)
    fold 2: train rows [0, 567)  → predict [567, 630)
    ...

At each fold the three candidate families are tuned with RandomizedSearchCV on
the training window only (inner TimeSeriesSplit), the family with the best
*cross-validated* balanced accuracy is selected (never the best test score),
and the next block is predicted out of sample. Concatenating the blocks gives
one continuous out-of-sample (OOS) prediction series, the only series the
backtest is allowed to use.

Why no gap between train and test? Labels look one day ahead: the label of the
last training row is the first test day's open-to-close move, which is known
at that day's close, exactly when the first test prediction is made. So no
future information reaches any prediction. ``gap`` exists for longer label
horizons (purging).

Why balanced accuracy? Up-days are not 50% of days, so plain accuracy rewards
a model that always predicts the majority class. Balanced accuracy scores
that degenerate model at exactly 0.5.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from app.config import settings
from app.estimators import MODEL_FAMILIES, N_ITER, PARAM_GRIDS, RANDOM_STATE, build_models

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardConfig:
    initial_train_size: int = 504      # ~2 trading years before the first OOS day
    step: int = 63                     # refit (and predict) one quarter at a time
    retune_every: int = 1              # re-run the search every N folds (0 = first fold only)
    gap: int = 0                       # rows purged between train and test
    max_train_size: Optional[int] = None  # None = expanding window, int = rolling window
    min_train_size: int = 120          # floor when the history is short
    cv_splits: int = 3                 # inner TimeSeriesSplit folds for tuning
    scoring: str = "balanced_accuracy"
    threshold: float = 0.5             # go long when P(up) > threshold
    n_jobs: int = -1
    random_state: int = RANDOM_STATE

    def __post_init__(self) -> None:
        if self.step < 1 or self.min_train_size < 2 * self.cv_splits:
            raise ValueError("step must be ≥ 1 and min_train_size large enough for inner CV")
        if self.initial_train_size < self.min_train_size:
            raise ValueError("initial_train_size must be ≥ min_train_size")
        if self.gap < 0 or self.retune_every < 0:
            raise ValueError("gap and retune_every must be non-negative")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be in (0, 1)")

    @classmethod
    def from_settings(cls) -> "WalkForwardConfig":
        return cls(
            initial_train_size=settings.WF_INITIAL_TRAIN_DAYS,
            step=settings.WF_STEP_DAYS,
            retune_every=settings.WF_RETUNE_EVERY,
            n_jobs=settings.ML_N_JOBS,
        )


def walk_forward_splits(
    n_samples: int, config: WalkForwardConfig
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Chronological (train_idx, test_idx) folds covering every row after the
    initial window exactly once.

    * Short histories shrink the initial window to half the data (≥ min_train_size).
    * A final block shorter than step/4 is merged into the previous fold.
    """
    min_tail = max(5, config.step // 4)
    needed = config.min_train_size + config.gap + min_tail
    if n_samples < needed:
        raise ValueError(
            f"Walk-forward validation needs at least {needed} labelled rows, "
            f"got {n_samples}. Use a longer lookback period."
        )
    initial = config.initial_train_size
    if n_samples < initial + config.gap + config.step:
        initial = max(config.min_train_size, (n_samples - config.gap) // 2)

    bounds = list(range(initial + config.gap, n_samples, config.step)) + [n_samples]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < min_tail:
        bounds.pop(-2)

    folds = []
    for test_start, test_end in itertools.pairwise(bounds):
        train_end = test_start - config.gap
        train_start = 0 if config.max_train_size is None else max(0, train_end - config.max_train_size)
        folds.append((np.arange(train_start, train_end), np.arange(test_start, test_end)))
    return folds


# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════

def classification_metrics(y_true, y_pred, proba=None) -> Dict[str, Any]:
    """Accuracy-family metrics; ROC AUC only when both classes are present."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
        "positive_rate": float(y_pred.mean()),
        "n": int(len(y_true)),
    }
    if proba is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, np.asarray(proba, dtype=float)))
    return out


def mcnemar_p_value(y_true, pred_a, pred_b) -> float:
    """
    Exact one-sided McNemar test: on the days where exactly one of the two
    models is right, is model A right more often than B?
    """
    y_true, pred_a, pred_b = (np.asarray(v, dtype=int) for v in (y_true, pred_a, pred_b))
    a_right, b_right = pred_a == y_true, pred_b == y_true
    wins, losses = int(np.sum(a_right & ~b_right)), int(np.sum(~a_right & b_right))
    if wins + losses == 0:
        return 1.0
    return float(stats.binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue)


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


# ═══════════════════════════════════════════════════════════════════════════
# WALK-FORWARD LOOP
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WalkForwardResult:
    config: WalkForwardConfig
    feature_cols: List[str]
    initial_train_size: int
    oos: pd.DataFrame                  # per OOS day: target, proba, signal, baseline, model, fold
    family_proba: pd.DataFrame         # OOS P(up) of every family (for comparison only)
    folds: List[Dict[str, Any]]
    fold_indices: List[Tuple[np.ndarray, np.ndarray]]
    metrics: Dict[str, Any]
    family_metrics: Dict[str, Dict[str, Any]]
    importances: pd.DataFrame          # Random Forest importances, fold × feature
    selected_model: str                # family chosen at the most recent tuning
    best_params: Dict[str, Dict[str, Any]]
    last_searches: Dict[str, RandomizedSearchCV]

    def summary(self) -> Dict[str, Any]:
        """JSON-friendly description of the validation protocol and folds."""
        counts = self.oos.drop_duplicates("fold")["model"].value_counts()
        return {
            "method": "walk_forward",
            "window": "expanding" if self.config.max_train_size is None else "rolling",
            "n_folds": len(self.folds),
            "initial_train_size": self.initial_train_size,
            "step": self.config.step,
            "retune_every": self.config.retune_every,
            "gap": self.config.gap,
            "selection_metric": self.config.scoring,
            "oos_start": self.oos.index[0].strftime("%Y-%m-%d"),
            "oos_end": self.oos.index[-1].strftime("%Y-%m-%d"),
            "n_oos": len(self.oos),
            "model_selection_counts": {f: int(counts.get(f, 0)) for f in MODEL_FAMILIES},
            "folds": self.folds,
        }


def _tune(name: str, X: pd.DataFrame, y: pd.Series, config: WalkForwardConfig) -> RandomizedSearchCV:
    search = RandomizedSearchCV(
        build_models()[name],
        PARAM_GRIDS[name],
        n_iter=N_ITER[name],
        cv=TimeSeriesSplit(n_splits=config.cv_splits),
        scoring=config.scoring,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
    )
    return search.fit(X, y)


def run_walk_forward(
    df: pd.DataFrame,
    feature_cols: List[str],
    config: Optional[WalkForwardConfig] = None,
) -> WalkForwardResult:
    """
    Run expanding-window walk-forward validation on labelled, date-sorted rows.

    Every model that predicts a fold's test block was fit (and selected) using
    only rows strictly before that block.
    """
    config = config or WalkForwardConfig.from_settings()
    if not df.index.is_monotonic_increasing:
        raise ValueError("Rows must be in chronological order for walk-forward validation.")
    if df["target"].isna().any():
        raise ValueError("Walk-forward validation needs labelled rows only (no NaN targets).")

    X, y = df[feature_cols], df["target"].astype(int)
    folds = walk_forward_splits(len(df), config)
    n = len(df)
    family_proba = {f: np.full(n, np.nan) for f in MODEL_FAMILIES}
    chosen = np.empty(n, dtype=object)
    fold_of = np.zeros(n, dtype=int)
    baseline = np.zeros(n, dtype=int)

    params: Dict[str, Dict[str, Any]] = {}
    cv_scores: Dict[str, float] = {}
    searches: Dict[str, RandomizedSearchCV] = {}
    selected = MODEL_FAMILIES[0]
    records: List[Dict[str, Any]] = []
    importances: List[np.ndarray] = []

    for k, (tr, te) in enumerate(folds):
        X_tr, y_tr, X_te = X.iloc[tr], y.iloc[tr], X.iloc[te]
        retune = k == 0 or (config.retune_every > 0 and k % config.retune_every == 0)
        if retune:
            searches = {f: _tune(f, X_tr, y_tr, config) for f in MODEL_FAMILIES}
            params = {f: s.best_params_ for f, s in searches.items()}
            cv_scores = {f: float(s.best_score_) for f, s in searches.items()}
            selected = max(
                MODEL_FAMILIES,
                key=lambda f: (np.nan_to_num(cv_scores[f], nan=-np.inf), -MODEL_FAMILIES.index(f)),
            )
            models: Dict[str, Pipeline] = {f: s.best_estimator_ for f, s in searches.items()}
        else:
            models = {f: build_models()[f].set_params(**params[f]).fit(X_tr, y_tr) for f in MODEL_FAMILIES}

        for f, model in models.items():
            family_proba[f][te] = model.predict_proba(X_te)[:, 1]
        importances.append(models["Random Forest"].named_steps["clf"].feature_importances_)
        chosen[te] = selected
        fold_of[te] = k + 1
        baseline[te] = int(y_tr.mean() >= 0.5)  # majority class of the training window

        fold_proba = family_proba[selected][te]
        fm = classification_metrics(y.iloc[te], (fold_proba > config.threshold).astype(int), fold_proba)
        records.append({
            "fold": k + 1,
            "train_start": df.index[tr[0]].strftime("%Y-%m-%d"),
            "train_end": df.index[tr[-1]].strftime("%Y-%m-%d"),
            "test_start": df.index[te[0]].strftime("%Y-%m-%d"),
            "test_end": df.index[te[-1]].strftime("%Y-%m-%d"),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "model": selected,
            "retuned": bool(retune),
            "cv_score": _round(cv_scores[selected]),
            "accuracy": _round(fm["accuracy"]),
            "balanced_accuracy": _round(fm["balanced_accuracy"]),
            "roc_auc": _round(fm["roc_auc"]),
            "positive_rate": _round(fm["positive_rate"]),
        })
        logger.info(
            "Fold %d/%d: train %d rows → test %d rows, %s, balanced acc %.3f",
            k + 1, len(folds), len(tr), len(te), selected, fm["balanced_accuracy"],
        )

    oos_idx = np.concatenate([te for _, te in folds])
    oos = pd.DataFrame(index=df.index[oos_idx])
    oos["target"] = y.to_numpy()[oos_idx]
    oos["model"] = chosen[oos_idx]
    oos["proba"] = [family_proba[m][i] for m, i in zip(chosen[oos_idx], oos_idx, strict=True)]
    oos["signal"] = (oos["proba"] > config.threshold).astype(int)
    oos["baseline"] = baseline[oos_idx]
    oos["fold"] = fold_of[oos_idx]
    fam = pd.DataFrame({f: family_proba[f][oos_idx] for f in MODEL_FAMILIES}, index=oos.index)

    metrics = classification_metrics(oos["target"], oos["signal"], oos["proba"])
    metrics["baseline_accuracy"] = float((oos["baseline"] == oos["target"]).mean())
    metrics["accuracy_lift"] = metrics["accuracy"] - metrics["baseline_accuracy"]
    metrics["p_value_vs_baseline"] = mcnemar_p_value(oos["target"], oos["signal"], oos["baseline"])
    family_metrics = {
        f: classification_metrics(oos["target"], (fam[f] > config.threshold).astype(int), fam[f])
        for f in MODEL_FAMILIES
    }

    return WalkForwardResult(
        config=config,
        feature_cols=list(feature_cols),
        initial_train_size=int(len(folds[0][0])),
        oos=oos,
        family_proba=fam,
        folds=records,
        fold_indices=folds,
        metrics=metrics,
        family_metrics=family_metrics,
        importances=pd.DataFrame(importances, columns=feature_cols, index=range(1, len(folds) + 1)),
        selected_model=selected,
        best_params=params,
        last_searches=searches,
    )


def fit_production_model(
    df: pd.DataFrame, feature_cols: List[str], result: WalkForwardResult
) -> Pipeline:
    """Refit the most recently selected configuration on every labelled row."""
    name = result.selected_model
    model = build_models()[name].set_params(**result.best_params[name])
    return model.fit(df[feature_cols], df["target"].astype(int))
