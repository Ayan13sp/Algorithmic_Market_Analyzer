"""
Candidate Estimators
====================
The three model families compared at every walk-forward fold, and their
hyperparameter search spaces.

Each family is a full sklearn ``Pipeline`` that owns its preprocessing, so
every candidate receives the same raw feature matrix and scalers are only
ever fit on training data.
"""

from typing import Dict, List

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42

# Fixed order: also the tie-break order when CV scores are equal.
MODEL_FAMILIES: List[str] = ["Logistic Regression", "Random Forest", "SVM"]


def build_models() -> Dict[str, Pipeline]:
    """
    Model registry: the three candidate pipelines with default hyperparameters.

    Tuning is the caller's job (walk-forward wraps these in RandomizedSearchCV).
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
            ("clf", SVC(probability=True, kernel="rbf", random_state=RANDOM_STATE)),
        ]),
    }


# Search spaces (clf__-prefixed for Pipeline compatibility). L2 is already
# LogisticRegression's default penalty; scikit-learn 1.10 removes `penalty`.
PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "Logistic Regression": {
        "clf__C": [0.1, 1.0, 10.0],
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
