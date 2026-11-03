"""Hyperparameter search spaces and helpers (scored with PR-AUC)."""
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ( BaseCrossValidator, GridSearchCV, RandomizedSearchCV,)
from sklearn.pipeline import Pipeline
from src.data import RANDOM_STATE
from src.evaluate import SCORERS, make_cv
from src.pipeline import build_pipeline

MAIN_METRIC = "pr_auc"

LOGREG_GRID = {
    "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
    "model__class_weight": [None, "balanced"],
}

RF_GRID = {
    "model__min_samples_leaf": [5, 10, 20, 40],
    "model__max_depth": [6, 10, None],
    "model__class_weight": [None, "balanced"],
}

HISTGB_SPACE = {
    "model__learning_rate": loguniform(0.02, 0.2),
    "model__max_iter": [100, 200, 300],
    "model__max_leaf_nodes": [4, 8, 15, 31],
    "model__min_samples_leaf": [20, 50, 100],
    "model__l2_regularization": [0.0, 0.1, 1.0, 10.0],
    "model__class_weight": [None, "balanced"],
}

HISTGB_N_ITER = 40


def _base_estimator(name: str, random_state: int) -> BaseEstimator:
    if name == "LogReg":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    if name == "HistGB":
        return HistGradientBoostingClassifier(random_state=random_state)
    raise ValueError(f"Unknown model name: {name!r}")


def make_search(name: str,X: pd.DataFrame,cv: BaseCrossValidator | None = None,random_state: int = RANDOM_STATE,n_jobs: int | None = None,) -> GridSearchCV | RandomizedSearchCV:
    """Search object for one model, selected by PR-AUC and refit on all train data."""
    cv = cv or make_cv()
    pipe = build_pipeline(_base_estimator(name, random_state), X)
    common = dict(scoring=SCORERS,refit=MAIN_METRIC,cv=cv,n_jobs=n_jobs,return_train_score=False,)
    if name == "LogReg":
        return GridSearchCV(pipe, LOGREG_GRID, **common)
    if name == "RandomForest":
        return GridSearchCV(pipe, RF_GRID, **common)
    if name == "HistGB":
        return RandomizedSearchCV(pipe,HISTGB_SPACE,n_iter=HISTGB_N_ITER,random_state=random_state,**common,)
    raise ValueError(f"Unknown model name: {name!r}")


def tuned_pipeline(name: str,params: dict,X: pd.DataFrame,random_state: int = RANDOM_STATE,) -> Pipeline:
    """Rebuild a pipeline with previously found best parameters (unfitted)."""
    pipe = build_pipeline(_base_estimator(name, random_state), X)
    return pipe.set_params(**params)
