"""Shared cross-validation setup and scoring."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from src.data import RANDOM_STATE

SCORERS: dict = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def make_cv(n_splits: int = 5, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """The stratified K-fold used for every model comparison."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cv_scores(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold | None = None,
) -> dict[str, float]:
    """Cross-validate ``model`` and return mean/std per metric plus train means."""
    cv = cv or make_cv()
    res = cross_validate(
        model, X, y, cv=cv, scoring=SCORERS, return_train_score=True
    )
    out: dict[str, float] = {}
    for m in SCORERS:
        out[f"{m}_mean"] = float(res[f"test_{m}"].mean())
        out[f"{m}_std"] = float(res[f"test_{m}"].std())
        out[f"train_{m}_mean"] = float(res[f"train_{m}"].mean())
    out["fit_time_s"] = float(res["fit_time"].mean())
    return out


def cv_fold_scores(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold | None = None,
) -> pd.DataFrame:
    """Per-fold test scores (rows = folds, columns = metrics)."""
    cv = cv or make_cv()
    res = cross_validate(model, X, y, cv=cv, scoring=SCORERS)
    return pd.DataFrame({m: res[f"test_{m}"] for m in SCORERS})


def oof_predictions(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold | None = None,
) -> np.ndarray:
    """Out-of-fold class predictions: every row is predicted by a model that
    did not see it during fitting."""
    cv = cv or make_cv()
    return cross_val_predict(model, X, y, cv=cv)


def format_table(results: pd.DataFrame) -> pd.DataFrame:
    """Turn ``*_mean`` / ``*_std`` columns into 'mean +/- std' strings."""

    def fmt(mean: float, std: float) -> str:
        return f"{mean:.3f}" if pd.isna(std) else f"{mean:.3f} +/- {std:.3f}"

    return pd.DataFrame(
        {
            m: [
                fmt(results.loc[n, f"{m}_mean"], results.loc[n, f"{m}_std"])
                for n in results.index
            ]
            for m in SCORERS
        },
        index=results.index,
    )
