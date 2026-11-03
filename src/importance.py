"""Permutation importance computed on held-out folds."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from src.data import RANDOM_STATE
from src.evaluate import make_cv


def cv_permutation_importance(model: BaseEstimator,X: pd.DataFrame,y: pd.Series,cv: StratifiedKFold | None = None,scoring: str = "average_precision",n_repeats: int = 5,random_state: int = RANDOM_STATE,) -> pd.DataFrame:
    """Permutation importance of each *original* column, averaged over CV folds."""
    cv = cv or make_cv()
    per_fold = []
    for train_idx, val_idx in cv.split(X, y):
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        res = permutation_importance(fitted,X.iloc[val_idx],y.iloc[val_idx],scoring=scoring,n_repeats=n_repeats, random_state=random_state,)
        per_fold.append(res.importances_mean)
    arr = np.vstack(per_fold)
    out = pd.DataFrame({"mean": arr.mean(axis=0), "std": arr.std(axis=0, ddof=1)},index=X.columns,)
    return out.sort_values("mean", ascending=False)


def aggregate_to_original(importances: pd.Series, columns: list[str]) -> pd.Series:
    """Sum one-hot/numeric feature importances back to the original columns. """
    totals = {c: 0.0 for c in columns}
    for name, value in importances.items():
        for col in columns:
            if name == f"num__{col}" or name.startswith(f"cat__{col}_") or name == f"cat__{col}":
                totals[col] += float(value)
                break
        else:
            raise ValueError(f"Cannot map feature {name!r} to an original column.")
    return pd.Series(totals).sort_values(ascending=False)
