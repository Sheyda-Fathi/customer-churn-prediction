import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from src.evaluate import make_cv
from src.importance import aggregate_to_original, cv_permutation_importance
from src.pipeline import build_pipeline

def _data(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "tenure": rng.normal(size=n),
            "MonthlyCharges": rng.normal(size=n),
            "Contract": rng.choice(["a", "b", "c"], n),
        }
    )
    y = pd.Series((X["tenure"] + 0.3 * rng.normal(size=n) > 0).astype(int))
    return X, y

def test_informative_feature_ranks_first():
    X, y = _data()
    pipe = build_pipeline(LogisticRegression(max_iter=1000), X)
    imp = cv_permutation_importance(pipe, X, y, cv=make_cv(3), n_repeats=3)
    assert imp.index[0] == "tenure"
    assert imp.loc["tenure", "mean"] > imp.loc["MonthlyCharges", "mean"]
    assert set(imp.columns) == {"mean", "std"}
    assert set(imp.index) == set(X.columns)

def test_aggregate_to_original_sums_one_hot_columns():
    s = pd.Series(
        {
            "num__tenure": 0.5,
            "cat__Contract_One year": 0.2,
            "cat__Contract_Two year": 0.1,
            "cat__Partner_Yes": 0.05,
        }
    )
    out = aggregate_to_original(s, ["tenure", "Contract", "Partner"])
    assert out["tenure"] == pytest.approx(0.5)
    assert out["Contract"] == pytest.approx(0.3)
    assert out["Partner"] == pytest.approx(0.05)

def test_aggregate_rejects_unknown_feature():
    with pytest.raises(ValueError):
        aggregate_to_original(pd.Series({"cat__Unknown_x": 1.0}), ["Contract"])
