import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from src.evaluate import SCORERS, cv_scores, make_cv
from src.pipeline import build_pipeline, build_preprocessor, split_columns

def _data(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n)
    monthly = rng.uniform(20, 120, n).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    X = pd.DataFrame(
        {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": tenure * monthly,
            "Contract": contract,
            "PaymentMethod": rng.choice(["Electronic check", "Mailed check"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
        }
    )
    logit = 1.5 * (contract == "Month-to-month") - 0.04 * tenure - 0.5
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y


def test_split_columns_treats_senior_citizen_as_categorical():
    X, _ = _data()
    num, cat = split_columns(X)
    assert num == ["tenure", "MonthlyCharges", "TotalCharges"]
    assert "SeniorCitizen" in cat and "Contract" in cat


def test_preprocessor_learns_statistics_from_train_only():
    X, _ = _data()
    X_tr, X_te = X.iloc[:200], X.iloc[200:]
    pre = build_preprocessor(X).fit(X_tr)
    tr_out, te_out = pre.transform(X_tr), pre.transform(X_te)
    assert abs(tr_out[:, 0].mean()) < 1e-8
    assert abs(te_out[:, 0].mean()) > 1e-8
    assert not np.isnan(tr_out).any()


def test_pipeline_predicts_probabilities_in_unit_interval():
    X, y = _data()
    pipe = build_pipeline(LogisticRegression(max_iter=1000), X).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.min() >= 0 and proba.max() <= 1


def test_pipeline_handles_unknown_category_and_missing_value():
    X, y = _data()
    pipe = build_pipeline(LogisticRegression(max_iter=1000), X).fit(X, y)
    row = X.iloc[[0]].copy()
    row["Contract"] = "Weekly"
    row["MonthlyCharges"] = np.nan
    proba = pipe.predict_proba(row)[:, 1]
    assert 0 <= proba[0] <= 1


def test_cv_scores_returns_all_metrics():
    X, y = _data()
    pipe = build_pipeline(LogisticRegression(max_iter=1000), X)
    out = cv_scores(pipe, X, y, cv=make_cv(3))
    for m in SCORERS:
        assert f"{m}_mean" in out and f"train_{m}_mean" in out
    assert 0 <= out["roc_auc_mean"] <= 1
