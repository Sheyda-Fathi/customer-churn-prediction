import numpy as np
from sklearn.linear_model import LogisticRegression
from src.evaluate import SCORERS, cv_scores, make_cv
from src.pipeline import build_pipeline, build_preprocessor, split_columns
from tests.helpers import make_data


def test_split_columns_treats_senior_citizen_as_categorical():
    X, _ = make_data()
    num, cat = split_columns(X)
    assert num == ["tenure", "MonthlyCharges", "TotalCharges"]
    assert "SeniorCitizen" in cat and "Contract" in cat


def test_preprocessor_learns_statistics_from_train_only():
    X, _ = make_data()
    X_tr, X_te = X.iloc[:200], X.iloc[200:]
    pre = build_preprocessor(X).fit(X_tr)
    tr_out, te_out = pre.transform(X_tr), pre.transform(X_te)
    assert abs(tr_out[:, 0].mean()) < 1e-8
    assert abs(te_out[:, 0].mean()) > 1e-8
    assert not np.isnan(tr_out).any()


def test_pipeline_predicts_probabilities_in_unit_interval():
    X, y = make_data()
    pipe = build_pipeline(LogisticRegression(max_iter=1000), X).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.min() >= 0 and proba.max() <= 1


def test_pipeline_handles_unknown_category_and_missing_value():
    X, y = make_data()
    pipe = build_pipeline(LogisticRegression(max_iter=1000), X).fit(X, y)
    row = X.iloc[[0]].copy()
    row["Contract"] = "Weekly"
    row["MonthlyCharges"] = np.nan
    proba = pipe.predict_proba(row)[:, 1]
    assert 0 <= proba[0] <= 1


def test_cv_scores_returns_all_metrics():
    X, y = make_data()
    pipe = build_pipeline(LogisticRegression(max_iter=1000), X)
    out = cv_scores(pipe, X, y, cv=make_cv(3))
    for m in SCORERS:
        assert f"{m}_mean" in out and f"train_{m}_mean" in out
    assert 0 <= out["roc_auc_mean"] <= 1
