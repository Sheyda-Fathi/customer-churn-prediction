import pytest
from src.evaluate import SCORERS, make_cv
from src.tuning import MAIN_METRIC, make_search, tuned_pipeline
from tests.helpers import make_data

MODEL_NAMES = ["LogReg", "RandomForest", "HistGB"]

def _space(search) -> dict:
    return getattr(search, "param_grid", None) or search.param_distributions

def test_main_metric_is_a_known_scorer():
    assert MAIN_METRIC in SCORERS

@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_search_parameter_exists_on_the_pipeline(name):
    X, _ = make_data()
    search = make_search(name, X)
    valid = search.estimator.get_params()
    assert set(_space(search)) <= set(valid)

def test_logreg_search_runs_and_refits():
    X, y = make_data()
    search = make_search("LogReg", X, cv=make_cv(3)).fit(X, y)
    assert set(search.best_params_) == {"model__C", "model__class_weight"}
    assert 0 <= search.best_score_ <= 1
    assert "mean_test_f1" in search.cv_results_
    proba = search.best_estimator_.predict_proba(X)[:, 1]
    assert proba.min() >= 0 and proba.max() <= 1

def test_tuned_pipeline_applies_params():
    X, _ = make_data()
    pipe = tuned_pipeline("RandomForest", {"model__min_samples_leaf": 7}, X)
    assert pipe.get_params()["model__min_samples_leaf"] == 7

def test_unknown_model_name_raises():
    X, _ = make_data()
    with pytest.raises(ValueError):
        make_search("Nope", X)
    with pytest.raises(ValueError):
        tuned_pipeline("Nope", {}, X)
