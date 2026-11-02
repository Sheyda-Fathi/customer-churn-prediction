import numpy as np
from src.evaluate import cv_scores, make_cv
from src.models import build_model_zoo
from tests.helpers import make_data


def test_zoo_has_each_model_with_and_without_class_weight():
    X, _ = make_data()
    zoo = build_model_zoo(X)
    bases = {"LogReg", "RandomForest", "SVM-RBF", "HistGB"}
    assert set(zoo) == bases | {f"{b} (balanced)" for b in bases}
    for name, pipe in zoo.items():
        weight = pipe.named_steps["model"].class_weight
        assert (weight == "balanced") == name.endswith("(balanced)")


def test_every_model_fits_and_scores():
    X, y = make_data(n=240)
    for name, pipe in build_model_zoo(X).items():
        pipe.fit(X, y)
        assert len(pipe.predict(X)) == len(X), name
        scores = cv_scores(pipe, X, y, cv=make_cv(3))
        assert 0 <= scores["roc_auc_mean"] <= 1, name
        assert np.isfinite(scores["f1_mean"]), name


def test_zoo_is_reproducible():
    X, y = make_data(n=240)
    rf_a = build_model_zoo(X)["RandomForest"].fit(X, y).predict_proba(X)
    rf_b = build_model_zoo(X)["RandomForest"].fit(X, y).predict_proba(X)
    assert np.allclose(rf_a, rf_b)
