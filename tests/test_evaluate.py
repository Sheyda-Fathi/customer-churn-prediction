from sklearn.linear_model import LogisticRegression
from src.evaluate import SCORERS, cv_fold_scores, make_cv, oof_predictions
from src.pipeline import build_pipeline
from tests.helpers import make_data

def test_cv_fold_scores_shape_and_columns():
    X, y = make_data()
    pipe = build_pipeline(LogisticRegression(max_iter=1000), X)
    out = cv_fold_scores(pipe, X, y, cv=make_cv(3))
    assert out.shape == (3, len(SCORERS))
    assert list(out.columns) == list(SCORERS)

def test_oof_predictions_cover_every_row_with_binary_labels():
    X, y = make_data()
    pipe = build_pipeline(LogisticRegression(max_iter=1000), X)
    pred = oof_predictions(pipe, X, y, cv=make_cv(3))
    assert len(pred) == len(y)
    assert set(pred) <= {0, 1}
