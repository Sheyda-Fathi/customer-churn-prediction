import contextlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.data import RAW_PATH, get_split
from src.predict import InputError, load_model, predict, predict_proba, to_frame
from src.predict import main as predict_main
from src.train import (META_FILE,MODEL_FILE,file_sha256,save_model,train_model,)
from src.train import main as train_main
from tests.helpers import make_data

PARAMS = {"model__C": 1.0, "model__class_weight": None}
ROOT = Path(__file__).resolve().parents[1]
FINAL_CONFIG = ROOT / "reports" / "final_model.json"


def _saved(tmp_path):
    """Train a small model, save it and load it back."""
    X, y = make_data()
    pipe, meta = train_model(X, y, "LogReg", PARAMS, threshold=0.33)
    save_model(pipe, meta, tmp_path)
    return X, pipe, load_model(tmp_path)


def test_probabilities_are_between_zero_and_one(tmp_path):
    X, _, model = _saved(tmp_path)
    proba = predict_proba(model, X)
    assert len(proba) == len(X)
    assert proba.min() >= 0 and proba.max() <= 1


def test_missing_column_gives_clear_error(tmp_path):
    X, _, model = _saved(tmp_path)
    row = X.iloc[0].to_dict()
    del row["Contract"]
    with pytest.raises(InputError) as err:
        predict(model, row)
    assert "Contract" in str(err.value)


def test_saved_model_reproduces_in_memory_predictions(tmp_path):
    X, pipe, model = _saved(tmp_path)
    expected = pipe.predict_proba(X)[:, 1]
    assert np.allclose(expected, predict_proba(model, X))


def test_metadata_has_required_fields(tmp_path):
    X, _, model = _saved(tmp_path)
    meta = json.loads((tmp_path / META_FILE).read_text(encoding="utf-8"))
    assert (tmp_path / MODEL_FILE).exists()
    for key in ("model", "params", "threshold", "feature_columns", "n_train","churn_rate_train", "created_at", "versions"):
        assert key in meta
    assert meta["threshold"] == pytest.approx(0.33)
    assert meta["feature_columns"] == list(X.columns)
    assert model.threshold == pytest.approx(0.33)


def test_decision_follows_threshold(tmp_path):
    X, _, model = _saved(tmp_path)
    out = predict(model, X.head(40))
    proba = predict_proba(model, X.head(40))
    assert [r["churn"] for r in out] == list(proba >= model.threshold)
    assert all(r["threshold"] == model.threshold for r in out)


def test_single_dict_is_accepted_and_extra_fields_ignored(tmp_path):
    X, _, model = _saved(tmp_path)
    row = X.iloc[0].to_dict()
    row["customerID"] = "7590-VHVEG"
    out = predict(model, row)
    assert len(out) == 1 and 0 <= out[0]["probability"] <= 1


def test_blank_total_charges_for_new_customer_becomes_zero():
    X, _ = make_data()
    row = X.iloc[0].to_dict()
    row.update(tenure=0, TotalCharges=" ")
    df = to_frame(row, list(X.columns))
    assert df.loc[0, "TotalCharges"] == 0.0


def test_non_numeric_value_is_rejected():
    X, _ = make_data()
    row = X.iloc[0].to_dict()
    row["tenure"] = "abc"
    with pytest.raises(InputError):
        to_frame(row, list(X.columns))


def test_empty_input_is_rejected():
    X, _ = make_data()
    with pytest.raises(InputError):
        to_frame([], list(X.columns))


def test_train_model_validates_threshold_and_name():
    X, y = make_data()
    with pytest.raises(ValueError):
        train_model(X, y, "LogReg", PARAMS, threshold=1.5)
    with pytest.raises(ValueError):
        train_model(X, y, "Nope", PARAMS, threshold=0.5)


def test_load_model_without_files_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path)


def test_file_sha256_is_deterministic(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("hello")
    b.write_text("hello")
    assert file_sha256(a) == file_sha256(b)
    b.write_text("hello!")
    assert file_sha256(a) != file_sha256(b)


def test_cli_scores_the_sample_customer(tmp_path):
    _saved(tmp_path)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        predict_main(["--json", str(ROOT / "examples" / "sample_customer.json"),
                      "--model-dir", str(tmp_path)])
    out = json.loads(buffer.getvalue())
    assert set(out) == {"probability", "churn", "threshold"}


@pytest.mark.skipif(not (RAW_PATH.exists() and FINAL_CONFIG.exists()),reason="needs the raw data and reports/final_model.json (run 03_modeling.ipynb)",)
def test_saved_model_reproduces_the_notebook_test_scores(tmp_path):
    from sklearn.metrics import (average_precision_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)

    train_main(["--out-dir", str(tmp_path)])
    model = load_model(tmp_path)
    _, X_test, _, y_test = get_split()
    proba = predict_proba(model, X_test)
    pred = (proba >= model.threshold).astype(int)

    expected = json.loads(FINAL_CONFIG.read_text(encoding="utf-8"))["test"]
    got = {
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }
    for key, value in got.items():
        assert value == pytest.approx(expected[key], abs=1e-3), key
