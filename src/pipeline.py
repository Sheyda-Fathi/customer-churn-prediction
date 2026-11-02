"""Preprocessing pipeline: imputation, scaling and one-hot encoding."""
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data import NUM_COLS


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_columns, categorical_columns) for the Telco features."""
    num_cols = [c for c in NUM_COLS if c in X.columns]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Median-impute + scale numerics; mode-impute + one-hot encode categoricals."""
    num_cols, cat_cols = split_columns(X)

    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore", drop="if_binary", sparse_output=False
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, num_cols), ("cat", categorical, cat_cols)]
    )


def build_pipeline(model: BaseEstimator, X: pd.DataFrame) -> Pipeline:
    """Preprocessor followed by a model, as one estimator."""
    return Pipeline([("pre", build_preprocessor(X)), ("model", model)])
