"""Candidate models, all wrapped in the same preprocessing pipeline."""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from src.data import RANDOM_STATE
from src.pipeline import build_pipeline


def build_model_zoo(
    X: pd.DataFrame, random_state: int = RANDOM_STATE
) -> dict[str, Pipeline]:
    """Default-hyperparameter pipelines, each with and without class weighting."""
    specs = {
        "LogReg": LogisticRegression(max_iter=1000, random_state=random_state),
        "LogReg (balanced)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "RandomForest (balanced)": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "SVM-RBF": SVC(kernel="rbf", random_state=random_state),
        "SVM-RBF (balanced)": SVC(
            kernel="rbf", class_weight="balanced", random_state=random_state
        ),
        "HistGB": HistGradientBoostingClassifier(random_state=random_state),
        "HistGB (balanced)": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=random_state
        ),
    }
    return {name: build_pipeline(model, X) for name, model in specs.items()}
