"""Load the saved churn pipeline and score customers."""
import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from src.data import NUM_COLS
from src.train import META_FILE, MODEL_DIR, MODEL_FILE


class InputError(ValueError):
    """The customer data is missing fields or has invalid values."""

@dataclass(frozen=True)
class ChurnModel:
    pipeline: Pipeline
    metadata: dict

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    @property
    def feature_columns(self) -> list[str]:
        return list(self.metadata["feature_columns"])


def load_model(model_dir: Path = MODEL_DIR) -> ChurnModel:
    model_dir = Path(model_dir)
    model_path = model_dir / MODEL_FILE
    meta_path = model_dir / META_FILE
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"No saved model in {model_dir}. Run `python -m src.train` first." )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    trained_with = metadata.get("versions", {}).get("scikit-learn")
    if trained_with and trained_with != sklearn.__version__:
        warnings.warn(
            f"Model was trained with scikit-learn {trained_with}, "
            f"but {sklearn.__version__} is installed.",
            UserWarning,
            stacklevel=2,
        )
    return ChurnModel(pipeline=joblib.load(model_path), metadata=metadata)


def to_frame(records, required: list[str]) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        df = records.copy()
    elif isinstance(records, dict):
        df = pd.DataFrame([records])
    else:
        df = pd.DataFrame(list(records))
    if df.empty:
        raise InputError("No customers were provided.")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise InputError(f"Missing required fields: {', '.join(missing)}")
    df = df[required].copy()

    for col in (c for c in NUM_COLS if c in df.columns):
        raw = df[col]
        blank = raw.astype(str).str.strip().eq("")
        numeric = pd.to_numeric(raw, errors="coerce")
        invalid = numeric.isna() & raw.notna() & ~blank
        if invalid.any():
            raise InputError(f"Field {col!r} must be numeric.")
        df[col] = numeric
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df.loc[df["TotalCharges"].isna() & df["tenure"].eq(0), "TotalCharges"] = 0.0
    return df


def predict_proba(model: ChurnModel, records) -> np.ndarray:
    X = to_frame(records, model.feature_columns)
    return model.pipeline.predict_proba(X)[:, 1]


def predict(model: ChurnModel, records) -> list[dict]:
    proba = predict_proba(model, records)
    return [
        {
            "probability": round(float(p), 4),
            "churn": bool(p >= model.threshold),
            "threshold": model.threshold,
        }
        for p in proba
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score customers with the saved model.")
    parser.add_argument("--json", type=Path, required=True,help="JSON file with one customer or a list of customers")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args(argv)

    try:
        model = load_model(args.model_dir)
        records = json.loads(args.json.read_text(encoding="utf-8"))
        result = predict(model, records)
    except (FileNotFoundError, InputError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Error: {exc}")
    print(json.dumps(result[0] if isinstance(records, dict) else result, indent=2))


if __name__ == "__main__":
    main()
