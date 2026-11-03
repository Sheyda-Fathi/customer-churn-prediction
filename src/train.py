"""Train the final churn pipeline and save it with its metadata."""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from src.data import RAW_PATH, clean, get_split, load_raw, make_xy
from src.tuning import tuned_pipeline

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_FILE = "churn_pipeline.joblib"
META_FILE = "metadata.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }


def train_model(X: pd.DataFrame,y: pd.Series,name: str,params: dict,threshold: float,) -> tuple[Pipeline, dict]:
    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1")
    pipe = tuned_pipeline(name, params, X).fit(X, y)
    metadata = {
        "model": name,
        "params": params,
        "threshold": float(threshold),
        "feature_columns": list(X.columns),
        "n_train": int(len(X)),
        "churn_rate_train": round(float(np.mean(y)), 4),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": library_versions(),
    }
    return pipe, metadata


def save_model(pipe: Pipeline, metadata: dict, out_dir: Path = MODEL_DIR) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / MODEL_FILE)
    with open(out_dir / META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and save the churn pipeline.")
    parser.add_argument("--config", type=Path, default=REPORT_DIR / "final_model.json",help="JSON with model name, params and threshold (written by 03_modeling.ipynb)")
    parser.add_argument("--data", type=Path, default=RAW_PATH, help="raw Telco CSV")
    parser.add_argument("--out-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--use-all-data", action="store_true",
                        help="fit on every row instead of the train split")
    args = parser.parse_args(argv)

    if not args.config.exists():
        raise SystemExit(f"Config not found: {args.config}. Run 03_modeling.ipynb first.")
    if not args.data.exists():
        raise SystemExit(f"Data not found: {args.data}")

    config = json.loads(args.config.read_text(encoding="utf-8"))

    if args.use_all_data:
        X, y = make_xy(clean(load_raw(args.data)))
        trained_on = "all_data"
    else:
        X, _, y, _ = get_split(args.data)
        trained_on = "train_split"

    pipe, metadata = train_model(X, y, config["model"], config["params"], config["threshold"])
    metadata.update(
        trained_on=trained_on,
        data_sha256=file_sha256(args.data),
        assumptions=config.get("assumptions"),
        test_metrics=config.get("test") if trained_on == "train_split" else None,
    )
    out = save_model(pipe, metadata, args.out_dir)
    print(f"Saved {config['model']} (threshold {metadata['threshold']:.2f}, "
          f"trained on {trained_on}, {metadata['n_train']} rows) to {out}")


if __name__ == "__main__":
    main()
