"""Data loading, cleaning and splitting for the Telco churn project."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
TARGET: str = "Churn"
ID_COL: str = "customerID"
NUM_COLS: list[str] = ["tenure", "MonthlyCharges", "TotalCharges"]
RAW_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw Telco CSV."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the fixed cleaning rules """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    new_customer = out["tenure"].eq(0)
    out.loc[out["TotalCharges"].isna() & new_customer, "TotalCharges"] = 0.0
    if out["TotalCharges"].isna().any():
        raise ValueError("Unexpected missing TotalCharges for customers with tenure > 0.")
    return out


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned frame into features X and a 0/1 target y."""
    y = df[TARGET].map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Churn contains values other than 'Yes'/'No'.")
    X = df.drop(columns=[TARGET, ID_COL])
    return X, y.astype(int)


def split_xy(X: pd.DataFrame,y: pd.Series,test_size: float = TEST_SIZE,random_state: int = RANDOM_STATE,) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified, reproducible train/test split."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def get_split(path: Path = RAW_PATH,test_size: float = TEST_SIZE,random_state: int = RANDOM_STATE,) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, clean and split the data: X_train, X_test, y_train, y_test."""
    X, y = make_xy(clean(load_raw(path)))
    return split_xy(X, y, test_size=test_size, random_state=random_state)
