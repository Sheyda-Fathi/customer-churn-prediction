import pandas as pd
import pytest
from src.data import clean, make_xy, split_xy

def _frame(n: int = 200) -> pd.DataFrame:
    """Small synthetic frame with the same columns the code relies on."""
    df = pd.DataFrame(
        {
            "customerID": [f"id-{i}" for i in range(n)],
            "tenure": [i % 72 for i in range(n)],
            "MonthlyCharges": [20.0 + (i % 50) for i in range(n)],
            "Contract": ["Month-to-month" if i % 2 else "Two year" for i in range(n)],
            "Churn": ["Yes" if i % 4 == 0 else "No" for i in range(n)],
        }
    )
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).astype(str)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "
    return df


def test_clean_fills_zero_for_new_customers():
    out = clean(_frame())
    assert out["TotalCharges"].dtype.kind == "f"
    assert out.loc[out["tenure"] == 0, "TotalCharges"].eq(0.0).all()
    assert out["TotalCharges"].notna().all()


def test_clean_raises_on_unexpected_blank():
    df = _frame()
    df.loc[df["customerID"] == "id-5", "TotalCharges"] = " "
    with pytest.raises(ValueError):
        clean(df)


def test_clean_does_not_modify_input():
    df = _frame()
    before = df.copy()
    clean(df)
    pd.testing.assert_frame_equal(df, before)


def test_make_xy_drops_id_and_encodes_target():
    X, y = make_xy(clean(_frame()))
    assert "customerID" not in X.columns and "Churn" not in X.columns
    assert set(y.unique()) == {0, 1}
    assert len(X) == len(y)


def test_make_xy_rejects_unknown_target_value():
    df = clean(_frame())
    df.loc[0, "Churn"] = "Maybe"
    with pytest.raises(ValueError):
        make_xy(df)


def test_split_is_stratified_and_reproducible():
    X, y = make_xy(clean(_frame()))
    X_tr, X_te, y_tr, y_te = split_xy(X, y)
    assert len(X_te) == pytest.approx(0.2 * len(X), abs=1)
    assert y_tr.mean() == pytest.approx(y_te.mean(), abs=0.02)
    X_tr2, X_te2, _, _ = split_xy(X, y)
    assert X_tr.index.equals(X_tr2.index) and X_te.index.equals(X_te2.index)
    assert set(X_tr.index).isdisjoint(X_te.index)
