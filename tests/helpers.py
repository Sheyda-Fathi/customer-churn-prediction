"""Shared synthetic data for tests."""
import numpy as np
import pandas as pd

def make_data(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n)
    monthly = rng.uniform(20, 120, n).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    X = pd.DataFrame(
        {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": tenure * monthly,
            "Contract": contract,
            "PaymentMethod": rng.choice(["Electronic check", "Mailed check"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
        }
    )
    logit = 1.5 * (contract == "Month-to-month") - 0.04 * tenure - 0.5
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y
