"""Simple expected-profit model for a churn-retention campaign."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class CampaignAssumptions:
    value: float
    cost: float
    success_rate: float

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError("value must be positive")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")
        if not 0 < self.success_rate <= 1:
            raise ValueError("success_rate must be in (0, 1]")


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    """Counts of true/false positives and negatives for 0/1 labels."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
    }

def expected_profit(tp: int, fp: int, a: CampaignAssumptions) -> float:
    """Profit of contacting every predicted churner (``tp`` + ``fp`` people)."""
    return tp * (a.success_rate * a.value - a.cost) - fp * a.cost

def break_even_precision(a: CampaignAssumptions) -> float:
    """Minimum precision for the campaign to make money."""
    return a.cost / (a.success_rate * a.value)
