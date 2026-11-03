"""Choosing a decision threshold by expected campaign profit."""
import numpy as np
import pandas as pd
from src.business import CampaignAssumptions, expected_profit


def threshold_table(y_true,proba,assumptions: CampaignAssumptions,thresholds=None,) -> pd.DataFrame:
    """Confusion counts, precision/recall/F1 and profit for each threshold."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)

    rows = []
    for t in thresholds:
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        tn = int((~pred & (y == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
        rows.append(
            {
                "threshold": float(t),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "profit_per_1000": expected_profit(tp, fp, assumptions) / len(y) * 1000,
            }
        )
    return pd.DataFrame(rows)


def best_threshold(table: pd.DataFrame) -> pd.Series:
    """Row of ``threshold_table`` with the highest profit (lowest threshold on ties)."""
    return table.loc[table["profit_per_1000"].idxmax()]
