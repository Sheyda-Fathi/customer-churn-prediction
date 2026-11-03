import numpy as np
import pytest
from src.business import CampaignAssumptions
from src.threshold import best_threshold, threshold_table

A = CampaignAssumptions(value=500, cost=50, success_rate=0.3)


def test_threshold_table_counts_and_metrics():
    y = [1, 1, 0, 0]
    p = [0.9, 0.4, 0.6, 0.1]
    t = threshold_table(y, p, A, thresholds=[0.5]).iloc[0]
    assert (t.tp, t.fp, t.fn, t.tn) == (1, 1, 1, 1)
    assert t.precision == pytest.approx(0.5)
    assert t.recall == pytest.approx(0.5)
    assert t.f1 == pytest.approx(0.5)
    # (1*(0.3*500-50) - 1*50) / 4 * 1000 = 12500
    assert t.profit_per_1000 == pytest.approx(12500)


def test_lower_threshold_flags_more_customers():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)
    tab = threshold_table(y, p, A, thresholds=[0.2, 0.5, 0.8])
    flagged = (tab.tp + tab.fp).tolist()
    assert flagged == sorted(flagged, reverse=True)
    assert tab.recall.is_monotonic_decreasing


def test_best_threshold_maximizes_profit():
    rng = np.random.default_rng(1)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    tab = threshold_table(y, p, A)
    best = best_threshold(tab)
    assert best.profit_per_1000 == tab.profit_per_1000.max()


def test_empty_prediction_has_nan_precision_and_zero_profit():
    tab = threshold_table([0, 1, 0, 1], [0.1, 0.2, 0.1, 0.3], A, thresholds=[0.99])
    row = tab.iloc[0]
    assert np.isnan(row.precision)
    assert row.profit_per_1000 == 0
