import pytest
from src.business import (
    CampaignAssumptions,
    break_even_precision,
    confusion_counts,
    expected_profit,
)

A = CampaignAssumptions(value=500, cost=50, success_rate=0.3)


def test_expected_profit_formula():
    # 10 * (0.3 * 500 - 50) - 5 * 50 = 750
    assert expected_profit(10, 5, A) == pytest.approx(750)


def test_break_even_precision_gives_zero_profit():
    assert break_even_precision(A) == pytest.approx(1 / 3)
    assert expected_profit(tp=1, fp=2, a=A) == pytest.approx(0)


def test_confusion_counts():
    c = confusion_counts([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
    assert c == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}


def test_invalid_assumptions_raise():
    with pytest.raises(ValueError):
        CampaignAssumptions(value=0, cost=10, success_rate=0.5)
    with pytest.raises(ValueError):
        CampaignAssumptions(value=100, cost=-1, success_rate=0.5)
    with pytest.raises(ValueError):
        CampaignAssumptions(value=100, cost=10, success_rate=1.5)
