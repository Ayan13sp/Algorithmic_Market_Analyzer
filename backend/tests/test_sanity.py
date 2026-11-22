"""Every backtest sanity check from experiments/backtest_sanity.py as a test case
(the same checks run as a dedicated CI job that publishes the results table)."""

import pytest

from experiments.backtest_sanity import CHECKS


@pytest.mark.parametrize("check", CHECKS, ids=[c.__name__ for c in CHECKS])
def test_sanity_check(check):
    result = check()
    assert result.passed, f"{result.name}: expected {result.claim}; observed {result.observed}"
