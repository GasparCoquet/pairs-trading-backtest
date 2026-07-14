"""Tests for signal generation and the walk-forward accounting."""

import numpy as np
import pandas as pd
import pytest

from pairs_trading.backtest import walk_forward_backtest
from pairs_trading.data import synthetic_dataset
from pairs_trading.signals import generate_signals


def _z(values):
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))


def test_no_stop_loss_holds_a_diverging_spread():
    """Without a stop, a spread that never reverts is held indefinitely.

    This is the mechanism behind the drawdown: entry at z=-2, and the spread just
    keeps going. There is no exit condition that ever fires.
    """
    z = _z([0.0, -2.5, -3.5, -5.0, -8.0, -12.0])
    positions = generate_signals(z, entry_threshold=2.0, exit_threshold=0.0)
    assert positions.tolist() == [0, 1, 1, 1, 1, 1]


def test_stop_loss_cuts_a_diverging_spread():
    z = _z([0.0, -2.5, -3.5, -5.0, -8.0, -12.0])
    positions = generate_signals(
        z, entry_threshold=2.0, exit_threshold=0.0, stop_loss_z=3.0
    )
    # Enters at -2.5, stops out once z <= -3.0.
    assert positions.tolist() == [0, 1, 0, 0, 0, 0]


def test_stop_loss_does_not_fire_on_a_reverting_spread():
    z = _z([0.0, -2.5, -1.5, -0.4, 0.1])
    positions = generate_signals(
        z, entry_threshold=2.0, exit_threshold=0.0, stop_loss_z=3.0
    )
    assert positions.tolist() == [0, 1, 1, 1, 0]


def test_stop_loss_below_entry_is_rejected():
    with pytest.raises(ValueError):
        generate_signals(_z([0.0, 1.0]), entry_threshold=2.0, stop_loss_z=1.5)


def test_short_side_is_symmetric():
    z = _z([0.0, 2.5, 3.5])
    assert generate_signals(z, 2.0, 0.0).tolist() == [0, -1, -1]
    assert generate_signals(z, 2.0, 0.0, stop_loss_z=3.0).tolist() == [0, -1, 0]


def test_flat_windows_are_scored_as_zero_not_dropped():
    """Windows with no selected pair must still occupy days in the return series.

    Otherwise every annualised metric is computed over days-in-market rather than
    calendar time, which is what the original code did.
    """
    # Bonferroni on pure random walks should select almost nothing, forcing flat windows.
    rng = np.random.default_rng(5)
    prices = pd.DataFrame({
        f"T{i}": 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, 900)))
        for i in range(6)
    }, index=pd.bdate_range("2019-01-01", periods=900))

    results = walk_forward_backtest(
        prices, tickers=list(prices.columns),
        formation_period=252, trading_period=126,
        correction="bonferroni",
    )

    log = results["window_log"]
    n_flat = sum(1 for w in log if w["n_selected"] == 0)
    assert n_flat > 0, "test needs at least one flat window to be meaningful"

    # Every trading window contributes its days, flat or not.
    expected_days = sum(
        len(prices.loc[w["trading"][0]:w["trading"][1]]) for w in log
    )
    assert len(results["combined_returns"]) == expected_days
    assert results["combined_returns"].index.is_monotonic_increasing


def test_walk_forward_runs_on_labelled_synthetic_data():
    ds = synthetic_dataset(
        ["A", "B", "C", "D", "E", "F"], "2018-01-01", "2024-12-31", seed=42
    )
    results = walk_forward_backtest(
        ds.prices, tickers=list(ds.prices.columns),
        formation_period=252, trading_period=126,
    )
    assert not results["combined_equity"].empty
    assert len(results["window_log"]) > 0
