"""Tests for the statistics that the original version got wrong."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from pairs_trading.cointegration import (
    benjamini_hochberg,
    bonferroni,
    select_pairs,
)
from pairs_trading.data import DataFetchError, fetch_prices, synthetic_dataset


# --- multiple-testing corrections -------------------------------------------------

def test_bh_matches_statsmodels_on_random_inputs():
    """My BH implementation must agree with a reference implementation exactly."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        m = int(rng.integers(1, 60))
        p = rng.random(m) ** int(rng.choice([1, 3]))
        alpha = float(rng.choice([0.01, 0.05, 0.10]))
        mine = benjamini_hochberg(list(p), alpha)
        reference = multipletests(p, alpha=alpha, method="fdr_bh")[0]
        assert np.array_equal(mine, reference)


def test_bonferroni_matches_statsmodels_on_random_inputs():
    rng = np.random.default_rng(1)
    for _ in range(300):
        m = int(rng.integers(1, 60))
        p = rng.random(m)
        alpha = float(rng.choice([0.01, 0.05, 0.10]))
        mine = bonferroni(list(p), alpha)
        reference = multipletests(p, alpha=alpha, method="bonferroni")[0]
        assert np.array_equal(mine, reference)


def test_bh_is_never_stricter_than_bonferroni():
    rng = np.random.default_rng(2)
    for _ in range(200):
        p = rng.random(45)
        assert benjamini_hochberg(list(p), 0.05).sum() >= bonferroni(list(p), 0.05).sum()


def test_corrections_reject_nothing_when_all_pvalues_are_large():
    p = [0.4, 0.6, 0.9, 0.99]
    assert not benjamini_hochberg(p, 0.05).any()
    assert not bonferroni(p, 0.05).any()


def test_bh_handles_empty_input():
    assert len(benjamini_hochberg([], 0.05)) == 0
    assert len(bonferroni([], 0.05)) == 0


# --- the multiple-testing bug itself ----------------------------------------------

def test_uncorrected_selection_fires_on_pure_noise():
    """45 independent random walks, no cointegration anywhere by construction.

    Uncorrected selection at alpha=0.05 should still find 'cointegrated' pairs.
    This is the bug: it is a false-positive generator, and the backtest fed on it.
    """
    rng = np.random.default_rng(11)
    n_universes = 12
    uncorrected_hits = 0
    bh_hits = 0

    for _ in range(n_universes):
        prices = pd.DataFrame({
            f"T{i}": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, 300)))
            for i in range(10)
        }, index=pd.bdate_range("2020-01-01", periods=300))

        uncorrected_hits += len(
            select_pairs(prices, max_pairs=45, correction="none")
        )
        bh_hits += len(select_pairs(prices, max_pairs=45, correction="bh"))

    # These are pure random walks. Every single hit is a false positive.
    assert uncorrected_hits > 0, "expected the uncorrected selector to fire on noise"
    assert bh_hits <= uncorrected_hits, "BH must not select more than uncorrected"


def test_selection_is_reproducible():
    ds = synthetic_dataset(["A", "B", "C", "D"], "2020-01-01", "2023-01-01", seed=3)
    first = select_pairs(ds.prices, max_pairs=6)
    second = select_pairs(ds.prices, max_pairs=6)
    assert first == second


def test_unknown_correction_and_criterion_are_rejected():
    ds = synthetic_dataset(["A", "B", "C", "D"], "2020-01-01", "2022-01-01", seed=3)
    with pytest.raises(ValueError):
        select_pairs(ds.prices, correction="fdr")
    with pytest.raises(ValueError):
        select_pairs(ds.prices, criterion="adf")


# --- the silent fallback ----------------------------------------------------------

def test_fetch_prices_raises_instead_of_returning_synthetic_data():
    """The whole point. A failed download must be fatal, not quietly simulated."""
    with pytest.raises(DataFetchError):
        fetch_prices(["__NOT_A_REAL_TICKER__", "__ALSO_NOT_REAL__"],
                     start="2020-01-01", end="2020-06-01")


def test_synthetic_dataset_labels_its_own_ground_truth():
    ds = synthetic_dataset(
        ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        "2018-01-01", "2024-12-31", seed=42,
    )
    # Slots (0,1)...(6,7) are cointegrated by construction; (8,9) is a random walk.
    assert ("A", "B") in ds.cointegrated_pairs
    assert ("I", "J") in ds.independent_pairs
    assert ("I", "J") not in ds.cointegrated_pairs
    # The answer key must cover every pair that is not cointegrated.
    assert len(ds.all_pairs) == 45
    assert len(ds.null_pairs) == 45 - len(ds.cointegrated_pairs)
