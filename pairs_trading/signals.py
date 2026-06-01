"""Z-score based trading signal generation."""

import numpy as np
import pandas as pd


def compute_zscore(
    spread: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """Compute rolling z-score of the spread."""
    mean = spread.rolling(window=lookback).mean()
    std = spread.rolling(window=lookback).std(ddof=0)
    zscore = (spread - mean) / std
    zscore = zscore.replace([np.inf, -np.inf], np.nan)
    return zscore


def generate_signals(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
) -> pd.Series:
    """Generate trading signals from z-score.

    Signal conventions:
        +1 = long spread (long A, short B) — z-score crossed below -entry_threshold
        -1 = short spread (short A, long B) — z-score crossed above +entry_threshold
         0 = flat (z-score returned within exit_threshold of zero)

    Returns a Series of positions: +1, -1, or 0.
    """
    positions = pd.Series(0, index=zscore.index, dtype=int)
    position = 0

    for i in range(len(zscore)):
        z = zscore.iloc[i]
        if np.isnan(z):
            positions.iloc[i] = 0
            continue

        if position == 0:
            if z <= -entry_threshold:
                position = 1  # spread is low — long spread
            elif z >= entry_threshold:
                position = -1  # spread is high — short spread
        elif position == 1:
            if z >= -exit_threshold:
                position = 0
        elif position == -1:
            if z <= exit_threshold:
                position = 0

        positions.iloc[i] = position

    return positions
