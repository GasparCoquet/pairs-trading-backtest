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
    stop_loss_z: float | None = None,
) -> pd.Series:
    """Generate trading signals from z-score.

    Signal conventions:
        +1 = long spread (long A, short B) — z-score crossed below -entry_threshold
        -1 = short spread (short A, long B) — z-score crossed above +entry_threshold
         0 = flat (z-score returned within exit_threshold of zero)

    `stop_loss_z`, when set, closes the position as soon as |z| exceeds it. The
    strategy's core assumption is that a stretched spread snaps back; a spread that
    keeps stretching is evidence against the assumption that got us into the trade,
    not a reason to size into it. Without a stop there is no state in which this
    strategy ever concedes a pair has stopped mean-reverting, and it will hold a
    de-cointegrated spread indefinitely.

    Once a pair stops out it is retired for the remainder of the trading window. This
    matters: a stop that only flattens the position would re-enter on the very next
    bar, because |z| beyond the stop is also beyond the entry threshold. That is not a
    stop-loss, it is a round-trip cost applied to the same losing trade. The pair gets
    re-tested from scratch at the next formation window.

    Returns a Series of positions: +1, -1, or 0.
    """
    if stop_loss_z is not None and stop_loss_z <= entry_threshold:
        raise ValueError(
            f"stop_loss_z ({stop_loss_z}) must exceed entry_threshold "
            f"({entry_threshold}), otherwise every entry stops out immediately."
        )

    positions = pd.Series(0, index=zscore.index, dtype=int)
    position = 0
    retired = False  # stopped out: no further entries this window

    for i in range(len(zscore)):
        z = zscore.iloc[i]
        if np.isnan(z) or retired:
            positions.iloc[i] = 0
            continue

        if position == 0:
            if z <= -entry_threshold:
                position = 1  # spread is low — long spread
            elif z >= entry_threshold:
                position = -1  # spread is high — short spread
        elif position == 1:
            if stop_loss_z is not None and z <= -stop_loss_z:
                position = 0  # spread kept diverging — cut it and stand down
                retired = True
            elif z >= -exit_threshold:
                position = 0
        elif position == -1:
            if stop_loss_z is not None and z >= stop_loss_z:
                position = 0  # spread kept diverging — cut it and stand down
                retired = True
            elif z <= exit_threshold:
                position = 0

        positions.iloc[i] = position

    return positions
