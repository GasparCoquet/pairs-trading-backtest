"""Market data loading.

Two entirely separate sources, never mixed:

* `fetch_prices`     -- real adjusted closes from Yahoo Finance. If the download
                        fails, this raises. It does NOT substitute anything.
* `synthetic_dataset` -- a labelled simulation with a KNOWN cointegration
                        structure, used only to validate the pair selector.

The original version of this module silently fell back to `generate_synthetic_prices`
whenever yfinance raised. That single `except` clause is how this repository ended
up publishing backtest statistics from manufactured data while claiming they came
from public equity prices. Synthetic data is now strictly opt-in.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class DataFetchError(RuntimeError):
    """Raised when real market data cannot be loaded.

    Deliberately fatal. There is no fallback path: a backtest on data we could not
    download is not a backtest.
    """


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    max_missing_frac: float = 0.05,
) -> pd.DataFrame:
    """Fetch real adjusted close prices from Yahoo Finance.

    Returns a DataFrame indexed by date with one column per surviving ticker.
    Tickers missing more than `max_missing_frac` of trading days are dropped.

    Raises:
        DataFetchError: if yfinance is not installed, the download fails, or the
            result is too sparse to backtest. Never returns synthetic data.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DataFetchError(
            "yfinance is not installed, so real prices cannot be downloaded. "
            "Install it (`pip install -r requirements.txt`), or run with --synthetic "
            "to exercise the simulated validation harness instead. Synthetic results "
            "are NOT market results."
        ) from exc

    try:
        raw = yf.download(
            tickers, start=start, end=end, auto_adjust=True, progress=False
        )
    except Exception as exc:
        raise DataFetchError(
            f"Yahoo Finance download failed for {tickers} ({start} to {end}): {exc}"
        ) from exc

    if raw is None or len(raw) == 0:
        raise DataFetchError(
            f"Yahoo Finance returned no rows for {tickers} ({start} to {end}). "
            "Check the tickers, the date range, and your network connection."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw.to_frame(name=tickers[0]) if len(tickers) == 1 else raw

    n_rows = len(prices)
    min_observations = n_rows - int(n_rows * max_missing_frac)
    prices = prices.dropna(axis=1, thresh=min_observations)
    prices = prices.ffill().dropna()

    dropped = sorted(set(tickers) - set(prices.columns))
    if dropped:
        print(f"[data] Dropped tickers with insufficient history: {dropped}")

    if len(prices) == 0 or len(prices.columns) < 2:
        raise DataFetchError(
            f"Yahoo Finance returned insufficient data: {len(prices)} rows, "
            f"{len(prices.columns)} usable tickers (need at least 2). "
            "Refusing to continue."
        )

    return prices


@dataclass
class SyntheticDataset:
    """Simulated prices together with the ground truth used to build them.

    This exists so the pair selector can be scored against a known answer key.
    It is a test fixture, not a data source.
    """

    prices: pd.DataFrame
    cointegrated_pairs: list[tuple[str, str]] = field(default_factory=list)
    independent_pairs: list[tuple[str, str]] = field(default_factory=list)

    @property
    def all_pairs(self) -> list[tuple[str, str]]:
        from itertools import combinations

        return [tuple(p) for p in combinations(self.prices.columns.tolist(), 2)]

    @property
    def null_pairs(self) -> list[tuple[str, str]]:
        """Every pair that is NOT cointegrated by construction.

        This is the answer key for false positives: any of these that the selector
        picks is a spurious pair.
        """
        truth = {frozenset(p) for p in self.cointegrated_pairs}
        return [p for p in self.all_pairs if frozenset(p) not in truth]


# (base_price, drift, vol, spread_mean_reversion, spread_vol)
# mean_reversion == 0 means the "spread" is itself a random walk, i.e. the two
# series are NOT cointegrated. That pair is the built-in negative control.
PAIR_CONFIGS = [
    (100, 0.0003, 0.015, 0.05, 0.01),   # cointegrated, fast mean reversion
    (50, 0.0002, 0.020, 0.03, 0.015),   # cointegrated, moderate
    (200, 0.0001, 0.012, 0.08, 0.008),  # cointegrated, fast
    (75, 0.0004, 0.025, 0.01, 0.020),   # cointegrated, slow (half-life ~69d)
    (150, 0.0005, 0.018, 0.0, 0.030),   # NOT cointegrated (random-walk spread)
]


def synthetic_dataset(
    tickers: list[str],
    start: str,
    end: str,
    seed: int = 42,
) -> SyntheticDataset:
    """Build simulated prices with a known cointegration structure.

    Consecutive ticker slots (0,1), (2,3), ... are generated from a shared
    stochastic trend plus an AR(1) spread. Where the AR(1) mean-reversion speed is
    non-zero the pair is cointegrated by construction; where it is zero the spread
    is a random walk and the pair is not. Every cross-pair -- e.g. slot 0 with slot
    3 -- is driven by an independent trend and is therefore not cointegrated either.

    The returned answer key lets `pairs_trading.validation` check that the selector
    recovers the true pairs and rejects the rest.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    prices: dict[str, np.ndarray] = {}
    cointegrated: list[tuple[str, str]] = []
    independent: list[tuple[str, str]] = []

    for i in range(0, len(tickers) - 1, 2):
        config_idx = (i // 2) % len(PAIR_CONFIGS)
        base_price, drift, vol, mr_speed, spread_vol = PAIR_CONFIGS[config_idx]

        common_shocks = rng.normal(drift, vol, n)
        common_log_price = np.cumsum(common_shocks)

        idio_a = rng.normal(0, vol * 0.3, n)
        prices[tickers[i]] = base_price * np.exp(common_log_price + idio_a)

        spread = np.zeros(n)
        for t in range(1, n):
            spread[t] = (1 - mr_speed) * spread[t - 1] + rng.normal(0, spread_vol)
        idio_b = rng.normal(0, vol * 0.3, n)
        prices[tickers[i + 1]] = base_price * np.exp(
            common_log_price + spread + idio_b
        )

        pair = (tickers[i], tickers[i + 1])
        if mr_speed > 0:
            cointegrated.append(pair)
        else:
            independent.append(pair)

    if len(tickers) % 2 == 1:
        last = tickers[-1]
        shocks = rng.normal(0.0003, 0.020, n)
        prices[last] = 120 * np.exp(np.cumsum(shocks))

    df = pd.DataFrame(prices, index=dates)
    return SyntheticDataset(
        prices=df,
        cointegrated_pairs=cointegrated,
        independent_pairs=independent,
    )


def generate_synthetic_prices(
    tickers: list[str],
    start: str,
    end: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulated prices only. Kept for backwards compatibility.

    WARNING: results computed on this data describe a simulation with an injected
    AR(1) mean-reverting spread. They say nothing about real markets. Nothing in
    the live code path calls this implicitly.
    """
    return synthetic_dataset(tickers, start, end, seed=seed).prices
