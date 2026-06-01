"""Data fetching and preprocessing from Yahoo Finance, with synthetic fallback."""

import numpy as np
import pandas as pd


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch adjusted close prices for a list of tickers from Yahoo Finance.

    Returns a DataFrame with dates as index and tickers as columns.
    Drops any ticker that has missing data for more than 5% of trading days.
    Falls back to synthetic cointegrated data if yfinance is unavailable or
    network access is restricted.
    """
    try:
        import yfinance as yf

        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw.to_frame(name=tickers[0]) if len(tickers) == 1 else raw

        # Drop tickers with too many missing values
        threshold = len(prices) * 0.05
        prices = prices.dropna(axis=1, thresh=len(prices) - int(threshold))
        prices = prices.ffill().dropna()

        if len(prices) > 0 and len(prices.columns) >= 2:
            return prices

        print("[WARN] Yahoo Finance returned insufficient data. Using synthetic data.")
    except Exception as e:
        print(f"[WARN] Yahoo Finance unavailable ({e}). Using synthetic data.")

    return generate_synthetic_prices(tickers, start, end)


def generate_synthetic_prices(
    tickers: list[str],
    start: str,
    end: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic price data with realistic cointegration relationships.

    Creates pairs of cointegrated series (with mean-reverting spreads) along
    with some non-cointegrated series for realistic pair selection.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    prices = {}
    pair_configs = [
        # (base_price, drift, vol, spread_mean_reversion, spread_vol)
        (100, 0.0003, 0.015, 0.05, 0.01),   # tight cointegration
        (50, 0.0002, 0.020, 0.03, 0.015),    # moderate cointegration
        (200, 0.0001, 0.012, 0.08, 0.008),   # tight cointegration
        (75, 0.0004, 0.025, 0.01, 0.020),    # loose cointegration
        (150, 0.0005, 0.018, 0.0, 0.030),    # independent (no cointegration)
    ]

    for i in range(0, len(tickers) - 1, 2):
        config_idx = (i // 2) % len(pair_configs)
        base_price, drift, vol, mr_speed, spread_vol = pair_configs[config_idx]

        # Generate a common stochastic trend (random walk)
        common_shocks = rng.normal(drift, vol, n)
        common_log_price = np.cumsum(common_shocks)

        # Ticker A follows the common trend + idiosyncratic noise
        idio_a = rng.normal(0, vol * 0.3, n)
        log_price_a = common_log_price + idio_a
        prices[tickers[i]] = base_price * np.exp(log_price_a)

        # Ticker B follows the common trend + mean-reverting spread
        spread = np.zeros(n)
        for t in range(1, n):
            spread[t] = (1 - mr_speed) * spread[t - 1] + rng.normal(0, spread_vol)
        idio_b = rng.normal(0, vol * 0.3, n)
        log_price_b = common_log_price + spread + idio_b
        prices[tickers[i + 1]] = base_price * np.exp(log_price_b)

    # If odd number of tickers, add one independent series
    if len(tickers) % 2 == 1:
        last = tickers[-1]
        shocks = rng.normal(0.0003, 0.020, n)
        prices[last] = 120 * np.exp(np.cumsum(shocks))

    df = pd.DataFrame(prices, index=dates)
    return df
