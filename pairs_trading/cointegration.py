"""Cointegration testing for pair selection."""

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint


def adf_test(series: np.ndarray | pd.Series, significance: float = 0.05) -> dict:
    """Augmented Dickey-Fuller test for stationarity of a single series.

    The null hypothesis is that the series has a unit root (non-stationary).
    A p-value below `significance` rejects the null, implying the series is
    stationary (mean-reverting) — exactly the property a tradeable spread needs.

    Returns a dict with:
        - adf_stat: the ADF test statistic
        - p_value: p-value of the test
        - crit_values: critical values at 1%, 5%, 10%
        - stationary: bool, True if p_value < significance
    """
    series = np.asarray(series, dtype=float)
    adf_stat, p_value, _, _, crit_values, _ = adfuller(series, autolag="AIC")
    return {
        "adf_stat": adf_stat,
        "p_value": p_value,
        "crit_values": crit_values,
        "stationary": p_value < significance,
    }


def test_cointegration(
    prices: pd.DataFrame,
    significance: float = 0.05,
) -> list[dict]:
    """Run Engle-Granger cointegration test on all ticker pairs.

    Two-step procedure:
      1. Engle-Granger cointegration test (statsmodels `coint`) on the price levels.
      2. Augmented Dickey-Fuller test on the OLS spread residuals to confirm the
         spread is itself stationary (mean-reverting).

    Returns a list of dicts sorted by cointegration p-value, each containing:
        - pair: tuple of (ticker_a, ticker_b)
        - t_stat: cointegration test statistic
        - p_value: p-value of the cointegration test
        - crit_values: critical values at 1%, 5%, 10%
        - adf_p_value: ADF p-value on the spread residuals
        - adf_stationary: bool, True if the spread is stationary
    """
    tickers = prices.columns.tolist()
    results = []

    for ticker_a, ticker_b in combinations(tickers, 2):
        series_a = prices[ticker_a].values
        series_b = prices[ticker_b].values
        t_stat, p_value, crit_values = coint(series_a, series_b)

        # Confirm spread stationarity with an explicit ADF test
        spread, _, _ = compute_spread(prices, ticker_a, ticker_b)
        adf = adf_test(spread, significance)

        results.append({
            "pair": (ticker_a, ticker_b),
            "t_stat": t_stat,
            "p_value": p_value,
            "crit_values": crit_values,
            "adf_p_value": adf["p_value"],
            "adf_stationary": adf["stationary"],
        })

    results.sort(key=lambda x: x["p_value"])
    return results


def select_pairs(
    prices: pd.DataFrame,
    significance: float = 0.05,
    max_pairs: int = 5,
    require_adf: bool = True,
) -> list[tuple[str, str]]:
    """Select cointegrated pairs that pass the significance threshold.

    A pair is selected only if the Engle-Granger cointegration p-value is below
    `significance`. When `require_adf` is True (default), the spread must also
    pass the ADF stationarity test, giving a stricter two-test confirmation.
    """
    results = test_cointegration(prices, significance)
    selected = [
        r["pair"] for r in results
        if r["p_value"] < significance
        and (not require_adf or r["adf_stationary"])
    ]
    return selected[:max_pairs]


def compute_spread(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
) -> tuple[pd.Series, float, float]:
    """Compute the hedge ratio and spread between two price series.

    Uses OLS regression: ticker_a = hedge_ratio * ticker_b + intercept + spread.

    Returns (spread_series, hedge_ratio, intercept).
    """
    y = prices[ticker_a].values
    x = prices[ticker_b].values
    x_with_const = np.column_stack([x, np.ones(len(x))])
    coeffs = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
    hedge_ratio, intercept = coeffs[0], coeffs[1]
    spread = pd.Series(
        y - hedge_ratio * x - intercept,
        index=prices.index,
        name=f"{ticker_a}-{hedge_ratio:.2f}*{ticker_b}",
    )
    return spread, hedge_ratio, intercept
