"""Cointegration testing and pair selection.

Two statistical mistakes lived in this file and between them they explain most of
this repository's original results. Both are documented inline so the failure mode
stays visible rather than being quietly patched out.

1. WRONG CRITICAL VALUES. The old `select_pairs` confirmed each candidate with a
   plain ADF test on the OLS residual of one price on the other, and compared the
   statistic to standard Dickey-Fuller critical values. Those critical values assume
   the tested series is observed. Here it is a *fitted residual*: OLS chose the
   cointegrating vector precisely to make it look as stationary as possible. The
   resulting statistic is biased toward rejecting the unit-root null, so the ADF
   "confirmation" step passed pairs that a correctly sized test rejects. Engle-Granger
   / MacKinnon critical values exist exactly to absorb that estimation effect, and
   statsmodels' `coint` already reports them.

2. NO MULTIPLE-TESTING CONTROL -- see `select_pairs` below.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint


def adf_test(series: np.ndarray | pd.Series, significance: float = 0.05) -> dict:
    """Augmented Dickey-Fuller test for a unit root in an OBSERVED series.

    Valid when the series being tested is data, or a spread formed with a hedge
    ratio that was fixed in advance (e.g. 1:1, or a known index weight).

    NOT VALID on the residual of a regression whose coefficients were estimated on
    the same sample -- that is the Engle-Granger setting and it needs MacKinnon
    critical values, not these. Using this function there inflates the rejection
    rate: it makes non-cointegrated pairs look stationary. Use
    `engle_granger_test` instead. See `naive_adf_on_residuals` for the deliberately
    incorrect version, which is retained only so the bug can be measured.
    """
    series = np.asarray(series, dtype=float)
    adf_stat, p_value, _, _, crit_values, _ = adfuller(series, autolag="AIC")
    return {
        "adf_stat": adf_stat,
        "p_value": p_value,
        "crit_values": crit_values,
        "stationary": p_value < significance,
    }


def engle_granger_test(
    series_a: np.ndarray | pd.Series,
    series_b: np.ndarray | pd.Series,
) -> dict:
    """Engle-Granger cointegration test with MacKinnon p-values.

    Null hypothesis: no cointegration. statsmodels' `coint` regresses a on b, runs
    the unit-root test on the residual, and -- crucially -- compares it against
    critical values that account for the cointegrating vector having been estimated.

    Returns dict with t_stat, p_value (MacKinnon), crit_values.
    """
    t_stat, p_value, crit_values = coint(
        np.asarray(series_a, dtype=float),
        np.asarray(series_b, dtype=float),
    )
    return {"t_stat": t_stat, "p_value": p_value, "crit_values": crit_values}


def naive_adf_on_residuals(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    significance: float = 0.05,
) -> dict:
    """The original (incorrect) confirmation step, preserved for measurement only.

    Estimates the hedge ratio by OLS on the sample, then hands the residual to a
    standard ADF test as though the hedge ratio had been known all along. This
    over-rejects the unit-root null. Do not select pairs with it; it exists so
    `run_diagnostics.py` can quantify how badly it over-rejects relative to
    `engle_granger_test`.
    """
    spread, _, _ = compute_spread(prices, ticker_a, ticker_b)
    return adf_test(spread, significance)


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg step-up procedure. Returns a boolean rejection mask.

    Controls the false discovery rate -- the expected share of *selected* pairs that
    are not really cointegrated -- at `alpha`. Sort the m p-values ascending, find
    the largest k with p_(k) <= k/m * alpha, and reject the k smallest.

    Caveat worth stating out loud: BH controls FDR exactly under independence or
    positive regression dependence. The C(n,2) pairwise cointegration tests here
    share underlying price series and so are dependent in a way that is not
    guaranteed to be PRDS. BH is therefore approximate in this application, which is
    why `bonferroni` -- valid under arbitrary dependence -- is offered alongside it.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    rejected = np.zeros(m, dtype=bool)
    if m == 0:
        return rejected

    order = np.argsort(p)
    ranks = np.arange(1, m + 1)
    thresholds = ranks / m * alpha
    passing = p[order] <= thresholds

    if not passing.any():
        return rejected

    k = np.max(np.flatnonzero(passing))  # largest index (0-based) that passes
    rejected[order[: k + 1]] = True
    return rejected


def bonferroni(p_values: list[float], alpha: float = 0.05) -> np.ndarray:
    """Bonferroni correction. Returns a boolean rejection mask.

    Controls the family-wise error rate -- the probability of *any* false positive --
    at `alpha`, under arbitrary dependence between the tests. Strictly more
    conservative than Benjamini-Hochberg.
    """
    p = np.asarray(p_values, dtype=float)
    if len(p) == 0:
        return np.zeros(0, dtype=bool)
    return p <= alpha / len(p)


def test_cointegration(
    prices: pd.DataFrame,
    significance: float = 0.05,
) -> list[dict]:
    """Test every ticker pair for cointegration.

    Runs C(n, 2) tests. For n = 10 that is 45 tests per formation window -- keep that
    number in mind, `select_pairs` has to pay for it.

    For each pair, reports BOTH:
      - p_value:        Engle-Granger / MacKinnon. The correct one. Used for selection.
      - naive_adf_p:    standard ADF on the fitted residual. WRONG, reported only so
                        the size distortion is visible in the diagnostics table.

    Returns a list of dicts sorted ascending by the MacKinnon p-value.
    """
    tickers = prices.columns.tolist()
    results = []

    for ticker_a, ticker_b in combinations(tickers, 2):
        eg = engle_granger_test(prices[ticker_a].values, prices[ticker_b].values)
        naive = naive_adf_on_residuals(prices, ticker_a, ticker_b, significance)

        results.append({
            "pair": (ticker_a, ticker_b),
            "t_stat": eg["t_stat"],
            "p_value": eg["p_value"],
            "crit_values": eg["crit_values"],
            "naive_adf_p": naive["p_value"],
            "naive_adf_stationary": naive["stationary"],
        })

    results.sort(key=lambda x: x["p_value"])
    return results


def select_pairs(
    prices: pd.DataFrame,
    significance: float = 0.05,
    max_pairs: int = 5,
    correction: str = "bh",
    criterion: str = "mackinnon",
) -> list[tuple[str, str]]:
    """Select cointegrated pairs from a formation window.

    THE MULTIPLE-TESTING PROBLEM. With n tickers we run m = C(n, 2) tests. At n = 10,
    m = 45. Testing each at alpha = 0.05 with no correction means that even if not a
    single pair is truly cointegrated we expect 45 * 0.05 = 2.25 rejections per
    window by chance alone. The backtest then takes the max_pairs = 3 most
    significant of them. In other words the entire book can be, and empirically often
    was, filled with false positives -- which is why the original run traded things
    like MSFT/SLV at a hedge ratio of -6.74.

    Args:
        correction: multiple-testing control applied across the m pairwise tests.
            "bh"         - Benjamini-Hochberg, controls FDR (default)
            "bonferroni" - controls FWER, most conservative
            "none"       - no correction; reproduces the original bug
        criterion: which p-value to select on.
            "mackinnon" - Engle-Granger p-value from statsmodels `coint` (correct)
            "naive_adf" - standard ADF on the fitted residual; the wrong-critical-values
                          test on its own. For diagnostics only.
            "original"  - the rule this repo originally shipped: Engle-Granger AND the
                          naive ADF must both reject, sold in the README as a "stricter
                          two-test confirmation". Measured on the default universe it
                          filtered out exactly zero of the 25 pairs Engle-Granger
                          accepted across 11 windows, because the naive ADF over-rejects
                          so much that it is never the binding constraint. Kept only so
                          the claim can be falsified rather than argued about.

    Returns up to `max_pairs` pairs, most significant first.
    """
    return apply_selection(
        test_cointegration(prices, significance),
        significance=significance,
        max_pairs=max_pairs,
        correction=correction,
        criterion=criterion,
    )


def apply_selection(
    results: list[dict],
    significance: float = 0.05,
    max_pairs: int = 5,
    correction: str = "bh",
    criterion: str = "mackinnon",
) -> list[tuple[str, str]]:
    """Apply a selection rule to already-computed test results.

    Split out from `select_pairs` so that the same 45 cointegration tests can be
    scored under several rules without paying to recompute them each time.
    """
    if correction not in {"bh", "bonferroni", "none"}:
        raise ValueError(f"unknown correction: {correction!r}")
    if criterion not in {"mackinnon", "naive_adf", "original"}:
        raise ValueError(f"unknown criterion: {criterion!r}")

    if not results:
        return []

    # The p-value the correction is applied to, and thus the ranking key.
    key = "naive_adf_p" if criterion == "naive_adf" else "p_value"
    p_values = [r[key] for r in results]

    if correction == "bh":
        mask = benjamini_hochberg(p_values, significance)
    elif correction == "bonferroni":
        mask = bonferroni(p_values, significance)
    else:
        mask = np.asarray(p_values) < significance

    if criterion == "original":
        # The inert second test: both must reject.
        mask = mask & np.asarray(
            [r["naive_adf_p"] < significance for r in results]
        )

    survivors = [
        (r["pair"], p)
        for r, p, keep in zip(results, p_values, mask)
        if keep
    ]
    survivors.sort(key=lambda item: item[1])  # most significant first
    return [pair for pair, _ in survivors[:max_pairs]]


def compute_spread(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
) -> tuple[pd.Series, float, float]:
    """Compute the hedge ratio and spread between two price series.

    OLS: ticker_a = hedge_ratio * ticker_b + intercept + spread.

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
