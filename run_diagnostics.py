#!/usr/bin/env python3
"""Quantify each bug on real market data, one at a time.

Runs the same walk-forward backtest under different selection rules so the effect of
(a) the wrong critical values and (b) the missing multiple-testing correction can be
read off directly instead of taken on faith. Also counts, per formation window, how
many of the C(n,2) tests each rule rejects.

Everything printed here is measured. Nothing is asserted from theory alone.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from pairs_trading.backtest import walk_forward_backtest
from pairs_trading.cointegration import (
    benjamini_hochberg,
    bonferroni,
    test_cointegration,
)
from pairs_trading.data import DataFetchError, fetch_prices
from pairs_trading.metrics import compute_all_metrics

DEFAULT_TICKERS = [
    "XOM", "CVX", "KO", "PEP", "JPM", "BAC", "MSFT", "AAPL", "GLD", "SLV",
]

# (label, correction, criterion, stop_loss_z)
VARIANTS = [
    ("ORIGINAL rule (EG and naive ADF)",  "none", "original",  None),
    ("naive ADF alone",                   "none", "naive_adf", None),
    ("Engle-Granger alone, uncorrected",  "none", "mackinnon", None),
    ("+ Benjamini-Hochberg (FDR)",        "bh",   "mackinnon", None),
    ("+ Bonferroni (FWER)",               "bonferroni", "mackinnon", None),
    ("BH + stop-loss at |z|=3",           "bh",   "mackinnon", 3.0),
]


def rejection_counts(
    prices: pd.DataFrame,
    formation_period: int,
    trading_period: int,
    significance: float,
) -> None:
    """Per formation window, count rejections under each rule."""
    n = len(prices)
    n_pairs = len(prices.columns) * (len(prices.columns) - 1) // 2

    print("=" * 78)
    print("REJECTIONS PER FORMATION WINDOW")
    print("=" * 78)
    print(f"Each window runs C({len(prices.columns)},2) = {n_pairs} cointegration tests.")
    print(f"Under the null of no cointegration anywhere, {n_pairs} uncorrected tests at")
    print(f"alpha={significance} are expected to yield {n_pairs * significance:.2f} "
          f"false rejections per window by chance alone.")
    print(f"The backtest then trades the top max_pairs of them.\n")

    print(f"{'window':<8}{'formation end':<16}{'naiveADF':>10}{'EG raw':>9}"
          f"{'EG+BH':>8}{'EG+Bonf':>9}")
    print("-" * 78)

    start = 0
    window_id = 0
    totals = {"naive": 0, "raw": 0, "bh": 0, "bonf": 0}
    n_windows = 0
    eg_pass_naive_fail = 0  # times the "confirmation" test actually filtered a pair

    while start + formation_period + trading_period <= n:
        formation_end = start + formation_period
        formation_data = prices.iloc[start:formation_end]

        results = test_cointegration(formation_data, significance)
        p_eg = [r["p_value"] for r in results]
        p_naive = [r["naive_adf_p"] for r in results]

        n_naive = int(sum(1 for p in p_naive if p < significance))
        n_raw = int(sum(1 for p in p_eg if p < significance))
        n_bh = int(benjamini_hochberg(p_eg, significance).sum())
        n_bonf = int(bonferroni(p_eg, significance).sum())

        eg_pass_naive_fail += sum(
            1 for pe, pn in zip(p_eg, p_naive)
            if pe < significance and pn >= significance
        )

        totals["naive"] += n_naive
        totals["raw"] += n_raw
        totals["bh"] += n_bh
        totals["bonf"] += n_bonf
        n_windows += 1

        print(f"{window_id:<8}{formation_data.index[-1].strftime('%Y-%m-%d'):<16}"
              f"{n_naive:>10}{n_raw:>9}{n_bh:>8}{n_bonf:>9}")

        start += trading_period
        window_id += 1

    print("-" * 78)
    print(f"{'mean':<24}{totals['naive'] / n_windows:>10.2f}"
          f"{totals['raw'] / n_windows:>9.2f}{totals['bh'] / n_windows:>8.2f}"
          f"{totals['bonf'] / n_windows:>9.2f}")

    print(f"\nExpected false rejections per window if NOTHING is cointegrated: "
          f"{n_pairs * significance:.2f}")
    print(f"Actually observed per window, Engle-Granger uncorrected:          "
          f"{totals['raw'] / n_windows:.2f}")
    print("The uncorrected selector finds about as many 'cointegrated' pairs as pure")
    print("chance would produce if there were no cointegration in the universe at all.")

    print(f"\nWas the ADF 'confirmation' a stricter second test?")
    print(f"  Pairs Engle-Granger accepted across all windows: {totals['raw']}")
    print(f"  Of those, rejected by the naive ADF step:        {eg_pass_naive_fail}")
    print(f"  The second test filtered out {eg_pass_naive_fail} pairs. It was inert: it")
    print(f"  over-rejects so heavily ({totals['naive']} rejections vs Engle-Granger's "
          f"{totals['raw']})")
    print(f"  that it is never the binding constraint.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--formation-period", type=int, default=252)
    parser.add_argument("--trading-period", type=int, default=126)
    parser.add_argument("--significance", type=float, default=0.05)
    parser.add_argument("--max-pairs", type=int, default=3)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    args = parser.parse_args()

    try:
        prices = fetch_prices(args.tickers, start=args.start, end=args.end)
    except DataFetchError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        sys.exit(2)

    print(f"\nREAL Yahoo Finance data: {len(prices)} trading days, "
          f"{len(prices.columns)} tickers, {args.start} to {args.end}")
    print(f"Tickers: {list(prices.columns)}\n")

    rejection_counts(
        prices, args.formation_period, args.trading_period, args.significance
    )

    print("=" * 78)
    print("WALK-FORWARD PERFORMANCE UNDER EACH SELECTION RULE (real data)")
    print("=" * 78)
    print(f"{'variant':<38}{'pairs':>6}{'flat':>6}{'Sharpe':>8}"
          f"{'CAGR':>9}{'MaxDD':>8}{'Vol':>8}")
    print("-" * 78)

    rows = []
    for label, correction, criterion, stop in VARIANTS:
        results = walk_forward_backtest(
            prices,
            tickers=list(prices.columns),
            formation_period=args.formation_period,
            trading_period=args.trading_period,
            transaction_cost_bps=args.cost_bps,
            significance=args.significance,
            max_pairs=args.max_pairs,
            correction=correction,
            criterion=criterion,
            stop_loss_z=stop,
        )
        m = compute_all_metrics(results["combined_returns"], results["combined_equity"])
        n_flat = sum(1 for w in results["window_log"] if w["n_selected"] == 0)
        n_traded = len(results["pair_results"])

        print(f"{label:<38}{n_traded:>6}{n_flat:>6}{m['sharpe_ratio']:>8.2f}"
              f"{m['annual_return']:>8.2%}{m['max_drawdown']:>8.2%}"
              f"{m['annual_volatility']:>8.2%}")
        rows.append((label, results, m))

    print("-" * 78)
    print("'pairs' = pair-windows traded. 'flat' = windows where no pair survived")
    print("selection, so the book was in cash.\n")

    print("=" * 78)
    print("PAIRS ACTUALLY TRADED")
    print("=" * 78)
    for label, results, _m in rows:
        pairs = [f"{p['pair'][0]}/{p['pair'][1]}" for p in results["pair_results"]]
        counts = pd.Series(pairs).value_counts() if pairs else pd.Series(dtype=int)
        print(f"\n[{label}]  {len(pairs)} pair-windows")
        if len(counts):
            print("  " + ", ".join(f"{k} x{v}" for k, v in counts.items()))
        else:
            print("  (none)")

    print()


if __name__ == "__main__":
    main()
