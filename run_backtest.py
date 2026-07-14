#!/usr/bin/env python3
"""Main entry point for the pairs trading backtest.

Real market data by default. If the download fails, this exits non-zero rather than
substituting simulated prices -- see pairs_trading/data.py for why that matters.

`--synthetic` runs the selector validation harness on labelled simulated data. It
prints a loud banner, and its output is a test result, never a performance claim.
"""

import argparse
import os
import sys

from pairs_trading.data import DataFetchError, fetch_prices
from pairs_trading.cointegration import test_cointegration, benjamini_hochberg, bonferroni
from pairs_trading.backtest import walk_forward_backtest
from pairs_trading.metrics import compute_all_metrics
from pairs_trading.validation import run_validation
from pairs_trading.visualization import (
    plot_equity_curve,
    plot_pair_signals,
    plot_monthly_returns,
)

# Default universe: large caps from related sectors, plus two metals ETFs.
DEFAULT_TICKERS = [
    "XOM", "CVX",   # Energy
    "KO", "PEP",    # Consumer staples
    "JPM", "BAC",   # Financials
    "MSFT", "AAPL", # Tech
    "GLD", "SLV",   # Precious metals ETFs
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pairs Trading Walk-Forward Backtest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS,
        help="List of ticker symbols to include in the universe",
    )
    parser.add_argument("--start", default="2018-01-01", help="Backtest start date")
    parser.add_argument("--end", default="2024-12-31", help="Backtest end date")
    parser.add_argument(
        "--formation-period", type=int, default=252,
        help="Number of trading days for the formation (in-sample) window",
    )
    parser.add_argument(
        "--trading-period", type=int, default=126,
        help="Number of trading days for the trading (out-of-sample) window",
    )
    parser.add_argument("--lookback", type=int, default=20, help="Z-score rolling lookback window")
    parser.add_argument("--entry-z", type=float, default=2.0, help="Z-score entry threshold")
    parser.add_argument("--exit-z", type=float, default=0.0, help="Z-score exit threshold")
    parser.add_argument(
        "--stop-loss-z", type=float, default=None,
        help="Close the position if |z| exceeds this. Off by default.",
    )
    parser.add_argument(
        "--cost-bps", type=float, default=10.0,
        help="Round-trip transaction cost in basis points",
    )
    parser.add_argument("--significance", type=float, default=0.05, help="Cointegration p-value threshold")
    parser.add_argument("--max-pairs", type=int, default=3, help="Max pairs to trade per window")
    parser.add_argument(
        "--correction", choices=["bh", "bonferroni", "none"], default="bh",
        help="Multiple-testing correction across the C(n,2) cointegration tests. "
             "'none' reproduces the original bug.",
    )
    parser.add_argument(
        "--criterion", choices=["mackinnon", "naive_adf", "original"],
        default="mackinnon",
        help="Which test to select on. 'mackinnon' is correct. 'naive_adf' is the "
             "wrong-critical-values test alone. 'original' is the rule this repo "
             "shipped (Engle-Granger AND naive ADF). The latter two are for "
             "reproducing the bugs, not for use.",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Run the selector validation harness on labelled SIMULATED data. "
             "Produces a test result, not a performance claim.",
    )
    parser.add_argument(
        "--n-universes", type=int, default=20,
        help="Number of independently seeded synthetic universes for --synthetic",
    )
    parser.add_argument("--output-dir", default="output", help="Directory for output plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.synthetic:
        # Simulated data. Never produces a performance number.
        run_validation(
            args.tickers,
            start=args.start,
            end=args.end,
            significance=args.significance,
            n_universes=args.n_universes,
            strict=True,
        )
        return

    print("=" * 68)
    print("PAIRS TRADING WALK-FORWARD BACKTEST")
    print("=" * 68)

    # --- 1. Fetch real data. No fallback: this either works or we stop. ---
    print(f"\nFetching REAL prices for {len(args.tickers)} tickers: {args.start} to {args.end}")
    try:
        prices = fetch_prices(args.tickers, start=args.start, end=args.end)
    except DataFetchError as exc:
        print(f"\n[FATAL] {exc}", file=sys.stderr)
        print(
            "\nNot substituting synthetic data. A backtest on prices we could not "
            "download is not a backtest.",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"Retrieved {len(prices)} trading days for {len(prices.columns)} tickers")
    print(f"Tickers: {list(prices.columns)}")

    # --- 2. Full-sample cointegration table (informational) ---
    n_pairs = len(prices.columns) * (len(prices.columns) - 1) // 2
    print("\n" + "-" * 68)
    print(f"COINTEGRATION TESTS (full sample, informational) -- {n_pairs} pairs")
    print("-" * 68)
    coint_results = test_cointegration(prices, args.significance)
    p_values = [r["p_value"] for r in coint_results]
    bh_mask = benjamini_hochberg(p_values, args.significance)
    bonf_mask = bonferroni(p_values, args.significance)

    print(f"{'Pair':<14}{'t-stat':>9}{'EG p':>9}{'naiveADF p':>12}"
          f"{'raw':>6}{'BH':>5}{'Bonf':>6}")
    print("-" * 68)
    for r, bh, bf in list(zip(coint_results, bh_mask, bonf_mask))[:15]:
        pair_str = f"{r['pair'][0]}/{r['pair'][1]}"
        raw = "YES" if r["p_value"] < args.significance else "no"
        print(f"{pair_str:<14}{r['t_stat']:>9.3f}{r['p_value']:>9.4f}"
              f"{r['naive_adf_p']:>12.4f}{raw:>6}"
              f"{('YES' if bh else 'no'):>5}{('YES' if bf else 'no'):>6}")

    n_raw = int(sum(1 for p in p_values if p < args.significance))
    n_naive = int(sum(1 for r in coint_results if r["naive_adf_p"] < args.significance))
    print("-" * 68)
    print(f"Rejections at alpha={args.significance}: "
          f"naive ADF {n_naive}/{n_pairs}, Engle-Granger raw {n_raw}/{n_pairs}, "
          f"BH {int(bh_mask.sum())}/{n_pairs}, Bonferroni {int(bonf_mask.sum())}/{n_pairs}")
    print(f"Expected false positives from {n_pairs} uncorrected tests at "
          f"alpha={args.significance}: {n_pairs * args.significance:.2f}")

    # --- 3. Walk-forward backtest ---
    print("\n" + "-" * 68)
    print("WALK-FORWARD BACKTEST")
    print("-" * 68)
    print(f"Formation period:   {args.formation_period} days")
    print(f"Trading period:     {args.trading_period} days")
    print(f"Entry z-score:      +/-{args.entry_z}")
    print(f"Exit z-score:       +/-{args.exit_z}")
    print(f"Stop-loss z-score:  {args.stop_loss_z if args.stop_loss_z else 'none'}")
    print(f"Transaction cost:   {args.cost_bps} bps")
    print(f"Max pairs/window:   {args.max_pairs}")
    print(f"Selection:          {args.criterion} p-value, {args.correction} correction")

    results = walk_forward_backtest(
        prices,
        tickers=list(prices.columns),
        formation_period=args.formation_period,
        trading_period=args.trading_period,
        lookback=args.lookback,
        entry_threshold=args.entry_z,
        exit_threshold=args.exit_z,
        transaction_cost_bps=args.cost_bps,
        significance=args.significance,
        max_pairs=args.max_pairs,
        correction=args.correction,
        criterion=args.criterion,
        stop_loss_z=args.stop_loss_z,
    )

    window_log = results["window_log"]
    n_windows = len(window_log)
    n_flat = sum(1 for w in window_log if w["n_selected"] == 0)

    if results["combined_equity"].empty:
        print("\nNo walk-forward windows could be formed. Check the date range.")
        sys.exit(1)

    print(f"\nWindows: {n_windows} total, {n_flat} selected no pairs (held flat)")
    print(f"Traded {len(results['pair_results'])} pair-windows:")
    for pr in results["pair_results"]:
        pair_str = f"{pr['pair'][0]}/{pr['pair'][1]}"
        print(f"  Window {pr['window']}: {pair_str:<12} "
              f"Formation {pr['formation'][0]}..{pr['formation'][1]}  "
              f"Trading {pr['trading'][0]}..{pr['trading'][1]}  "
              f"Hedge ratio: {pr['hedge_ratio']:>7.3f}")

    # --- 4. Performance metrics ---
    print("\n" + "-" * 68)
    print("PERFORMANCE METRICS (real market data)")
    print("-" * 68)

    metrics = compute_all_metrics(results["combined_returns"], results["combined_equity"])
    print(f"Total Return:       {metrics['total_return']:>10.2%}")
    print(f"Annual Return:      {metrics['annual_return']:>10.2%}")
    print(f"Annual Volatility:  {metrics['annual_volatility']:>10.2%}")
    print(f"Sharpe Ratio:       {metrics['sharpe_ratio']:>10.2f}")
    print(f"Max Drawdown:       {metrics['max_drawdown']:>10.2%}")
    print(f"Calmar Ratio:       {metrics['calmar_ratio']:>10.2f}")
    print(f"Win Rate:           {metrics['win_rate']:>10.2%}")
    print(f"Trading Days:       {metrics['num_trading_days']:>10d}")

    # --- 5. Plots ---
    if not args.no_plots:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\nSaving plots to {args.output_dir}/")

        plot_equity_curve(
            results["combined_equity"],
            title="Pairs Trading - Walk-Forward Equity Curve (real data)",
            save_path=os.path.join(args.output_dir, "equity_curve.png"),
        )
        print("  -> equity_curve.png")

        plot_monthly_returns(
            results["combined_returns"],
            title="Pairs Trading - Monthly Returns (real data)",
            save_path=os.path.join(args.output_dir, "monthly_returns.png"),
        )
        print("  -> monthly_returns.png")

        plotted_windows = set()
        for pr in results["pair_results"]:
            if pr["window"] in plotted_windows or len(plotted_windows) >= 4:
                continue
            plotted_windows.add(pr["window"])
            ticker_a, ticker_b = pr["pair"]
            fname = f"signals_w{pr['window']}_{ticker_a}_{ticker_b}.png"
            plot_pair_signals(
                pr["result"], ticker_a, ticker_b, prices,
                save_path=os.path.join(args.output_dir, fname),
            )
            print(f"  -> {fname}")

    print("\n" + "=" * 68)
    print("BACKTEST COMPLETE")
    print("=" * 68)


if __name__ == "__main__":
    main()
