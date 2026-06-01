#!/usr/bin/env python3
"""Main entry point for the pairs trading backtest."""

import argparse
import os
import sys

import pandas as pd

from pairs_trading.data import fetch_prices
from pairs_trading.cointegration import test_cointegration, compute_spread
from pairs_trading.backtest import walk_forward_backtest
from pairs_trading.metrics import compute_all_metrics
from pairs_trading.visualization import (
    plot_equity_curve,
    plot_pair_signals,
    plot_monthly_returns,
)

# Default universe: large-cap stocks from related sectors that may cointegrate
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
        "--cost-bps", type=float, default=10.0,
        help="Round-trip transaction cost in basis points",
    )
    parser.add_argument("--significance", type=float, default=0.05, help="Cointegration p-value threshold")
    parser.add_argument("--max-pairs", type=int, default=3, help="Max pairs to trade per window")
    parser.add_argument("--output-dir", default="output", help="Directory for output plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("PAIRS TRADING WALK-FORWARD BACKTEST")
    print("=" * 60)

    # --- 1. Fetch data ---
    print(f"\nFetching prices for {len(args.tickers)} tickers: {args.start} to {args.end}")
    prices = fetch_prices(args.tickers, start=args.start, end=args.end)
    print(f"Retrieved {len(prices)} trading days for {len(prices.columns)} tickers")
    print(f"Tickers: {list(prices.columns)}")

    # --- 2. Show cointegration results on full sample (informational) ---
    print("\n" + "-" * 60)
    print("COINTEGRATION TEST RESULTS (full sample, informational)")
    print("-" * 60)
    coint_results = test_cointegration(prices)
    print(f"{'Pair':<20} {'t-stat':>10} {'p-value':>10} {'ADF p':>10} {'Cointegrated':>14}")
    print("-" * 68)
    for r in coint_results[:15]:
        pair_str = f"{r['pair'][0]}/{r['pair'][1]}"
        coint_flag = "YES" if (r["p_value"] < args.significance and r["adf_stationary"]) else "no"
        print(f"{pair_str:<20} {r['t_stat']:>10.3f} {r['p_value']:>10.4f} "
              f"{r['adf_p_value']:>10.4f} {coint_flag:>14}")

    # --- 3. Run walk-forward backtest ---
    print("\n" + "-" * 60)
    print("WALK-FORWARD BACKTEST")
    print("-" * 60)
    print(f"Formation period: {args.formation_period} days")
    print(f"Trading period:   {args.trading_period} days")
    print(f"Entry z-score:    +/-{args.entry_z}")
    print(f"Exit z-score:     +/-{args.exit_z}")
    print(f"Transaction cost: {args.cost_bps} bps")
    print(f"Max pairs/window: {args.max_pairs}")

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
    )

    if results["combined_equity"].empty:
        print("\nNo cointegrated pairs found in any window. Try adjusting parameters.")
        sys.exit(1)

    # --- 4. Print trade windows ---
    print(f"\nTraded {len(results['pair_results'])} pair-windows:")
    for pr in results["pair_results"]:
        pair_str = f"{pr['pair'][0]}/{pr['pair'][1]}"
        print(f"  Window {pr['window']}: {pair_str}  "
              f"Formation {pr['formation'][0]}..{pr['formation'][1]}  "
              f"Trading {pr['trading'][0]}..{pr['trading'][1]}  "
              f"Hedge ratio: {pr['hedge_ratio']:.3f}")

    # --- 5. Performance metrics ---
    print("\n" + "-" * 60)
    print("PERFORMANCE METRICS")
    print("-" * 60)

    metrics = compute_all_metrics(results["combined_returns"], results["combined_equity"])
    print(f"Total Return:       {metrics['total_return']:>10.2%}")
    print(f"Annual Return:      {metrics['annual_return']:>10.2%}")
    print(f"Annual Volatility:  {metrics['annual_volatility']:>10.2%}")
    print(f"Sharpe Ratio:       {metrics['sharpe_ratio']:>10.2f}")
    print(f"Max Drawdown:       {metrics['max_drawdown']:>10.2%}")
    print(f"Calmar Ratio:       {metrics['calmar_ratio']:>10.2f}")
    print(f"Win Rate:           {metrics['win_rate']:>10.2%}")
    print(f"Trading Days:       {metrics['num_trading_days']:>10d}")

    # --- 6. Generate plots ---
    if not args.no_plots:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\nSaving plots to {args.output_dir}/")

        plot_equity_curve(
            results["combined_equity"],
            title="Pairs Trading Strategy - Walk-Forward Equity Curve",
            save_path=os.path.join(args.output_dir, "equity_curve.png"),
        )
        print("  -> equity_curve.png")

        plot_monthly_returns(
            results["combined_returns"],
            title="Pairs Trading Strategy - Monthly Returns",
            save_path=os.path.join(args.output_dir, "monthly_returns.png"),
        )
        print("  -> monthly_returns.png")

        # Plot first pair from each window (up to 4 windows)
        plotted_windows = set()
        for pr in results["pair_results"]:
            if pr["window"] in plotted_windows:
                continue
            if len(plotted_windows) >= 4:
                break
            plotted_windows.add(pr["window"])
            ticker_a, ticker_b = pr["pair"]
            fname = f"signals_w{pr['window']}_{ticker_a}_{ticker_b}.png"
            plot_pair_signals(
                pr["result"],
                ticker_a, ticker_b,
                prices,
                save_path=os.path.join(args.output_dir, fname),
            )
            print(f"  -> {fname}")

    print("\n" + "=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
