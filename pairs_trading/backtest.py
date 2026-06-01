"""Walk-forward backtesting engine with transaction costs."""

import numpy as np
import pandas as pd

from pairs_trading.cointegration import compute_spread, select_pairs
from pairs_trading.signals import compute_zscore, generate_signals


def backtest_pair(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    hedge_ratio: float,
    intercept: float,
    lookback: int = 20,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    transaction_cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Run backtest for a single pair over a given period.

    Returns DataFrame with columns: spread, zscore, position, strategy_returns, equity.
    """
    price_a = prices[ticker_a]
    price_b = prices[ticker_b]

    spread = price_a - hedge_ratio * price_b - intercept
    zscore = compute_zscore(spread, lookback=lookback)
    positions = generate_signals(zscore, entry_threshold, exit_threshold)

    # Compute daily returns of each leg
    ret_a = price_a.pct_change()
    ret_b = price_b.pct_change()

    # Spread return: long A, short B (per unit notional on each leg)
    spread_returns = positions.shift(1) * (ret_a - hedge_ratio * ret_b) / (1 + abs(hedge_ratio))

    # Transaction costs on position changes
    position_changes = positions.diff().abs()
    cost_per_trade = transaction_cost_bps / 10_000
    costs = position_changes * cost_per_trade

    strategy_returns = spread_returns - costs
    strategy_returns = strategy_returns.fillna(0)
    equity = (1 + strategy_returns).cumprod()

    return pd.DataFrame({
        "spread": spread,
        "zscore": zscore,
        "position": positions,
        "strategy_returns": strategy_returns,
        "equity": equity,
    })


def walk_forward_backtest(
    prices: pd.DataFrame,
    tickers: list[str],
    formation_period: int = 252,
    trading_period: int = 126,
    lookback: int = 20,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    transaction_cost_bps: float = 10.0,
    significance: float = 0.05,
    max_pairs: int = 3,
) -> dict:
    """Walk-forward backtest: re-select pairs and re-estimate hedge ratios periodically.

    Splits the data into rolling windows:
    - Formation window: used for cointegration testing and hedge ratio estimation
    - Trading window: out-of-sample trading period

    Returns dict with:
        - combined_equity: pd.Series of portfolio equity curve
        - pair_results: list of per-window results
        - trades: summary of all trade windows
    """
    all_prices = prices[tickers]
    n = len(all_prices)
    pair_results = []
    equity_segments = []

    start = 0
    window_id = 0

    while start + formation_period + trading_period <= n:
        formation_end = start + formation_period
        trading_end = min(formation_end + trading_period, n)

        formation_data = all_prices.iloc[start:formation_end]
        trading_data = all_prices.iloc[formation_end:trading_end]

        # Select pairs on formation data
        selected = select_pairs(formation_data, significance=significance, max_pairs=max_pairs)

        if not selected:
            start += trading_period
            window_id += 1
            continue

        # Backtest each selected pair on the trading window
        window_equities = []
        for ticker_a, ticker_b in selected:
            spread, hedge_ratio, intercept = compute_spread(
                formation_data, ticker_a, ticker_b
            )

            result = backtest_pair(
                trading_data,
                ticker_a, ticker_b,
                hedge_ratio, intercept,
                lookback=lookback,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                transaction_cost_bps=transaction_cost_bps,
            )

            pair_results.append({
                "window": window_id,
                "pair": (ticker_a, ticker_b),
                "hedge_ratio": hedge_ratio,
                "formation": (
                    all_prices.index[start].strftime("%Y-%m-%d"),
                    all_prices.index[formation_end - 1].strftime("%Y-%m-%d"),
                ),
                "trading": (
                    all_prices.index[formation_end].strftime("%Y-%m-%d"),
                    all_prices.index[trading_end - 1].strftime("%Y-%m-%d"),
                ),
                "result": result,
            })
            window_equities.append(result["strategy_returns"])

        # Equal-weight portfolio across selected pairs
        if window_equities:
            portfolio_returns = pd.concat(window_equities, axis=1).mean(axis=1)
            equity_segments.append(portfolio_returns)

        start += trading_period
        window_id += 1

    # Chain equity segments together
    if equity_segments:
        all_returns = pd.concat(equity_segments)
        # Remove any duplicate indices from overlapping windows
        all_returns = all_returns[~all_returns.index.duplicated(keep="first")]
        combined_equity = (1 + all_returns).cumprod()
    else:
        combined_equity = pd.Series(dtype=float)

    return {
        "combined_equity": combined_equity,
        "combined_returns": all_returns if equity_segments else pd.Series(dtype=float),
        "pair_results": pair_results,
    }
