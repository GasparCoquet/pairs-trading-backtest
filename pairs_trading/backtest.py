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
    stop_loss_z: float | None = None,
) -> pd.DataFrame:
    """Run backtest for a single pair over a given period.

    The hedge ratio is estimated on the formation window and held fixed out of
    sample, which is the point of walk-forward: no lookahead.

    `stop_loss_z` is the risk control the original strategy did not have. A pair is
    selected because its spread mean-reverted in the formation window; nothing
    guarantees it still does out of sample. Without a stop, a pair that simply stops
    being cointegrated is held all the way down, because the z-score never comes back
    to the exit band. When set, the position is closed once |z| exceeds stop_loss_z,
    on the view that a spread this far from its mean is more likely broken than
    stretched.

    Returns DataFrame with columns: spread, zscore, position, strategy_returns, equity.
    """
    price_a = prices[ticker_a]
    price_b = prices[ticker_b]

    spread = price_a - hedge_ratio * price_b - intercept
    zscore = compute_zscore(spread, lookback=lookback)
    positions = generate_signals(
        zscore, entry_threshold, exit_threshold, stop_loss_z=stop_loss_z
    )

    ret_a = price_a.pct_change()
    ret_b = price_b.pct_change()

    # Spread return: long A, short B. Divided by (1 + |h|) so that one unit of
    # position corresponds to one unit of gross notional across both legs.
    spread_returns = (
        positions.shift(1) * (ret_a - hedge_ratio * ret_b) / (1 + abs(hedge_ratio))
    )

    position_changes = positions.diff().abs()
    cost_per_trade = transaction_cost_bps / 10_000
    costs = position_changes * cost_per_trade

    strategy_returns = (spread_returns - costs).fillna(0)
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
    correction: str = "bh",
    criterion: str = "mackinnon",
    stop_loss_z: float | None = None,
) -> dict:
    """Walk-forward backtest: re-select pairs and re-estimate hedge ratios per window.

    Formation window -> cointegration testing and hedge ratio estimation.
    Trading window   -> out-of-sample trading with those frozen parameters.

    Windows in which no pair survives selection are held FLAT (zero return) rather
    than dropped from the series. The original code skipped them entirely, so the
    equity curve silently omitted the days the strategy was not invested and every
    annualised figure was computed over days-in-market instead of calendar time. A
    strategy that sits in cash is a strategy earning zero, and it should be scored
    that way.

    Returns dict with combined_equity, combined_returns, pair_results, window_log.
    """
    all_prices = prices[tickers]
    n = len(all_prices)
    pair_results: list[dict] = []
    return_segments: list[pd.Series] = []
    window_log: list[dict] = []

    start = 0
    window_id = 0

    while start + formation_period + trading_period <= n:
        formation_end = start + formation_period
        trading_end = min(formation_end + trading_period, n)

        formation_data = all_prices.iloc[start:formation_end]
        trading_data = all_prices.iloc[formation_end:trading_end]

        selected = select_pairs(
            formation_data,
            significance=significance,
            max_pairs=max_pairs,
            correction=correction,
            criterion=criterion,
        )

        window_log.append({
            "window": window_id,
            "formation": (
                all_prices.index[start].strftime("%Y-%m-%d"),
                all_prices.index[formation_end - 1].strftime("%Y-%m-%d"),
            ),
            "trading": (
                all_prices.index[formation_end].strftime("%Y-%m-%d"),
                all_prices.index[trading_end - 1].strftime("%Y-%m-%d"),
            ),
            "n_selected": len(selected),
            "pairs": list(selected),
        })

        if not selected:
            # Flat: no pair passed selection, so the book is in cash and earns zero.
            return_segments.append(
                pd.Series(0.0, index=trading_data.index, dtype=float)
            )
            start += trading_period
            window_id += 1
            continue

        pair_returns = []
        for ticker_a, ticker_b in selected:
            _, hedge_ratio, intercept = compute_spread(
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
                stop_loss_z=stop_loss_z,
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
            pair_returns.append(result["strategy_returns"])

        # Equal weight across the pairs selected in this window.
        return_segments.append(pd.concat(pair_returns, axis=1).mean(axis=1))

        start += trading_period
        window_id += 1

    if not return_segments:
        empty = pd.Series(dtype=float)
        return {
            "combined_equity": empty,
            "combined_returns": empty,
            "pair_results": [],
            "window_log": window_log,
        }

    all_returns = pd.concat(return_segments)
    all_returns = all_returns[~all_returns.index.duplicated(keep="first")].sort_index()
    combined_equity = (1 + all_returns).cumprod()

    return {
        "combined_equity": combined_equity,
        "combined_returns": all_returns,
        "pair_results": pair_results,
        "window_log": window_log,
    }
