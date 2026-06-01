"""Performance metrics for strategy evaluation."""

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 252) -> float:
    """Annualized Sharpe ratio."""
    excess = returns - risk_free_rate / periods
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(periods) * excess.mean() / excess.std())


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown as a positive fraction (e.g. 0.15 = 15% drawdown)."""
    running_max = equity.cummax()
    drawdown = (running_max - equity) / running_max
    return float(drawdown.max())


def win_rate(returns: pd.Series) -> float:
    """Fraction of positive-return days."""
    nonzero = returns[returns != 0]
    if len(nonzero) == 0:
        return 0.0
    return float((nonzero > 0).sum() / len(nonzero))


def total_return(equity: pd.Series) -> float:
    """Total cumulative return."""
    if len(equity) < 2:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def annual_return(equity: pd.Series, periods: int = 252) -> float:
    """Annualized return (CAGR)."""
    if len(equity) < 2:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    n_years = len(equity) / periods
    if n_years <= 0:
        return 0.0
    return float(total ** (1 / n_years) - 1)


def annual_volatility(returns: pd.Series, periods: int = 252) -> float:
    """Annualized volatility."""
    return float(returns.std() * np.sqrt(periods))


def calmar_ratio(equity: pd.Series, periods: int = 252) -> float:
    """Calmar ratio: annualized return / max drawdown."""
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return annual_return(equity, periods) / mdd


def compute_all_metrics(
    returns: pd.Series,
    equity: pd.Series,
) -> dict[str, float]:
    """Compute a full suite of performance metrics."""
    return {
        "total_return": total_return(equity),
        "annual_return": annual_return(equity),
        "annual_volatility": annual_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "calmar_ratio": calmar_ratio(equity),
        "win_rate": win_rate(returns),
        "num_trading_days": len(returns),
    }
