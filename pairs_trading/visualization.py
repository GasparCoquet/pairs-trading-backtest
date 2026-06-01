"""Visualization of backtest results."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def plot_equity_curve(
    equity: pd.Series,
    title: str = "Portfolio Equity Curve",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot the equity curve with drawdown shading."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Equity curve
    ax1 = axes[0]
    ax1.plot(equity.index, equity.values, linewidth=1.5, color="#1f77b4")
    ax1.fill_between(equity.index, 1, equity.values, alpha=0.1, color="#1f77b4")
    ax1.set_ylabel("Equity (starting = 1.0)")
    ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    ax2.fill_between(drawdown.index, 0, drawdown.values, color="#d62728", alpha=0.4)
    ax2.plot(drawdown.index, drawdown.values, linewidth=0.8, color="#d62728")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pair_signals(
    result: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    prices: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot spread, z-score, positions, and price series for a single pair."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"Pair: {ticker_a} / {ticker_b}", fontsize=14, fontweight="bold")

    idx = result.index

    # Normalized prices
    ax0 = axes[0]
    p_a = prices[ticker_a].reindex(idx)
    p_b = prices[ticker_b].reindex(idx)
    ax0.plot(idx, p_a / p_a.iloc[0], label=ticker_a, linewidth=1.2)
    ax0.plot(idx, p_b / p_b.iloc[0], label=ticker_b, linewidth=1.2)
    ax0.set_ylabel("Normalized Price")
    ax0.legend(loc="upper left")
    ax0.grid(True, alpha=0.3)

    # Spread
    ax1 = axes[1]
    ax1.plot(idx, result["spread"], linewidth=1, color="#2ca02c")
    ax1.axhline(y=result["spread"].mean(), color="gray", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("Spread")
    ax1.grid(True, alpha=0.3)

    # Z-score with thresholds
    ax2 = axes[2]
    ax2.plot(idx, result["zscore"], linewidth=1, color="#9467bd")
    ax2.axhline(y=2.0, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="Entry (+/-2)")
    ax2.axhline(y=-2.0, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5, label="Exit (0)")
    ax2.set_ylabel("Z-Score")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Positions and entry/exit markers
    ax3 = axes[3]
    ax3.fill_between(idx, 0, result["position"], step="mid", alpha=0.4, color="#ff7f0e")
    ax3.set_ylabel("Position")
    ax3.set_xlabel("Date")
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(["Short Spread", "Flat", "Long Spread"])
    ax3.grid(True, alpha=0.3)

    # Mark entries and exits
    pos = result["position"]
    entries = pos[(pos != 0) & (pos.shift(1) == 0)]
    exits = pos[(pos == 0) & (pos.shift(1) != 0)]
    ax3.scatter(entries.index, entries.values, marker="^", color="green", s=40, zorder=5, label="Entry")
    ax3.scatter(exits.index, [0] * len(exits), marker="v", color="red", s=40, zorder=5, label="Exit")
    ax3.legend(loc="upper left", fontsize=8)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_monthly_returns(
    returns: pd.Series,
    title: str = "Monthly Returns Heatmap",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot monthly returns as a heatmap table."""
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot = monthly_df.pivot_table(values="return", index="year", columns="month")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.6)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-0.05, vmax=0.05)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=8,
                        color="black" if abs(val) < 0.03 else "white")

    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Return", shrink=0.8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
