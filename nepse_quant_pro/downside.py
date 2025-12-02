import math
from typing import Dict
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import PERIODS_PER_YEAR
from .types import DownsideRiskResult


def compute_drawdowns(price_series: pd.Series) -> pd.DataFrame:
    """Return price, rolling peak, and drawdown percentage."""
    price = price_series.dropna()
    peak = price.cummax()
    drawdown = price / peak - 1.0
    return pd.DataFrame({"price": price, "peak": peak, "drawdown": drawdown})


def compute_downside_stats(returns: pd.Series, freq: int = PERIODS_PER_YEAR) -> Dict[str, float]:
    """Compute downside volatility, drawdown stats, and Sortino ratio."""
    r = returns.dropna()
    if r.empty:
        return {
            "semi_vol_annual": 0.0,
            "max_drawdown": 0.0,
            "current_drawdown": 0.0,
            "longest_dd_days": 0,
            "sortino": 0.0,
        }

    downside = r[r < 0]
    if len(downside) > 0:
        semi_var = (downside ** 2).mean()
        semi_vol_annual = math.sqrt(semi_var) * math.sqrt(freq)
    else:
        semi_vol_annual = 0.0

    simple_returns = np.expm1(r)
    price_index = (1 + simple_returns).cumprod()
    dd_df = compute_drawdowns(price_index)
    max_dd = float(dd_df["drawdown"].min()) if not dd_df.empty else 0.0
    current_dd = float(dd_df["drawdown"].iloc[-1]) if not dd_df.empty else 0.0

    in_dd = dd_df["drawdown"] < 0 if not dd_df.empty else pd.Series(dtype=bool)
    longest = 0
    current = 0
    for val in in_dd:
        if val:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    mean_annual = r.mean() * freq
    sortino = mean_annual / semi_vol_annual if semi_vol_annual > 0 else 0.0
    return {
        "semi_vol_annual": semi_vol_annual,
        "max_drawdown": max_dd,
        "current_drawdown": current_dd,
        "longest_dd_days": longest,
        "sortino": sortino,
    }


def compute_downside_risk(prices: pd.Series, log_returns: pd.Series) -> DownsideRiskResult:
    negative_returns = log_returns[log_returns < 0]
    semi_variance = np.mean(negative_returns ** 2) if not negative_returns.empty else 0.0
    downside_vol_annual = math.sqrt(semi_variance) * math.sqrt(PERIODS_PER_YEAR)

    rolling_max = prices.cummax()
    drawdown = prices / rolling_max - 1
    max_drawdown_idx = drawdown.idxmin()
    max_drawdown_pct = float(drawdown.min())
    max_drawdown_start = rolling_max.loc[:max_drawdown_idx].idxmax() if not drawdown.empty else None
    current_drawdown_pct = float(drawdown.iloc[-1])

    longest_duration = 0
    current_duration = 0
    for value in drawdown:
        if value < 0:
            current_duration += 1
            longest_duration = max(longest_duration, current_duration)
        else:
            current_duration = 0

    equity_curve = prices / prices.iloc[0] * 100
    return DownsideRiskResult(
        downside_vol_annual=downside_vol_annual,
        max_drawdown_pct=max_drawdown_pct,
        max_drawdown_start=max_drawdown_start,
        max_drawdown_end=max_drawdown_idx,
        current_drawdown_pct=current_drawdown_pct,
        longest_duration_days=longest_duration,
        drawdown_series=drawdown,
        equity_curve=equity_curve,
    )


def build_drawdown_figure(risk_result: DownsideRiskResult) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=risk_result.equity_curve.index,
            y=risk_result.equity_curve,
            mode="lines",
            name="Equity Curve (Index=100)",
            line=dict(color="#26c6da", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=risk_result.drawdown_series.index,
            y=risk_result.drawdown_series * 100,
            mode="lines",
            name="Drawdown (%)",
            line=dict(color="#ef5350", width=2),
            yaxis="y2",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=40),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Equity Index (Start=100)"),
        yaxis2=dict(title="Drawdown %", overlaying="y", side="right"),
    )
    return fig


__all__ = [
    "compute_drawdowns",
    "compute_downside_stats",
    "compute_downside_risk",
    "build_drawdown_figure",
]
