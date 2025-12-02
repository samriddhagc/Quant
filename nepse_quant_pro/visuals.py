import math
import os
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from typing import Optional, Tuple, List, Union

logger = logging.getLogger(__name__)
FAST_SIGNAL_MODE = os.environ.get("FAST_SIGNAL_MODE", "0") == "1"

def build_trend_figure(data: pd.DataFrame, short_window: int, long_window: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close", line=dict(color="#00bcd4", width=2)))
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA_short"], mode="lines", name=f"SMA {short_window}", line=dict(color="#ffd54f", width=2)))
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA_long"], mode="lines", name=f"SMA {long_window}", line=dict(color="#b39ddb", width=2)))
    if "Golden_Cross" in data.columns:
        crosses = data[data["Golden_Cross"]]
        if not crosses.empty:
            fig.add_trace(go.Scatter(x=crosses.index, y=crosses["Close"], mode="markers", name="Golden Cross", marker=dict(symbol="triangle-up", size=12, color="#00e676", line=dict(color="#ffffff", width=1))))
    fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=40, b=40), xaxis_title="Date", yaxis_title="Price", hovermode="x unified")
    return fig

def build_returns_histogram(log_returns: pd.Series) -> go.Figure:
    data = log_returns.dropna().values
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, nbinsx=50, histnorm="probability density", name="Empirical Returns", marker_color="#00bcd4", opacity=0.6))
    if len(data) > 2:
        x_vals = np.linspace(data.min(), data.max(), 200)
        pdf = norm.pdf(x_vals, loc=data.mean(), scale=data.std(ddof=1))
        fig.add_trace(go.Scatter(x=x_vals, y=pdf, mode="lines", name="Normal PDF", line=dict(color="#ffab40", width=2)))
    fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=40, b=40), xaxis_title="Daily Log Return", yaxis_title="Density")
    return fig

def build_qq_plot(log_returns: pd.Series) -> go.Figure:
    data = log_returns.dropna().values
    fig = go.Figure()
    if len(data) < 10: return fig
    sorted_returns = np.sort(data)
    n = len(sorted_returns)
    quantiles = (np.arange(1, n + 1) - 0.5) / n
    theoretical = norm.ppf(quantiles) * data.std(ddof=1) + data.mean()
    fig.add_trace(go.Scatter(x=theoretical, y=sorted_returns, mode="markers", name="Returns"))
    fig.add_trace(go.Scatter(x=theoretical, y=theoretical, mode="lines", name="45° Reference", line=dict(color="#ff7043", dash="dash")))
    fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=40, b=40), xaxis_title="Theoretical Quantiles", yaxis_title="Empirical Quantiles")
    return fig

def build_mc_paths_figure(paths: np.ndarray, title: str) -> go.Figure:
    days = paths.shape[0] - 1
    x = np.arange(days + 1)
    fig = go.Figure()
    sample_sims = min(paths.shape[1], 200)
    for i in range(sample_sims):
        fig.add_trace(go.Scatter(x=x, y=paths[:, i], mode="lines", line=dict(width=1), opacity=0.08, showlegend=False))
    fig.update_layout(title=title, template="plotly_dark", margin=dict(l=10, r=10, t=40, b=40), xaxis_title="Days into Future", yaxis_title="Simulated Price", hovermode="x")
    return fig

def build_volatility_figure(index: pd.DatetimeIndex, constant_sigma: float, ewma_sigma: Optional[pd.Series] = None, garch_sigma: Optional[pd.Series] = None) -> go.Figure:
    from nepse_quant_pro.config import PERIODS_PER_YEAR
    fig = go.Figure()
    const = constant_sigma * math.sqrt(PERIODS_PER_YEAR)
    fig.add_trace(go.Scatter(x=index, y=[const] * len(index), mode="lines", name="Constant σ", line=dict(color="#ffd54f", width=2, dash="dash")))
    if ewma_sigma is not None:
        fig.add_trace(go.Scatter(x=ewma_sigma.index, y=ewma_sigma * math.sqrt(PERIODS_PER_YEAR), mode="lines", name="EWMA σ", line=dict(color="#00bcd4", width=2)))
    if garch_sigma is not None:
        fig.add_trace(go.Scatter(x=garch_sigma.index, y=garch_sigma * math.sqrt(PERIODS_PER_YEAR), mode="lines", name="GARCH σ", line=dict(color="#ef5350", width=2)))
    fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=40, b=40), xaxis_title="Date", yaxis_title="Annualized Volatility", hovermode="x unified")
    return fig

def build_strategy_backtest_chart(
    history: pd.DataFrame,
    log_returns: pd.Series,
    prices: Union[pd.Series, pd.DataFrame],
    atr_series: pd.Series,
    atr_mean: float,
    initial_capital: float = 10000.0,
    cost_per_trade: float = 0.003,
    trailing_stop_mult: float = 6.0,
    prob_entry_thresh: float = 0.52,
    min_exposure: float = 0.0,
    asset_type: str = "LIQUID_STOCK",
    regime_window: int = 200,
    enable_liquidity: bool = True,
    slippage_pct: float = 0.002,
    settlement_days: int = 2,
    model_confidence: Optional[float] = None,
    fast_signal_mode: Optional[bool] = None,
) -> Tuple[go.Figure, go.Figure, dict]:
    """
    NEPSE-Compliant Backtester with Robust Error Handling.
    """
    # [CRASH FIX]: Guard Clause for Insufficient Data
    exec_stats_placeholder = {"error": "Not enough data"}
    if history is None or history.empty:
        # Return empty placeholder figures if the model failed to train
        fig = go.Figure()
        fig.update_layout(
            title="Strategy Backtest (Insufficient Data)",
            template="plotly_dark",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(text="Not enough signal data to run backtest.", showarrow=False, font=dict(size=20))]
        )
        return fig, go.Figure(), exec_stats_placeholder

    if isinstance(prices, pd.Series): df_prices = prices.to_frame(name="Close")
    elif isinstance(prices, pd.DataFrame): df_prices = prices.copy()
    else: return go.Figure(), go.Figure(), exec_stats_placeholder

    col_map = {c.lower(): c for c in df_prices.columns}
    if "close" in col_map: price_col = col_map["close"]
    elif "adj close" in col_map: price_col = col_map["adj close"]
    else: candidates = [c for c in df_prices.columns if "vol" not in c.lower()]; price_col = candidates[0] if candidates else df_prices.columns[0]

    df_prices = df_prices.rename(columns={price_col: "Close"})
    if "High" not in df_prices.columns: df_prices["High"] = df_prices["Close"]
    if "Low" not in df_prices.columns: df_prices["Low"] = df_prices["Close"]
    if "Volume" not in df_prices.columns: df_prices["Volume"] = 100000

    df = history[["Probability"]].join(log_returns.rename("LogRet"), how="inner")
    df = df.join(df_prices[["Close", "High", "Low", "Volume"]], how="inner")
    if atr_series is not None: df = df.join(atr_series.rename("ATR"), how="left")
    if df.empty or len(df) < 5: return go.Figure(), go.Figure(), exec_stats_placeholder

    df["ATR"] = df["ATR"].ffill().fillna(0.0)
    df["SimpleRet"] = np.exp(df["LogRet"]) - 1
    
    # --- 1. DYNAMIC THRESHOLD ---
    # Buy if confidence is in the top 20% of the last quarter (60 days)
    df["Smoothed_Prob"] = df["Probability"].ewm(alpha=0.3, adjust=False).mean()
    df["Dynamic_Thresh"] = df["Smoothed_Prob"].rolling(window=60).quantile(0.70)
    df["Dynamic_Thresh"] = df["Dynamic_Thresh"].fillna(prob_entry_thresh)
    conf = 0.60
    if model_confidence is not None:
        conf = float(np.clip(model_confidence, 0.35, 0.85))
    fast_signals = FAST_SIGNAL_MODE if fast_signal_mode is None else bool(fast_signal_mode)
    buffer_anchor = 0.53 if fast_signals else 0.60
    confidence_shift = (buffer_anchor - conf) * 0.20
    df["Dynamic_Thresh"] = df["Dynamic_Thresh"] + confidence_shift
    df["Dynamic_Thresh"] = df["Dynamic_Thresh"].clip(lower=0.50, upper=0.80)
    logger.debug(
        "[Backtest] entry=%.3f | dynamic_range=(%.3f, %.3f) | fast_mode=%s",
        prob_entry_thresh,
        float(df["Dynamic_Thresh"].min()),
        float(df["Dynamic_Thresh"].max()),
        fast_signals,
    )

    is_index = (asset_type in ["MEGA_INDEX", "EM_INDEX"])
    MIN_VOLUME_REQ = 1 if is_index else 100
    if not enable_liquidity: MIN_VOLUME_REQ = -1
    SLIPPAGE = slippage_pct 
    
    start_index = max(1, regime_window)
    settled_cash = initial_capital
    unsettled_cash_queue = []
    demat_shares = 0.0
    unsettled_shares_queue = []
    equity_curve = [initial_capital]
    position_series: List[float] = []
    trailing_stop_price = np.nan
    
    trade_events = []

    exec_stats = {
        "signals_seen": 0,
        "signals_confirmed": 0,
        "entries": 0,
        "skipped_threshold": 0,
        "skipped_liquidity": 0,
        "skipped_hysteresis": 0,
        "skipped_other": 0,
    }

    confirmation_req = 1 if fast_signals else 2
    if not fast_signals and conf >= 0.65:
        confirmation_req = 1

    debug_rows: List[dict] = []

    for i in range(start_index, len(df)):
        # --- 2. SETTLEMENT ---
        cleared_cash = 0.0
        remaining_cash_queue = []
        for amount, settle_idx in unsettled_cash_queue:
            if i >= settle_idx: cleared_cash += amount
            else: remaining_cash_queue.append((amount, settle_idx))
        settled_cash += cleared_cash
        unsettled_cash_queue = remaining_cash_queue
        
        cleared_shares = 0.0
        remaining_shares_queue = []
        for shares, settle_idx in unsettled_shares_queue:
            if i >= settle_idx: cleared_shares += shares
            else: remaining_shares_queue.append((shares, settle_idx))
        demat_shares += cleared_shares
        unsettled_shares_queue = remaining_shares_queue

        # Data Points
        current_close = float(df["Close"].iloc[i])
        if current_close < 0.5 and not is_index: 
            equity_curve.append(equity_curve[-1]); position_series.append(0.0); continue

        current_open = float(df["Close"].iloc[i-1])
        current_low = float(df["Low"].iloc[i])
        current_vol = float(df["Volume"].iloc[i])
        
        # Signals
        signal_prob = float(df["Smoothed_Prob"].iloc[i - 1])
        current_thresh = float(df["Dynamic_Thresh"].iloc[i - 1])
        
        # Risk Management
        current_atr = float(df["ATR"].iloc[i])
        if current_atr <= 0 or not np.isfinite(current_atr): current_atr = current_close * 0.02
        stop_distance = current_atr * trailing_stop_mult
        
        # Portfolio Value
        total_pending_shares = sum(s for s, _ in unsettled_shares_queue)
        total_shares_owned = demat_shares + total_pending_shares
        total_pending_cash = sum(c for c, _ in unsettled_cash_queue)
        total_cash_value = settled_cash + total_pending_cash
        current_equity = total_cash_value + (total_shares_owned * current_close)
        
        position_held = (demat_shares > 0)
        action = "HOLD"
        executed_price = current_close

        # --- 3. DECISION LOGIC (HYSTERESIS) ---
        # "Sticky Signal": Needs 3 days of high conviction to enter
        recent_probs = df["Smoothed_Prob"].iloc[max(0, i - confirmation_req):i]
        if len(recent_probs) >= confirmation_req and (recent_probs > current_thresh).all():
            signal_confirmed = True
        else:
            signal_confirmed = False
            exec_stats["skipped_hysteresis"] += 1

        if position_held:
            if current_vol < MIN_VOLUME_REQ: action = "HOLD_STUCK"
            elif current_low <= trailing_stop_price:
                action = "SELL_TSL"
                exit_price = min(trailing_stop_price, current_open) 
                executed_price = exit_price * (1 - SLIPPAGE)
            # Exit if probability trend breaks (drops below 50% neutral)
            elif signal_prob < 0.50: 
                action = "SELL_PROB"
                executed_price = current_close * (1 - SLIPPAGE)
            else:
                action = "HOLD"
                new_stop = current_close - stop_distance
                trailing_stop_price = max(trailing_stop_price, new_stop)
        else:
            exec_stats["signals_seen"] += 1
            if signal_confirmed:
                exec_stats["signals_confirmed"] += 1
                if settled_cash > (current_close * 10):
                    if current_vol >= MIN_VOLUME_REQ:
                        action = "BUY"
                        executed_price = current_close * (1 + SLIPPAGE)
                        trailing_stop_price = current_close - stop_distance
                    else:
                        action = "NO_FILL"
                        exec_stats["skipped_liquidity"] += 1
                else:
                    action = "WAIT"
                    exec_stats["skipped_other"] += 1
            else:
                action = "WAIT"
                exec_stats["skipped_threshold"] += 1

        debug_rows.append(
            {
                "date": df.index[i],
                "smoothed_prob": signal_prob,
                "dynamic_thresh": current_thresh,
                "confirmed": signal_confirmed,
                "action": action,
                "position_held": position_held,
            }
        )

        # --- 4. EXECUTION (VOLATILITY SCALING) ---
        if action == "BUY":
            if executed_price > 0:
                # Risk Parity Sizing: Risk 1% of equity per trade
                risk_amount = current_equity * 0.01 
                # How many shares can I buy such that if I hit stop loss, I only lose risk_amount?
                # Loss per share = Stop Distance
                # Qty = Risk / Stop Distance
                risk_based_qty = risk_amount / max(stop_distance, 0.01)
                
                # Cap at available cash (No leverage)
                max_affordable = settled_cash / (executed_price * (1 + cost_per_trade))
                
                # Final Qty is the smaller of Risk-Based or Cash-Based
                final_qty = math.floor(min(risk_based_qty, max_affordable))
                
                # Minimum size filter (don't buy 1 share)
                if final_qty > 5:
                    total_outflow = (final_qty * executed_price) * (1 + cost_per_trade)
                    settled_cash -= total_outflow
                    unsettled_shares_queue.append((final_qty, i + settlement_days))
                    trade_events.append({"date": df.index[i], "equity": current_equity, "type": "BUY", "price": executed_price, "shares": final_qty, "desc": f"Prob {signal_prob:.2f} (Vol-Scaled)"})
                    exec_stats["entries"] += 1
        
        elif action in ["SELL_TSL", "SELL_PROB"]:
            if demat_shares > 0:
                net_inflow = (demat_shares * executed_price) * (1 - cost_per_trade)
                shares_sold = demat_shares 
                demat_shares = 0.0
                unsettled_cash_queue.append((net_inflow, i + settlement_days))
                trailing_stop_price = np.nan
                trade_events.append({"date": df.index[i], "equity": current_equity, "type": "SELL", "price": executed_price, "shares": shares_sold, "desc": "Stop Loss" if action == "SELL_TSL" else "Signal Exit"})

        # Record Position for Chart
        # Recalculate owned for position sizing graph
        total_pending_shares = sum(s for s, _ in unsettled_shares_queue)
        total_shares_owned = demat_shares + total_pending_shares
        
        # Exposure % = Market Value of Shares / Total Equity
        market_val = total_shares_owned * current_close
        exposure_pct = market_val / current_equity if current_equity > 0 else 0
        
        equity_curve.append(current_equity)
        position_series.append(exposure_pct)

    # --- 5. FINALIZE ---
    df = df.iloc[start_index:].copy()
    if df.empty: return go.Figure(), go.Figure(), exec_stats_placeholder

    df["Equity_Strat"] = equity_curve[1:]
    market_shares = initial_capital / df["Close"].iloc[0]
    df["Equity_Market"] = market_shares * df["Close"]
    df["Position_Size"] = position_series

    if initial_capital > 0:
        perf_strat = (df["Equity_Strat"].iloc[-1] / initial_capital) - 1
        perf_market = (df["Equity_Market"].iloc[-1] / initial_capital) - 1
    else: perf_strat = 0.0; perf_market = 0.0

    buys = [t for t in trade_events if t["type"] == "BUY"]
    sells = [t for t in trade_events if t["type"] == "SELL"]
    total_bought = sum(b["shares"] for b in buys)
    total_sold = sum(s["shares"] for s in sells)
    shares_held = demat_shares + sum(s for s, _ in unsettled_shares_queue)

    exec_stats.update(
        {
            "entry_threshold": prob_entry_thresh,
            "dynamic_threshold_min": float(df["Dynamic_Thresh"].min()),
            "dynamic_threshold_max": float(df["Dynamic_Thresh"].max()),
            "fast_signal_mode": fast_signals,
            "probability_curve": df[["Smoothed_Prob", "Dynamic_Thresh"]]
            .tail(200)
            .reset_index()
            .to_dict(orient="records"),
            "debug_rows": debug_rows,
        }
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Equity_Market"], mode="lines", name=f"Market ({perf_market:+.1%})", line=dict(color="gray", width=2, dash="dash"), opacity=0.6))
    line_color = "#00e676" if perf_strat >= 0 else "#ff1744"
    fig.add_trace(go.Scatter(x=df.index, y=df["Equity_Strat"], mode="lines", name=f"AI Strategy ({perf_strat:+.1%})", line=dict(color=line_color, width=2)))
    
    # Add Dynamic Threshold visualization
    fig.add_trace(go.Scatter(x=df.index, y=df["Dynamic_Thresh"], mode="lines", name="Buy Threshold (Dynamic)", line=dict(color="#ab47bc", width=1, dash="dot"), yaxis="y2"))
    
    if buys: fig.add_trace(go.Scatter(x=[b["date"] for b in buys], y=[b["equity"] for b in buys], mode="markers", name="Buy Signal", marker=dict(symbol="triangle-up", size=12, color="#00e676", line=dict(width=1, color="black")), customdata=np.stack(([b["price"] for b in buys], [b["desc"] for b in buys], [b["shares"] for b in buys]), axis=-1), hovertemplate="<b>BUY</b><br>Date: %{x}<br>Equity: %{y:,.0f}<br>Price: %{customdata[0]:,.2f}<br>Shares: %{customdata[2]:,.0f}<br>Reason: %{customdata[1]}<extra></extra>"))
    if sells: fig.add_trace(go.Scatter(x=[s["date"] for s in sells], y=[s["equity"] for s in sells], mode="markers", name="Sell Signal", marker=dict(symbol="triangle-down", size=12, color="#ff1744", line=dict(width=1, color="black")), customdata=np.stack(([s["price"] for s in sells], [s["desc"] for s in sells], [s["shares"] for s in sells]), axis=-1), hovertemplate="<b>SELL</b><br>Date: %{x}<br>Equity: %{y:,.0f}<br>Price: %{customdata[0]:,.2f}<br>Shares Sold: %{customdata[2]:,.0f}<br>Reason: %{customdata[1]}<extra></extra>"))

    realism_tag = "ON" if enable_liquidity else "OFF"
    title_text = f"<b>NEPSE T+2 Backtest</b> (Settlement: 2 Days) + Volatility Scaled<br><span style='font-size:10px'>Liquidity: {realism_tag} | Cost: {cost_per_trade*100:.1f}% | <b>Shares Bought: {total_bought:,.0f} | Sold: {total_sold:,.0f}</b> (Held: {shares_held:,.0f})</span>"
    
    fig.update_layout(
        title=dict(text=title_text), 
        xaxis_title="Date", 
        yaxis_title="Portfolio Equity", 
        template="plotly_dark", 
        margin=dict(l=20, r=20, t=60, b=40), 
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"), 
        hovermode="x unified",
        yaxis2=dict(title="Probability Threshold", overlaying="y", side="right", showgrid=False, range=[0, 1])
    )
    allocation_fig = build_allocation_chart(df)
    return fig, allocation_fig, exec_stats

def build_allocation_chart(backtest_df: pd.DataFrame) -> go.Figure:
    if "Position_Size" not in backtest_df.columns: backtest_df = backtest_df.assign(Position_Size=0.0)
    avg_exposure = float(backtest_df["Position_Size"].mean())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["Position_Size"] * 100, mode="lines", name=f"Avg Exposure: {avg_exposure*100:.1f}%", line=dict(color="#8c9eff", width=1.5), fill="tozeroy"))
    fig.update_layout(title="Strategy Capital Allocation History (Risk-Scaled)", xaxis_title="Date", yaxis_title="Market Exposure (%)", yaxis=dict(range=[0, 105]), template="plotly_dark", margin=dict(l=10, r=10, t=40, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

__all__ = ["build_trend_figure", "build_returns_histogram", "build_qq_plot", "build_mc_paths_figure", "build_volatility_figure", "build_strategy_backtest_chart", "build_allocation_chart"]
