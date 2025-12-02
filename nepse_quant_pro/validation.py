from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


def _compute_equity_curve(trade_returns: pd.Series) -> pd.Series:
    if trade_returns is None or trade_returns.empty:
        return pd.Series(dtype=float)
    equity = (1.0 + trade_returns.fillna(0.0)).cumprod()
    equity.name = "equity"
    return equity


def _equity_payload(equity: pd.Series) -> Optional[list]:
    if equity is None or equity.empty:
        return None
    return [{"date": idx.isoformat(), "equity": float(val)} for idx, val in equity.items()]


def run_post_cv_validation(
    history_df: Optional[pd.DataFrame],
    log_returns: Optional[pd.Series],
    price_series: Optional[pd.Series],
    volume_series: Optional[pd.Series],
    threshold: float = 0.55,
    slippage_bps: float = 30.0,
    liquidity_quantile: float = 0.4,
) -> Dict[str, Any]:
    metrics: Dict[str, float] = {}
    if history_df is None or history_df.empty or log_returns is None or log_returns.empty:
        return {"passed": False, "reason": "Validation skipped (insufficient history).", "metrics": metrics, "equity": None}

    df = history_df.copy()
    df = df.join(log_returns.rename("log_ret"), how="inner")
    df["next_ret"] = df["log_ret"].shift(-1)
    if df["next_ret"].abs().sum() == 0 or df["next_ret"].isna().all():
        return {"passed": False, "reason": "Validation skipped (no future returns).", "metrics": metrics, "equity": None}

    if price_series is not None:
        df = df.join(price_series.rename("price"), how="left")
    if volume_series is not None:
        df = df.join(volume_series.rename("volume"), how="left")

    df["liquidity"] = df.get("price", 0.0) * df.get("volume", 0.0)
    liquidity_floor = float(df["liquidity"].quantile(liquidity_quantile)) if df["liquidity"].notna().any() else 0.0
    df["tradable"] = True
    if liquidity_floor > 0:
        df["tradable"] = df["liquidity"] >= liquidity_floor

    df["signal"] = ((df["Probability"] >= threshold).astype(int)) & df["tradable"]
    effective_slip = slippage_bps / 10000.0
    df["trade_ret"] = df["signal"] * (df["next_ret"].fillna(0.0) - effective_slip)
    trade_returns = df.loc[df["signal"], "trade_ret"]

    ic = df["Probability"].corr(df["next_ret"], method="spearman")
    metrics["ic"] = float(ic) if ic is not np.nan else None
    valid_ic = df[["Probability", "next_ret"]].dropna()
    n_eff = len(valid_ic)
    if metrics["ic"] is not None and n_eff > 2 and abs(metrics["ic"]) < 1:
        metrics["ic_tstat"] = float(metrics["ic"] * np.sqrt((n_eff - 2) / max(1e-6, (1 - metrics["ic"] ** 2))))
    else:
        metrics["ic_tstat"] = None

    trade_count = int(df["signal"].sum())
    metrics["trade_count"] = trade_count
    win_rate = float((trade_returns > 0).mean()) if trade_count > 0 else None
    metrics["win_rate"] = win_rate
    metrics["avg_trade_return"] = float(trade_returns.mean()) if trade_count > 0 else None

    equity = _compute_equity_curve(df["trade_ret"])
    if not equity.empty:
        total_periods = len(equity)
        total_return = float(equity.iloc[-1] - 1.0)
        metrics["total_return"] = total_return
        if total_periods > 1:
            metrics["cagr"] = float(equity.iloc[-1] ** (252 / total_periods) - 1.0)
        else:
            metrics["cagr"] = None
        drawdown = equity / equity.cummax() - 1.0
        metrics["max_drawdown"] = float(drawdown.min())
    else:
        metrics["total_return"] = None
        metrics["cagr"] = None
        metrics["max_drawdown"] = None

    effectiveness = [
        metrics["ic"] is not None and metrics["ic"] >= 0.02,
        metrics["ic_tstat"] is not None and metrics["ic_tstat"] >= 1.25,
        trade_count >= 25,
        metrics["cagr"] is not None and metrics["cagr"] >= 0.0,
    ]
    passed = all(effectiveness)
    reason = None
    if not passed:
        ic_display = f"{metrics['ic']:.3f}" if metrics["ic"] is not None else "NA"
        tstat_display = f"{metrics['ic_tstat']:.2f}" if metrics["ic_tstat"] is not None else "NA"
        reason = f"Post-validation failed (IC={ic_display}, t-stat={tstat_display}, trades={trade_count})."

    return {
        "passed": passed,
        "reason": reason,
        "metrics": metrics,
        "equity": _equity_payload(equity),
    }


__all__ = ["run_post_cv_validation"]
