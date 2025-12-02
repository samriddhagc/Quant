import numpy as np
import pandas as pd
from typing import Optional, Tuple


def compute_beta(stock_ret: pd.Series, index_ret: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute alpha, beta, and R^2 via OLS regression of stock versus index returns."""
    df = pd.concat([stock_ret, index_ret], axis=1).dropna()
    if df.shape[0] < 30:
        return None, None, None
    y = df.iloc[:, 0].values
    x = df.iloc[:, 1].values
    X = np.column_stack([np.ones(len(x)), x])
    try:
        beta_hat = np.linalg.inv(X.T @ X) @ (X.T @ y)
    except np.linalg.LinAlgError:
        return None, None, None
    alpha = beta_hat[0]
    beta = beta_hat[1]
    y_pred = X @ beta_hat
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = ((y - y_pred) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    return alpha, beta, r2


def build_hedged_returns(stock_ret: pd.Series, index_ret: pd.Series, beta: float) -> pd.Series:
    """Return hedged daily returns series: stock - beta * index."""
    df = pd.concat([stock_ret, index_ret], axis=1).dropna()
    if df.shape[0] == 0:
        return pd.Series(dtype=float)
    stock = df.iloc[:, 0]
    idx = df.iloc[:, 1]
    hedged = stock - beta * idx
    hedged.name = f"{stock.name}_hedged"
    return hedged


def estimate_capm_er(beta: float, market_er_annual: float, rf_annual: float) -> float:
    """CAPM expected return: E[Ri] = Rf + beta * (E[Rm] - Rf)."""
    return rf_annual + beta * (market_er_annual - rf_annual)


__all__ = ["compute_beta", "build_hedged_returns", "estimate_capm_er"]
