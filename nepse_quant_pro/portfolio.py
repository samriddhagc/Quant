import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.cluster.hierarchy as sch
from typing import Optional, Tuple, Dict

from .config import PERIODS_PER_YEAR

def estimate_mean_cov(returns_df: pd.DataFrame, freq: int = PERIODS_PER_YEAR) -> Tuple[pd.Series, pd.DataFrame]:
    """Estimate annualized mean vector and covariance matrix from daily returns."""
    mu_daily = returns_df.mean()
    cov_daily = returns_df.cov()
    mu_annual = mu_daily * freq
    cov_annual = cov_daily * freq
    return mu_annual, cov_annual

def compute_portfolio_stats(weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame) -> Tuple[float, float]:
    port_return = float(np.dot(weights, mu.values))
    port_vol = float(np.sqrt(weights.T @ cov.values @ weights))
    return port_return, port_vol

# --- [NEW] BLACK-LITTERMAN OPTIMIZATION ---
def black_litterman_posterior(
    prior_mu: pd.Series, 
    cov_matrix: pd.DataFrame, 
    views: Dict[str, float], 
    view_confidences: Dict[str, float],
    tau: float = 0.05
) -> pd.Series:
    """
    Computes the Black-Litterman Posterior Expected Returns.
    
    Formula: E[R] = [(tau*Sigma)^-1 + P^T Omega^-1 P]^-1 * [(tau*Sigma)^-1 Pi + P^T Omega^-1 Q]
    
    Inputs:
    - prior_mu (Pi): Market Equilibrium Returns (What the market thinks)
    - views (Q): Your AI's view (e.g., {'NABIL': 0.40})
    - view_confidences: How sure the AI is (affects Omega)
    """
    tickers = cov_matrix.index.tolist()
    n = len(tickers)
    
    # 1. Setup Matrices
    Sigma = cov_matrix.values
    Pi = prior_mu.values.reshape(-1, 1)
    tau_Sigma = tau * Sigma
    
    # 2. Construct Views Matrices (P and Q)
    # P: Link matrix (identifies which asset corresponds to which view)
    # Q: View vector (the values)
    # Omega: Uncertainty matrix (diagonal)
    
    active_views = {k: v for k, v in views.items() if k in tickers}
    k = len(active_views)
    
    if k == 0:
        return prior_mu # No views, trust the market
        
    P = np.zeros((k, n))
    Q = np.zeros((k, 1))
    Omega = np.zeros((k, k))
    
    for i, (ticker, view_ret) in enumerate(active_views.items()):
        idx = tickers.index(ticker)
        P[i, idx] = 1
        Q[i, 0] = view_ret
        
        # Omega construction: Heuristic based on confidence
        # Lower confidence = Higher Variance (Uncertainty) in Omega
        # Standard heuristic: P * (tau * Sigma) * P^T / Confidence
        conf = view_confidences.get(ticker, 0.5)
        # Avoid division by zero
        uncertainty_scalar = (1.0 - conf) / (conf + 1e-6) 
        # Base variance of the asset
        asset_var = Sigma[idx, idx]
        Omega[i, i] = asset_var * tau * uncertainty_scalar

    # 3. Compute Posterior (The Black-Litterman Formula)
    # We use linear algebra inverse: inv(...)
    try:
        tau_Sigma_inv = np.linalg.inv(tau_Sigma)
        Omega_inv = np.linalg.inv(Omega)
        
        # M = (tau_Sigma_inv + P.T @ Omega_inv @ P)^-1
        M = np.linalg.inv(tau_Sigma_inv + P.T @ Omega_inv @ P)
        
        # Posterior Mean
        # term1 = tau_Sigma_inv @ Pi
        # term2 = P.T @ Omega_inv @ Q
        BL_mu = M @ (tau_Sigma_inv @ Pi + P.T @ Omega_inv @ Q)
        
        return pd.Series(BL_mu.flatten(), index=tickers)
        
    except Exception:
        # Fallback if matrix inversion fails
        return prior_mu

# --- HIERARCHICAL RISK PARITY (HRP) ---
def get_hrp_weights(cov_matrix: pd.DataFrame) -> pd.Series:
    corr = cov_matrix.corr()
    dist = np.sqrt((1 - corr) / 2)
    link = sch.linkage(dist, 'single')
    sort_ix = sch.leaves_list(link)
    sorted_cov = cov_matrix.iloc[sort_ix, sort_ix]
    inv_var = 1 / np.diag(sorted_cov)
    weights_vals = inv_var / inv_var.sum()
    hrp_series = pd.Series(weights_vals, index=sorted_cov.index)
    return hrp_series.reindex(cov_matrix.index)

# --- TRADITIONAL OPTIMIZERS ---
def min_variance_portfolio(cov: pd.DataFrame) -> Optional[np.ndarray]:
    n = cov.shape[0]
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.ones(n) / n
    def objective(w): return float(w.T @ cov.values @ w)
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x if res.success else None

def tangency_portfolio(mu: pd.Series, cov: pd.DataFrame, rf: float) -> Optional[np.ndarray]:
    n = len(mu)
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.ones(n) / n
    def negative_sharpe(w: np.ndarray) -> float:
        port_ret = float(np.dot(w, mu.values))
        port_vol = float(np.sqrt(w.T @ cov.values @ w))
        if port_vol < 1e-8: return 0.0
        return -((port_ret - rf) / port_vol)
    res = minimize(negative_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x if res.success else None

__all__ = [
    "estimate_mean_cov", "compute_portfolio_stats",
    "min_variance_portfolio", "tangency_portfolio",
    "get_hrp_weights", "black_litterman_posterior"
]