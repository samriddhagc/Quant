import math
import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm, t as student_t, kurtosis
from scipy.optimize import minimize_scalar
from numba import jit

from .config import (
    PERIODS_PER_YEAR,
    RISK_AVERSION_LAMBDA,
    NO_TRADE_UTILITY_THRESHOLD,
    BLOCK_BOOTSTRAP_SIZE,
    R_F_ANNUAL,
    MAX_POS_ER,
    MAX_NEG_ER,
    ES_PENALTY_STRESSED,
    STRESS_ALLOC_CAP
)
from .types import DecisionResult, RegimeResult

logger = logging.getLogger(__name__)

# --- DYNAMIC REGIME PARAMETERS ---
REGIME_PARAMS = {
    "index": {
        "bull": {"prob_threshold": 0.52, "min_exposure": 0.80},
        "neutral": {"prob_threshold": 0.55, "min_exposure": 0.50},
        "bear": {"prob_threshold": 0.60, "min_exposure": 0.20},
    },
    "emerging_stock": {
        "bull": {"prob_threshold": 0.56, "min_exposure": 0.50},
        "neutral": {"prob_threshold": 0.60, "min_exposure": 0.20},
        "bear": {"prob_threshold": 0.65, "min_exposure": 0.00},
    },
}

def get_dynamic_params(asset_type: str, regime: str) -> dict:
    asset_key = "index" if asset_type == "index" else "emerging_stock"
    asset_params = REGIME_PARAMS.get(asset_key, REGIME_PARAMS["emerging_stock"])
    return asset_params.get(regime, asset_params["neutral"])

# --- ROBUST DRIFT ESTIMATION (James-Stein Shrinkage) ---
def james_stein_drift(stock_returns: pd.Series, market_returns: Optional[pd.Series] = None, annualized: bool = True) -> float:
    mu_stock = stock_returns.mean()
    if market_returns is not None:
        mu_target = market_returns.mean()
    else:
        mu_target = 0.0 
        
    var_stock = stock_returns.var()
    n = len(stock_returns)
    distance = (mu_stock - mu_target) ** 2
    
    if distance < 1e-9:
        lambda_shrink = 0.0
    else:
        lambda_shrink = min(1.0, var_stock / (n * distance))
        
    mu_est = (1 - lambda_shrink) * mu_stock + lambda_shrink * mu_target
    
    if annualized:
        return mu_est * PERIODS_PER_YEAR
    return mu_est

# ==========================================
# NUMBA KERNELS (High Performance loops)
# ==========================================

@jit(nopython=True, cache=True)
def _mc_kernel(
    current_price, days, sims, dt, 
    step_mu, step_sigma, rho, 
    dist_flag, # 0=Normal, 1=Student-t
    df,
    model_flag, # 0=GBM, 1=OU, 2=Jump
    kappa, theta_log, 
    lambda_jump, jump_mean, jump_std,
    use_sv, garch_vol_arr
):
    paths = np.zeros((days + 1, sims))
    paths[0, :] = current_price
    
    prev_z = np.zeros(sims)
    jump_prob = lambda_jump * dt
    
    t_scale = 1.0
    if dist_flag == 1 and df > 2:
        t_scale = math.sqrt((df - 2) / df)

    for t in range(1, days + 1):
        if dist_flag == 1:
            eps = np.random.standard_t(df, sims) * t_scale
        else:
            eps = np.random.standard_normal(sims)

        if abs(rho) > 1e-6:
            z = rho * prev_z + eps * math.sqrt(1 - rho**2)
        else:
            z = eps
        prev_z = z 

        sigma_t = step_sigma
        if use_sv and garch_vol_arr is not None:
            idx = min(t - 1, len(garch_vol_arr) - 1)
            sigma_t = garch_vol_arr[idx]

        if model_flag == 1: # OU
            prev_prices = np.maximum(paths[t - 1, :], 1e-9)
            prev_log = np.log(prev_prices)
            drift = kappa * (theta_log - prev_log) * dt
            diffusion = sigma_t * z
            paths[t, :] = np.exp(prev_log + drift + diffusion)
            
        else: # GBM / Jump
            drift = step_mu - 0.5 * sigma_t**2
            diffusion = sigma_t * z
            total_log_ret = drift + diffusion
            
            if model_flag == 2 or lambda_jump > 0:
                jumps = np.random.poisson(jump_prob, sims)
                if np.any(jumps):
                    jump_mags = np.random.normal(jump_mean, jump_std, sims) * jumps
                    total_log_ret += jump_mags

            paths[t, :] = paths[t - 1, :] * np.exp(total_log_ret)

    return paths

@jit(nopython=True, cache=True)
def _bootstrap_kernel(current_price, returns, days, sims, block_size):
    paths = np.zeros((days + 1, sims))
    paths[0, :] = current_price
    n_ret = len(returns)
    
    for s in range(sims):
        current_idx = 1
        while current_idx <= days:
            start = np.random.randint(0, n_ret - block_size)
            remaining = days - current_idx + 1
            take = min(block_size, remaining)
            for i in range(take):
                r = returns[start + i]
                paths[current_idx, s] = paths[current_idx - 1, s] * (1 + r)
                current_idx += 1    
    return paths

# --- EXTENDED MONTE CARLO CORE ---
def run_monte_carlo_paths(
    current_price: float,
    annual_mu: float,
    annual_sigma: float,
    days: int,
    sims: int,
    distribution: str = "Normal",
    df: int = 5,
    return_generation_method: str = "GBM",
    rho: float = 0.0,
    garch_vol: Optional[pd.Series] = None,
    hist_returns: Optional[pd.Series] = None,
    daily_mu: Optional[float] = None,
    daily_sigma: Optional[float] = None,
    kappa: float = 0.0,
    theta: Optional[float] = None,
    lambda_jump: float = 0.0,
    jump_mean: float = 0.0,
    jump_std: float = 0.0,
) -> np.ndarray:
    dt = 1.0 / PERIODS_PER_YEAR
    step_mu = daily_mu if daily_mu is not None else (annual_mu / PERIODS_PER_YEAR)
    step_sigma = daily_sigma if daily_sigma is not None else (annual_sigma / math.sqrt(PERIODS_PER_YEAR))
    
    theta_val = theta if theta is not None else current_price
    theta_log = math.log(theta_val) if theta_val > 0 else 0.0

    dist_flag = 1 if distribution == "Student-t (fat-tailed)" else 0
    
    model_flag = 0
    if return_generation_method == "MeanReversion" or return_generation_method == "Mean Reverting (OU)":
        model_flag = 1
    elif return_generation_method == "JumpDiffusion" or return_generation_method == "Crash Risk (Jump Diffusion)":
        model_flag = 2

    use_sv = False
    garch_arr = np.array([], dtype=np.float64)
    if return_generation_method == "StochVol" or return_generation_method == "Growth (Stochastic Vol)":
        if garch_vol is not None and not garch_vol.empty:
            use_sv = True
            garch_arr = garch_vol.values.astype(np.float64)
    
    if return_generation_method == "BlockBootstrap" or return_generation_method == "Block Bootstrap":
        if hist_returns is None or hist_returns.empty:
            return np.tile(current_price, (days + 1, sims))
        ret_arr = hist_returns.dropna().values.astype(np.float64)
        block_size = min(int(BLOCK_BOOTSTRAP_SIZE), days, len(ret_arr))
        return _bootstrap_kernel(float(current_price), ret_arr, int(days), int(sims), int(block_size))
    else:
        return _mc_kernel(
            float(current_price), int(days), int(sims), float(dt),
            float(step_mu), float(step_sigma), float(rho),
            int(dist_flag), int(df), int(model_flag),
            float(kappa), float(theta_log),
            float(lambda_jump), float(jump_mean), float(jump_std),
            use_sv, garch_arr
        )

def compute_var_cvar(returns: np.ndarray, confidence: float) -> Tuple[float, float]:
    if returns is None or len(returns) == 0:
        return 0.0, 0.0
    log_r = np.log1p(returns)
    percentile = (1 - confidence) * 100
    var_log = float(np.percentile(log_r, percentile))
    tail = log_r[log_r <= var_log]
    es_log = float(tail.mean()) if tail.size > 0 else var_log
    var_simple = math.expm1(var_log)
    es_simple = math.expm1(es_log)
    var_loss = abs(min(var_simple, 0.0))
    es_loss = abs(min(es_simple, 0.0))
    return var_loss, es_loss

def build_risk_table(terminal_returns: np.ndarray) -> pd.DataFrame:
    mean_ret = float(np.mean(terminal_returns))
    median_ret = float(np.median(terminal_returns))
    var_95, cvar_95 = compute_var_cvar(terminal_returns, 0.95)
    var_99, cvar_99 = compute_var_cvar(terminal_returns, 0.99)
    return pd.DataFrame([
        {"Metric": "Mean Return", "95%": mean_ret, "99%": mean_ret},
        {"Metric": "Median Return", "95%": median_ret, "99%": median_ret},
        {"Metric": "VaR (one-sided)", "95%": var_95, "99%": var_99},
        {"Metric": "CVaR / Expected Shortfall", "95%": cvar_95, "99%": cvar_99},
    ])

def identify_drawdown_windows(prices: pd.Series, top_n: int = 3) -> List[dict]:
    rows = []
    rolling_max = prices.cummax()
    peak_date = prices.index[0]
    peak_value = prices.iloc[0]
    for date, price in prices.items():
        if price >= peak_value:
            peak_value = price
            peak_date = date
        drawdown = price / peak_value - 1
        rows.append({"date": date, "drawdown": drawdown, "peak_date": peak_date})
    dd_df = pd.DataFrame(rows).set_index("date")
    worst = dd_df.nsmallest(top_n, "drawdown")
    scenarios = []
    for idx, row in worst.iterrows():
        scenarios.append({
            "label": f"{row['peak_date'].date()} ➜ {idx.date()} ({row['drawdown']*100:.1f}%)",
            "start": row["peak_date"],
            "end": idx,
            "shock": float(row["drawdown"]),
        })
    return scenarios

def apply_stress_scenario(final_prices: np.ndarray, shock: float) -> np.ndarray:
    return final_prices * (1 + shock)

def build_terminal_histogram(final_prices: np.ndarray, baseline_mean_price: float, var95_price: float, scenario_lines: List[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=final_prices, nbinsx=60, histnorm="probability",
        name="Terminal Prices", marker_color="#00bcd4", opacity=0.7
    ))
    fig.add_vline(x=baseline_mean_price, line=dict(color="#ffd54f", width=2), annotation_text="Mean", annotation_position="top")
    fig.add_vline(x=var95_price, line=dict(color="#ff1744", width=2, dash="dot"), annotation_text="95% VaR", annotation_position="top")
    colors = ["#26c6da", "#66bb6a", "#ab47bc", "#ffa726"]
    for idx, line in enumerate(scenario_lines):
        fig.add_vline(
            x=line["price"], line=dict(color=colors[idx % len(colors)], dash="dash"),
            annotation_text=line["name"], annotation_position="top",
        )
    fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=40, b=40), xaxis_title="Terminal Price", yaxis_title="Probability")
    return fig

def estimate_drift_er(log_returns: pd.Series, horizon_days: int, min_obs: int = 200) -> float:
    r = log_returns.dropna().astype(float)
    if len(r) < min_obs or horizon_days <= 0: return 0.0
    mu_hat = r.mean()
    sigma_hat = r.std()
    T = horizon_days / PERIODS_PER_YEAR
    drift_term = mu_hat + 0.5 * (sigma_hat**2)
    return max(min(math.exp(drift_term * T) - 1.0, 2.0), -0.99)

def estimate_mc_er(simulated_terminal_prices: np.ndarray, current_price: float) -> float:
    if simulated_terminal_prices is None or len(simulated_terminal_prices) == 0 or current_price <= 0: return 0.0
    terminal_returns = simulated_terminal_prices / current_price - 1.0
    return max(min(float(np.median(terminal_returns)), 2.0), -0.99)

# --- CANONICAL ENGINE ---
def run_canonical_engine(
    current_price: float,
    sigma_daily: float,
    er_annual: float,
    rf_annual: float,
    horizon_days: int,
    sims: int,
    es_conf_level: float,
    distribution: str,
    student_df: int,
    return_generation_method: str,
    rho: float = 0.0,
    garch_vol: Optional[pd.Series] = None,
    kappa: float = 0.0,
    theta: Optional[float] = None,
    lambda_jump: float = 0.0,
    jump_mean: float = 0.0,
    jump_std: float = 0.0,
    er_recent_annual: float = 0.0,
    er_capm_annual: float = 0.0,
    shrinkage_factor: float = 0.0,
    capm_weight: float = 0.0,
    hist_returns: Optional[pd.Series] = None,
) -> dict:
    if horizon_days <= 0 or sims <= 0 or current_price <= 0:
        return {
            "er_blended": 0.0, "paths": np.zeros((1, 1)), "terminal_returns": np.zeros(1),
            "mc_expected_return": 0.0, "mc_expected_shortfall": 0.0, "prob_gain": 0.5,
            "es_annual": 0.0, "sim_kurtosis": 0.0
        }

    model_er = max(min(er_annual, MAX_POS_ER), MAX_NEG_ER)
    er_baseline = max(model_er, -0.99)

    daily_mu = math.log1p(er_baseline) / PERIODS_PER_YEAR
    annual_mu = daily_mu * PERIODS_PER_YEAR
    annual_sigma = sigma_daily * math.sqrt(PERIODS_PER_YEAR)

    paths = run_monte_carlo_paths(
        current_price=current_price,
        annual_mu=annual_mu,
        annual_sigma=annual_sigma,
        days=int(horizon_days),
        sims=int(sims),
        distribution=distribution,
        df=student_df,
        return_generation_method=return_generation_method,
        rho=rho,
        garch_vol=garch_vol,
        daily_mu=daily_mu,
        daily_sigma=sigma_daily,
        kappa=kappa,
        theta=theta,
        lambda_jump=lambda_jump,
        jump_mean=jump_mean,
        jump_std=jump_std,
        hist_returns=hist_returns,
    )

    terminal_prices = paths[-1, :]
    terminal_returns = terminal_prices / current_price - 1
    
    mc_mean_return = float(np.mean(terminal_returns)) if terminal_returns.size else 0.0
    _, es_horizon = compute_var_cvar(terminal_returns, es_conf_level) if terminal_returns.size else (0.0, 0.0)
    prob_gain = float(np.mean(terminal_returns > 0)) if terminal_returns.size else 0.0
    sim_kurtosis = float(kurtosis(terminal_returns)) if terminal_returns.size > 0 else 0.0
    es_annual = es_horizon * math.sqrt(PERIODS_PER_YEAR / max(float(horizon_days), 1.0))

    final_decision_er = mc_mean_return if return_generation_method in ["MeanReversion", "BlockBootstrap", "JumpDiffusion"] else model_er

    return {
        "er_blended": final_decision_er,
        "er_used": model_er,
        "paths": paths,
        "terminal_returns": terminal_returns,
        "mc_expected_return": mc_mean_return,
        "mc_expected_shortfall": es_horizon,
        "prob_gain": prob_gain,
        "es_annual": es_annual,
        "sim_kurtosis": sim_kurtosis 
    }

def compute_horizon_er_es(
    price_series: pd.Series,
    horizon_days: int,
    annualized_mu: float,
    annualized_sigma: float,
    mc_sims: int,
    es_conf_level: float,
    use_student_t: bool = True,
    blend_drift_and_mc: bool = True,
    student_df: int = 8,
    return_generation_method: str = "GBM",
    rho: float = 0.0,
    garch_vol: Optional[pd.Series] = None,
    hist_returns: Optional[pd.Series] = None,
) -> Tuple[float, float]:
    current_price = float(price_series.iloc[-1])
    paths_normal = run_monte_carlo_paths(
        current_price=current_price,
        annual_mu=annualized_mu,
        annual_sigma=annualized_sigma,
        days=int(horizon_days),
        sims=int(mc_sims),
        distribution="Normal",
        return_generation_method=return_generation_method,
        rho=rho,
        garch_vol=garch_vol,
        hist_returns=hist_returns,
    )
    terminal_returns_normal = paths_normal[-1, :] / current_price - 1
    er_mc = float(np.mean(terminal_returns_normal))
    _, es_normal = compute_var_cvar(terminal_returns_normal, es_conf_level)

    es_t = es_normal
    if use_student_t and return_generation_method != "BlockBootstrap":
        paths_t = run_monte_carlo_paths(
            current_price=current_price,
            annual_mu=annualized_mu,
            annual_sigma=annualized_sigma,
            days=int(horizon_days),
            sims=int(mc_sims),
            distribution="Student-t (fat-tailed)",
            df=student_df,
            return_generation_method=return_generation_method,
            rho=rho,
            garch_vol=garch_vol,
            hist_returns=hist_returns,
        )
        terminal_returns_t = paths_t[-1, :] / current_price - 1
        _, es_t = compute_var_cvar(terminal_returns_t, es_conf_level)
        er_mc = float(np.mean(terminal_returns_t))

    es_blend = 0.5 * es_normal + 0.5 * es_t if use_student_t else es_normal
    daily_mu = annualized_mu / PERIODS_PER_YEAR
    daily_sigma = annualized_sigma / math.sqrt(PERIODS_PER_YEAR)
    T = horizon_days / PERIODS_PER_YEAR
    er_drift = math.exp((daily_mu + 0.5 * daily_sigma**2) * T) - 1.0

    expected_return = 0.5 * er_drift + 0.5 * er_mc if blend_drift_and_mc else er_mc
    return expected_return, es_blend

def evaluate_failure_probability(final_prices: np.ndarray, threshold_price: float) -> float:
    return float(np.mean(final_prices <= threshold_price))


# --- CRRA UTILITY OPTIMIZATION (THE BRAIN UPGRADE) ---

def get_crra_utility(wealth: np.ndarray, gamma: float) -> float:
    """
    Calculates the CRRA Utility of a wealth distribution.
    U(W) = (W^(1-gamma)) / (1-gamma)
    """
    # Safety: Wealth <= 0 implies ruin. Clamp to epsilon.
    safe_wealth = np.maximum(wealth, 1e-6)
    
    if abs(gamma - 1.0) < 1e-6:
        return float(np.mean(np.log(safe_wealth)))
    else:
        return float(np.mean((safe_wealth ** (1 - gamma)) / (1 - gamma)))

def optimize_crra_allocation(terminal_returns: np.ndarray, gamma: float) -> Tuple[float, float]:
    """
    Finds the allocation fraction 'f' (0.0 to 1.0) that maximizes Expected Utility.
    """
    def objective(f):
        # Wealth = 1 + f * r
        # We want to MAXIMIZE utility, so we MINIMIZE negative utility
        port_returns = f * terminal_returns
        wealth = 1.0 + port_returns
        return -get_crra_utility(wealth, gamma)

    # Bounded optimization: f must be between 0% and 100%
    result = minimize_scalar(objective, bounds=(0.0, 1.0), method='bounded')
    
    optimal_f = result.x
    max_utility = -result.fun
    return optimal_f, max_utility

def _ai_confidence(ai_prob: Optional[float]) -> float:
    if ai_prob is None:
        return 0.0
    return float(min(max(abs(ai_prob - 0.5) * 2.0, 0.0), 1.0))


def _fuse_probabilities(
    mc_prob: float,
    ai_prob: Optional[float],
    regime: Optional[str],
    trend_score: float,
) -> Tuple[float, str, float]:
    """
    Probability sourcing logic.

    In the calibrated regime, we trust the isotonic AI probability when it is
    available. Otherwise we fall back to the Monte Carlo probability. The goal
    is to keep the probability in a single "space" before feeding it to the
    Kelly sizing logic rather than averaging incomparable signals.
    """
    if ai_prob is None:
        return mc_prob, "Source: Monte Carlo (AI unavailable)", 0.0

    confidence = _ai_confidence(ai_prob)
    method = (
        f"Source: Calibrated AI ({confidence * 100:.0f}% confidence)"
        if confidence > 0
        else "Source: Calibrated AI"
    )
    return ai_prob, method, confidence

def decision_engine(
    er_blended: float,
    expected_shortfall: float,
    prob_gain: float,  # Monte Carlo probability (used if AI absent)
    terminal_returns: Optional[np.ndarray] = None, 
    sim_kurtosis: float = 0.0,
    risk_aversion_lambda: float = RISK_AVERSION_LAMBDA,
    no_trade_threshold: float = NO_TRADE_UTILITY_THRESHOLD,
    regime: Optional[str] = None,
    regime_confidence: Optional[float] = None,
    transaction_cost: float = 0.001,
    win_loss_ratio: float = 1.0,
    kelly_scale: float = 0.5,
    trend_score: float = 1.0, 
    ai_raw_prob: Optional[float] = None,
    kelly_cap: float = 0.20,
) -> Tuple[str, float, float, float, List[str]]:
    """
    Institutional-Grade Decision Engine (Kelly-calibrated)

    Rather than applying layered heuristics, we trust the calibrated
    probability estimate and the Monte Carlo payoff distribution to determine
    position sizing through a half-Kelly framework. This keeps behaviour
    continuous and mathematically grounded.
    """
    reasons = []
    
    # --- Probability Sourcing ------------------------------------------------
    fused_prob, fuse_method, ai_confidence = _fuse_probabilities(
        prob_gain, ai_raw_prob, regime, trend_score
    )
    reasons.append(f"Probability Source: {fuse_method}")
    reasons.append(f"  • MC (History): {prob_gain:.1%}")
    ai_line = f"{ai_raw_prob:.1%}" if ai_raw_prob is not None else "N/A"
    reasons.append(f"  • AI (Signal): {ai_line}")
    reasons.append(f"  • Final Model Probability: {fused_prob:.1%}")

    if terminal_returns is None or terminal_returns.size == 0:
        return "Neutral", 0.0, 0.0, fused_prob, reasons + ["No Monte Carlo paths provided"]

    # --- Payoff Statistics ---------------------------------------------------
    net_paths = terminal_returns - transaction_cost
    wins = net_paths[net_paths > 0]
    losses = -net_paths[net_paths <= 0]

    avg_win = float(wins.mean()) if wins.size else max(float(net_paths.mean()), 1e-4)
    avg_loss = float(losses.mean()) if losses.size else max(avg_win, 1e-4)
    payoff_ratio = avg_win / max(avg_loss, 1e-6)

    reasons.append(f"Estimated Payoff Ratio (avg win / avg loss): {payoff_ratio:.2f}")

    # --- Kelly Sizing --------------------------------------------------------
    raw_kelly = fused_prob - ((1.0 - fused_prob) / payoff_ratio)
    reasons.append(f"Raw Kelly Fraction: {raw_kelly * 100:.1f}%")

    # Risk aversion parameter maps to how aggressively we apply Kelly.
    kelly_scale = max(0.05, min(1.0, kelly_scale))
    scaled_kelly = raw_kelly * kelly_scale
    reasons.append(
        f"Scaled Kelly ({kelly_scale * 100:.0f}% of raw) -> {scaled_kelly * 100:.1f}% "
        f"(λ={risk_aversion_lambda:.2f})"
    )

    final_size = max(0.0, min(scaled_kelly, kelly_cap))
    reasons.append(f"Position Cap {kelly_cap * 100:.0f}% applied -> {final_size * 100:.1f}%")

    expected_edge = fused_prob * avg_win - (1.0 - fused_prob) * avg_loss
    reasons.append(f"Expected Edge (per $1 risked): {expected_edge:.4f}")

    if final_size <= 0.0:
        return "Avoid", 0.0, expected_edge, fused_prob, reasons

    return "Bullish", final_size, expected_edge, fused_prob, reasons

def compute_path_risk_metrics(
    mc_paths: np.ndarray,
    current_price: float,
    horizon_days: int,
    max_dd_threshold: float,
) -> dict:
    if mc_paths is None or mc_paths.size == 0:
        return {"path_es": 0.0, "mdd_prob": 0.0, "tuw_fraction": 0.0, "tuw_days": 0.0}

    paths = mc_paths
    peaks = np.maximum.accumulate(paths, axis=0)
    drawdowns = paths / peaks - 1.0
    max_dd = drawdowns.min(axis=0)

    var_level = np.percentile(max_dd, 5)
    tail = max_dd[max_dd <= var_level]
    path_es = abs(float(tail.mean()) if tail.size > 0 else float(var_level))

    mdd_prob = float(np.mean(max_dd <= max_dd_threshold))

    below_start = paths < current_price
    tuw_fraction = float(below_start.sum(axis=0).mean() / (horizon_days + 1))
    tuw_days = tuw_fraction * (horizon_days + 1)

    return {"path_es": path_es, "mdd_prob": mdd_prob, "tuw_fraction": tuw_fraction, "tuw_days": tuw_days}

def evaluate_decision_rule(
    er_blended: float,
    expected_shortfall: float,
    probability_of_gain: Optional[float],
    risk_aversion_lambda: float,
    no_trade_threshold: float,
    regime_result: RegimeResult,
    top_factors: List[str],
    transaction_cost: float = 0.001,
    win_loss_ratio: float = 1.0,
    kelly_scale: float = 0.5,
    sim_kurtosis: float = 0.0,
    terminal_returns: Optional[np.ndarray] = None # Added for compatibility
) -> DecisionResult:
    if probability_of_gain is None:
        return DecisionResult("INSUFFICIENT DATA", ["Signal model unavailable; decision deferred."])

    label, size, expected_edge, fused_prob, reasons = decision_engine(
        er_blended=er_blended,
        expected_shortfall=expected_shortfall,
        prob_gain=probability_of_gain,
        terminal_returns=terminal_returns, # Pass through
        sim_kurtosis=sim_kurtosis,
        risk_aversion_lambda=risk_aversion_lambda,
        no_trade_threshold=no_trade_threshold,
        regime=regime_result.regime,
        regime_confidence=(regime_result.confidence / 100.0),
        transaction_cost=transaction_cost,
        win_loss_ratio=win_loss_ratio,
        kelly_scale=kelly_scale,
        # IMPORTANT: We cannot pass trend_score here because app.py calls this.
        # But wait, app.py calls decision_engine directly in some versions?
        # Assuming app.py calls decision_engine, we ensure decision_engine has the logic.
    )
    reasons.append(f"Fused probability of gain: {fused_prob:.1%}")
    reasons.append(f"Recommended allocation: {size*100:.1f}%")
    if top_factors:
        reasons.append("Top drivers: " + ", ".join(top_factors))
    return DecisionResult(label, reasons)

def get_dynamic_trading_params(asset_class: str) -> dict:
    if asset_class == "MEGA_INDEX":
        return {"sl_mult": 4.5, "regime_ma": 260, "prob_thresh": 0.52, "min_exposure": 0.40, "max_weight": 1.0, "index_confirm": False}
    elif asset_class == "EM_INDEX":
        return {"sl_mult": 3.0, "regime_ma": 200, "prob_thresh": 0.58, "min_exposure": 0.00, "max_weight": 1.0, "index_confirm": False}
    elif asset_class == "LIQUID_STOCK":
        return {"sl_mult": 2.0, "regime_ma": 130, "prob_thresh": 0.62, "min_exposure": 0.00, "max_weight": 0.30, "index_confirm": False}
    elif asset_class == "THIN_STOCK":
        return {"sl_mult": 4.0, "regime_ma": 240, "prob_thresh": 0.68, "min_exposure": 0.00, "max_weight": 0.10, "index_confirm": True}
    else:
        return get_dynamic_trading_params("LIQUID_STOCK")

__all__ = [
    "run_monte_carlo_paths", "compute_var_cvar", "build_risk_table",
    "identify_drawdown_windows", "apply_stress_scenario", "build_terminal_histogram",
    "estimate_drift_er", "estimate_mc_er", "run_canonical_engine",
    "compute_horizon_er_es", "evaluate_failure_probability", "decision_engine",
    "compute_path_risk_metrics", "evaluate_decision_rule", "get_dynamic_params",
    "get_dynamic_trading_params", "james_stein_drift"
]
