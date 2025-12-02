import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from typing import Tuple

# ============================================================
# PART 1: FRACTAL MATH & STATIONARITY (FIXED WINDOW)
# ============================================================

def get_weights_ffd(d: float, thres: float) -> np.ndarray:
    """
    Calculates weights for Fixed-Width Fractional Differentiation.
    Stops when weights drop below threshold to ensure fixed window size.
    """
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])  # Reverse to align with dot product (oldest -> newest)

def frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-4) -> pd.Series:
    """
    Apply Fractional Differentiation (FFD) using a fixed-width window.
    This preserves trend memory better than integer differencing while
    maintaining stationarity.
    """
    # 1. Compute weights
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    
    # 2. Apply weights using a rolling window (Vectorized Loop)
    output = {}
    series_vals = series.values
    dates = series.index
    
    if len(series) > width:
        # Loop optimization: Apply dot product over the series
        for i in range(width, len(series)):
            # Window: from i-width to i (inclusive)
            window_vals = series_vals[i-width : i+1]
            if len(window_vals) == len(w):
                 output[dates[i]] = np.dot(w, window_vals)
    
    return pd.Series(output).sort_index()

def find_min_d(series: pd.Series, max_d: float = 1.0, step: float = 0.05) -> float:
    """
    Returns the minimum differentiation order 'd' that preserves 
    stationarity (ADF p-value < 0.05).
    
    Improvements over original:
    1. Uses autolag='AIC' instead of fixed maxlag=1 for better statistical rigor.
    2. Handles short series gracefully.
    3. Returns 1.0 (fully diffed) if no fractional value works.
    """
    # Clean data: Remove infinite values and NaNs
    clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
    
    # If series is too short for any meaningful analysis, return standard diff
    if len(clean_series) < 20: 
        return 1.0
    
    # Iterate through d values from step to max_d
    for d in np.arange(step, max_d + step, step):
        # 1. Apply Fractional Differentiation
        diffed = frac_diff_ffd(clean_series, d=d, thres=1e-4).dropna()
        
        # Skip if differentiation ate all the data (window too large)
        if diffed.empty or len(diffed) < 10:
            continue
            
        try:
            # 2. Run Augmented Dickey-Fuller Test
            # autolag='AIC' lets the test pick the optimal lag length automatically
            result = adfuller(diffed, regression="c", autolag='AIC')
            p_value = result[1]
            
            # 3. Check for Stationarity (p-value < 5%)
            if p_value < 0.05:
                return round(float(d), 2)
                
        except Exception:
            # If ADF crashes (usually due to singular matrix on constant data), skip
            continue
            
    # Fallback: If no fractional d works, assume full integer differentiation is needed
    return 1.0
  

# ============================================================
# PART 2: REGIME METRICS
# ============================================================

def get_shannon_entropy(series: pd.Series, window: int = 60, bins: int = 50) -> pd.Series:
    """
    Calculates Rolling Shannon Entropy (Information Density).
    High Entropy = Random Walk (Noise).
    Low Entropy = Deterministic Trend (Signal).
    """
    # Use returns for stationarity
    returns = series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    entropy_series = pd.Series(index=series.index, dtype=float)
    
    # Rolling calculation
    for i in range(window, len(series)):
        window_data = returns.iloc[i-window : i]
        hist, _ = np.histogram(window_data, bins=bins, density=True)
        probs = hist[hist > 0] # Remove zeros to avoid log(0)
        ent = -np.sum(probs * np.log(probs))
        entropy_series.iloc[i] = ent
        
    return entropy_series.fillna(method='bfill').fillna(0)

def get_market_efficiency_ratio(close_prices: pd.Series, window: int) -> pd.Series:
    """
    Kaufman's Efficiency Ratio (Fractal Efficiency).
    ER = Directional Move / Total Path Travelled.
    ER -> 1.0: Strong Trend.
    ER -> 0.0: Chop/Noise.
    """
    direction = close_prices.diff(window).abs()
    volatility = close_prices.diff().abs().rolling(window).sum()
    er = direction / volatility
    return er.replace([np.inf, -np.inf], 0.0).fillna(0.0)

# ============================================================
# PART 3: MICROSTRUCTURE & FLOW
# ============================================================

def calculate_kyles_lambda(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Proxy for Kyle's Lambda (Cost of Liquidity)."""
    ret_abs = close.pct_change().abs()
    log_vol = np.log(volume.replace(0, 1))
    kyle = (ret_abs / log_vol).rolling(window).mean()
    return kyle.fillna(0)

def calculate_bekker_parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """High-Efficiency Volatility Estimator using High/Low range."""
    hl_ratio = np.log(high / low.replace(0, 1))
    bp_var = (1 / (4 * np.log(2))) * (hl_ratio ** 2)
    return np.sqrt(bp_var.rolling(window).mean())

def calculate_smart_money_flow(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Twiggs Money Flow: Volume-weighted buying/selling pressure."""
    # True Range High/Low
    tr_h = pd.concat([high, close.shift(1)], axis=1).max(axis=1)
    tr_l = pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    
    # Buying/Selling Pressure
    range_len = (tr_h - tr_l).replace(0, 1) # Avoid div by zero
    adv = ((2 * close - tr_l - tr_h) / range_len) * volume
    
    # Exponential Smoothing
    smf = adv.ewm(span=window, min_periods=window).mean()
    vol_smooth = volume.ewm(span=window, min_periods=window).mean()
    
    return (smf / vol_smooth.replace(0, 1)).fillna(0)

__all__ = [
    "frac_diff_ffd", "find_min_d", 
    "get_shannon_entropy", "get_market_efficiency_ratio",
    "calculate_kyles_lambda", "calculate_bekker_parkinson_vol",
    "calculate_smart_money_flow"
]
