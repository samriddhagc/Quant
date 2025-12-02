import math
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from .config import (
    PERIODS_PER_YEAR,
    DRIFT_SHRINKAGE_ALPHA,
    RECENT_DRIFT_WINDOW,
    LONG_DRIFT_WINDOW,
    AUTOCORRELATION_WINDOW,
)


def compute_features(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    data = df.copy()
    data["SMA_short"] = data["Close"].rolling(window=short_window).mean()
    data["SMA_long"] = data["Close"].rolling(window=long_window).mean()
    data["Returns"] = data["Close"].pct_change()
    data["Volatility_20d"] = data["Returns"].rolling(window=20).std()
    data["Signal"] = np.where(
        (data["SMA_short"] > data["SMA_long"]) & (data["Close"] > data["SMA_long"]),
        1,
        0,
    )
    data["Safe_To_Enter"] = data["Volatility_20d"] < 0.03
    data["Golden_Cross"] = data["Signal"].diff() == 1
    return data


def compute_log_returns(prices: pd.Series) -> pd.Series:
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna()


def summarize_returns(log_returns: pd.Series) -> pd.DataFrame:
    if log_returns.empty:
        return pd.DataFrame()
    stats = {
        "Mean": log_returns.mean(),
        "Std Dev": log_returns.std(),
        "Min": log_returns.min(),
        "Max": log_returns.max(),
        "Skewness": skew(log_returns, bias=False),
        "Kurtosis": kurtosis(log_returns, fisher=True, bias=False),
    }
    return pd.DataFrame.from_dict(stats, orient="index", columns=["Value"])


def compute_autocorrelation(log_returns: pd.Series, window: int = AUTOCORRELATION_WINDOW) -> float:
    """Estimate first-order autocorrelation over the last `window` observations."""
    r = log_returns.dropna().tail(window)
    if r.shape[0] < 2:
        return 0.0
    return float(r.autocorr(lag=1))


def estimate_blended_drift(
    log_returns: pd.Series,
    window_days: int,
    alpha: float,
    risk_free_rate_prior: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """
    Estimate a shrunk annualized drift using a recent window and a prior.
    For log returns, the sample drift is the mean log return scaled by periods_per_year.
    Shrinkage: mu_shrunk = (1 - alpha) * mu_sample + alpha * mu_prior.
    """
    if log_returns.empty or window_days <= 0:
        return 0.0
    window = log_returns.dropna().tail(window_days)
    if window.empty:
        return 0.0
    mu_sample_annual = float(window.mean()) * periods_per_year
    mu_shrunk = (1 - alpha) * mu_sample_annual + alpha * risk_free_rate_prior
    return mu_shrunk


def estimate_mu_sigma(
    log_returns: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
    alpha: float = DRIFT_SHRINKAGE_ALPHA,
    risk_free_rate_prior: float = 0.0,
    recent_window: int = RECENT_DRIFT_WINDOW,
    long_window: int = LONG_DRIFT_WINDOW,
) -> Dict[str, float]:
    daily_mu = float(log_returns.mean())
    daily_sigma = float(log_returns.std())
    annual_mu = daily_mu * periods_per_year
    annual_sigma = daily_sigma * math.sqrt(periods_per_year)

    annual_mu_recent = estimate_blended_drift(
        log_returns,
        recent_window,
        alpha,
        risk_free_rate_prior,
        periods_per_year,
    )
    annual_mu_long = estimate_blended_drift(
        log_returns,
        long_window,
        alpha,
        risk_free_rate_prior,
        periods_per_year,
    )

    return {
        "daily_mu": daily_mu,
        "daily_sigma": daily_sigma,
        "annual_mu": annual_mu,
        "annual_sigma": annual_sigma,
        "annual_mu_recent": annual_mu_recent,
        "annual_mu_long": annual_mu_long,
    }


def compute_ewma_vol(log_returns: pd.Series, lam: float = 0.94) -> pd.Series:
    if log_returns.empty:
        return pd.Series(dtype=float)
    variances = []
    prev_var = log_returns.var()
    for r in log_returns:
        prev_var = lam * prev_var + (1 - lam) * (r ** 2)
        variances.append(prev_var)
    sigma = np.sqrt(variances)
    return pd.Series(sigma, index=log_returns.index)


def fit_garch_vol(log_returns: pd.Series) -> Tuple[Optional[pd.Series], Optional[pd.DataFrame], Optional[str]]:
    if log_returns.empty or len(log_returns) < 60:
        return None, None, "Not enough data for GARCH(1,1)."
    try:
        from arch import arch_model
    except ImportError:
        return None, None, "Package 'arch' not installed. Run `pip install arch`."
    scaled = log_returns * 100
    model = arch_model(scaled, vol="GARCH", p=1, q=1, mean="constant", dist="normal")
    try:
        res = model.fit(disp="off")
        cond_vol = (res.conditional_volatility / 100).rename("GARCH")
        params_df = res.params.rename(
            {"omega": "omega", "alpha[1]": "alpha", "beta[1]": "beta"}
        ).to_frame(name="Value")
        return cond_vol, params_df, None
    except Exception as exc:  # pragma: no cover
        return None, None, f"GARCH failed: {exc}"


def calculate_atr(price_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR) from OHLC data."""
    required_cols = {"High", "Low", "Close"}
    if price_df is None or price_df.empty or not required_cols.issubset(price_df.columns):
        return pd.Series(0.0, index=price_df.index if price_df is not None else None, name="ATR")

    df = price_df.copy()
    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - df["Close"].shift(1)).abs()
    low_prev_close = (df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    atr = df["TR"].ewm(span=period, adjust=False).mean()
    return atr.rename("ATR")


def calculate_mean_atr(atr_series: pd.Series) -> float:
    """Return average ATR for volatility scaling."""
    if atr_series is None or atr_series.empty:
        return 1.0
    return float(atr_series.mean())


def get_regime(price_series: pd.Series, ma_period: int = 200) -> str:
    """Classify regime using price vs MA and slope."""
    if price_series is None or len(price_series) < ma_period:
        return "neutral"
    window = price_series.iloc[-ma_period:]
    ma = window.mean()
    ma_slope = window.iloc[-1] - window.iloc[0]
    current_price = price_series.iloc[-1]
    if current_price > ma and ma_slope > 0:
        return "bull"
    if current_price < ma and ma_slope < 0:
        return "bear"
    return "neutral"


def get_daily_volatility_class(log_returns: pd.Series) -> float:
    """Calculates the mean annualized daily volatility (StdDev * sqrt(252))."""
    if log_returns.empty:
        return 0.05 # Default 5%
    return log_returns.std() * np.sqrt(252)


def check_data_liquidity(price_df: pd.DataFrame) -> dict:
    """
    Checks for liquidity proxies: missing OHLC and presence of Volume data.
    CRITICAL FALLBACK: If OHLC is missing, High/Low are set to Close.
    Includes Average Daily Volume calculation for liquidity scoring.
    """
    # Defensive check for columns
    has_ohlc = all(col in price_df.columns for col in ['Open', 'High', 'Low'])
    has_volume = 'Volume' in price_df.columns
    
    # CRITICAL FALLBACK: Use Close for High/Low if missing
    if not has_ohlc:
        price_df['High'] = price_df['Close'] 
        price_df['Low'] = price_df['Close'] 
    
    # Calculate zero-volume ratio (proxy for thinness)
    zero_volume_ratio = (price_df['Volume'] == 0).sum() / len(price_df) if has_volume else 0.0
    
    # [NEW] Calculate Average Volume (last 50 days)
    avg_vol = 0.0
    if has_volume:
        avg_vol = price_df['Volume'].tail(50).mean()

    return {
        "has_ohlc": has_ohlc,
        "has_volume": has_volume,
        "daily_range_mean": (price_df['High'] - price_df['Low']).mean() / price_df['Close'].mean(),
        "zero_volume_ratio": zero_volume_ratio,
        "avg_volume": avg_vol  # <--- Added this metric
    }


def get_asset_class(symbol: str, daily_vol_ann: float, liquidity_data: dict) -> str:
    """Classifies the asset based on symbol, volatility, and liquidity."""
    
    if symbol in ['SPX', 'SPY', 'VOO', 'spx.csv', 'SPX.csv']:
        return 'MEGA_INDEX'
    
    VOL_HIGH_THRESHOLD = 0.35 
    VOL_EM_THRESHOLD = 0.18   
    
    # CRITICAL NEPSE FIX: 
    # Holidays account for ~35-40% of the year. 
    # We allow up to 50% zero-volume days to accommodate holidays + low activity.
    # Stocks with >50% zero days are genuinely illiquid "Thin Stocks".
    LIQUIDITY_THIN_RATIO = 0.50 
    
    # CRITICAL: Thin Stock classification (NEPSE hydros)
    if liquidity_data["has_volume"] and liquidity_data["zero_volume_ratio"] > LIQUIDITY_THIN_RATIO:
        return 'THIN_STOCK'
    
    if daily_vol_ann >= VOL_HIGH_THRESHOLD:
        return 'LIQUID_STOCK' 
        
    if daily_vol_ann >= VOL_EM_THRESHOLD:
        return 'EM_INDEX'
    
    return 'LIQUID_STOCK' # Default


__all__ = [
    "compute_features",
    "compute_log_returns",
    "summarize_returns",
    "compute_autocorrelation",
    "estimate_mu_sigma",
    "estimate_blended_drift",
    "compute_ewma_vol",
    "fit_garch_vol",
    "calculate_atr",
    "calculate_mean_atr",
    "get_regime",
    "get_daily_volatility_class",
    "check_data_liquidity",
    "get_asset_class"
]