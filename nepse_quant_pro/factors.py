from typing import Optional, List, Dict, Tuple
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight  # <--- NEW IMPORT
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score

from .features_robust import (
    frac_diff_ffd, find_min_d, get_market_efficiency_ratio,
    calculate_kyles_lambda, calculate_smart_money_flow,
)
from .types import FactorModelResult
from .config import (
    TRIPLE_BARRIER_PT_MULT,
    TRIPLE_BARRIER_SL_MULT,
    TRIPLE_BARRIER_HORIZON,
)
from .returns import calculate_atr
from .data_quality import align_to_trading_calendar

LEGACY_CUTOFF = pd.Timestamp("2018-01-01")
MAX_FEATURE_LOOKBACK = 252

SECTOR_BARRIER_RULES: Dict[str, Dict[str, object]] = {
    "Commercial Banks": {
        "pt": {"normal": 0.85, "trend": 1.0, "choppy": 0.75},
        "sl": 2.5,
    },
    "Life Insurance": {
        "pt": {"normal": 0.85, "trend": 1.05, "choppy": 0.70},
        "sl": 2.5,
    },
    "Development Banks": {
        "pt": {"normal": 1.15, "trend": 1.40, "choppy": 0.90},
        "sl": 2.5,
    },
    "Finance": {
        "pt": {"normal": 1.15, "trend": 1.40, "choppy": 0.90},
        "sl": 2.5,
    },
    "Hydropower": {
        "pt": {"normal": 1.60, "trend": 2.20, "choppy": 1.25},
        "sl": 3.0,
    },
    "Non-Life Insurance": {
        "pt": {"normal": 1.40, "trend": 1.80, "choppy": 1.20},
        "sl": 2.7,
    },
    "Microfinance": {
        "pt": {"normal": 2.10, "trend": 3.00, "choppy": 1.50},
        "sl": 3.2,
    },
    "Hotels & Tourism": {
        "pt": {"normal": 1.35, "trend": 1.60, "choppy": 1.00},
        "sl": 2.7,
    },
    "Manufacturing & Processing": {
        "pt": {"normal": 1.35, "trend": 1.60, "choppy": 1.00},
        "sl": 2.7,
    },
    "Investment": {
        "pt": {"normal": 1.35, "trend": 1.60, "choppy": 1.00},
        "sl": 2.7,
    },
    "Trading": {
        "pt": {"normal": 1.35, "trend": 1.60, "choppy": 1.00},
        "sl": 2.7,
    },
    "Others": {
        "pt": {"normal": 1.35, "trend": 1.60, "choppy": 1.00},
        "sl": 2.7,
    },
}

DEFAULT_BARRIER_RULE = {
    "pt": {
        "normal": TRIPLE_BARRIER_PT_MULT,
        "trend": max(TRIPLE_BARRIER_PT_MULT * 1.2, TRIPLE_BARRIER_PT_MULT + 0.1),
        "choppy": max(0.7, TRIPLE_BARRIER_PT_MULT * 0.8),
    },
    "sl": TRIPLE_BARRIER_SL_MULT,
}


def _infer_barrier_regime(rsi_series: pd.Series, vol_ratio_series: pd.Series) -> str:
    if rsi_series is None or rsi_series.empty or vol_ratio_series is None or vol_ratio_series.empty:
        return "normal"
    rsi_val = rsi_series.dropna()
    rsi_val = float(rsi_val.iloc[-1]) if not rsi_val.empty else None
    vol_val = vol_ratio_series.dropna()
    vol_val = float(vol_val.iloc[-1]) if not vol_val.empty else None
    if rsi_val is None or vol_val is None:
        return "normal"
    if rsi_val >= 55 and vol_val >= 1.05:
        return "trend"
    if rsi_val <= 45 and vol_val <= 0.95:
        return "choppy"
    return "normal"


def _resolve_sector_barrier(sector_name: Optional[str], regime: str) -> Tuple[float, float]:
    rules = SECTOR_BARRIER_RULES.get(sector_name, DEFAULT_BARRIER_RULE)
    pt_rules = rules.get("pt", {})
    pt = pt_rules.get(regime) or pt_rules.get("normal") or DEFAULT_BARRIER_RULE["pt"]["normal"]
    sl = rules.get("sl", DEFAULT_BARRIER_RULE["sl"])
    return float(pt), float(sl)


def _cross_sectional_normalize(df: pd.DataFrame, prefix: str = "XSec_") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(index=df.index)
    row_mean = df.mean(axis=1)
    row_std = df.std(axis=1).replace(0.0, np.nan)
    normalized = df.sub(row_mean, axis=0).div(row_std, axis=0)
    normalized = normalized.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    normalized.columns = [f"{prefix}{col}" for col in df.columns]
    return normalized


def _rolling_autocorr(series: pd.Series, window: int, lag: int = 1) -> pd.Series:
    if series is None or series.empty or window <= lag:
        return pd.Series(index=series.index if series is not None else None, dtype=float)
    values = series.values
    result = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        sample = values[i - window + 1 : i + 1]
        if np.isnan(sample).any():
            continue
        x = sample[:-lag]
        y = sample[lag:]
        if len(x) == 0 or x.std() == 0 or y.std() == 0:
            continue
        result[i] = np.corrcoef(x, y)[0, 1]
    return pd.Series(result, index=series.index)

# --- FEATURE ENGINEERING ---

def calculate_rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """Robust Z-Score with clipping to handle outliers."""
    roll_mean = series.rolling(window, min_periods=max(10, window//2)).mean()
    roll_std = series.rolling(window, min_periods=max(10, window//2)).std()
    z = (series - roll_mean) / roll_std.replace(0, 1)
    return z.clip(-4.0, 4.0).fillna(0.0)

def calculate_adx_structure(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Directional Movement Index (ADX) Structural Strength."""
    high, low, close = high.astype(float), low.astype(float), close.astype(float)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean().replace(0.0, np.nan)
    
    up, down = high.diff(), -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    plus_di = 100 * (pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1)
    return dx.rolling(period).mean().fillna(0.0)

def calculate_volatility_adjusted_momentum(prices: pd.Series, window: int) -> pd.Series:
    """Returns / Volatility (Sharpe Ratio proxy)."""
    r = prices.pct_change()
    mu = r.rolling(window).mean()
    sigma = r.rolling(window).std().replace(0, np.nan)
    return (mu / sigma * np.sqrt(252)).fillna(0.0)


def calculate_bb_squeeze(prices: pd.Series, window: int = 20) -> pd.Series:
    """Normalized Bollinger band width to capture volatility contraction."""
    if prices is None or prices.empty:
        return pd.Series(index=prices.index if prices is not None else None, dtype=float)
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    band_width = (rolling_std * 4).div(rolling_mean.replace(0.0, np.nan))
    band_width = band_width.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return calculate_rolling_zscore(band_width, window=126).rename("BB_Width_Z")


def calculate_mean_reversion_features(prices: pd.Series, window: int = 60) -> pd.DataFrame:
    df = pd.DataFrame(index=prices.index)
    if prices is None or prices.empty:
        return df
    ma = prices.rolling(window).mean()
    dist = (prices / ma - 1.0).fillna(0.0)
    df["Dist_MA_Z"] = calculate_rolling_zscore(dist, window=126)
    std = prices.rolling(window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    bb_pos = (prices - lower).div((upper - lower).replace(0.0, np.nan)).clip(0.0, 1.0)
    df["BB_Position"] = bb_pos.fillna(0.5)
    long_lookback = 400
    roll_min = prices.rolling(long_lookback).min()
    roll_max = prices.rolling(long_lookback).max()
    range_pos = (prices - roll_min).div((roll_max - roll_min).replace(0.0, np.nan)).clip(0.0, 1.0)
    df["Price_Range_Pos"] = range_pos.fillna(0.5)
    sma_200 = prices.rolling(200).mean()
    dist_200 = (prices / sma_200 - 1.0).fillna(0.0)
    df["Dist_SMA200_Z"] = calculate_rolling_zscore(dist_200, window=252)
    return df

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index with EMA smoothing."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0).rename("RSI")

def calculate_regime_features(prices: pd.Series, long_window: int) -> pd.DataFrame:
    """Trend strength and slope around the long moving average."""
    df = pd.DataFrame(index=prices.index)
    long_ma = prices.rolling(long_window).mean()
    df["Trend_Strength"] = (prices / long_ma - 1.0).fillna(0.0)
    slope = long_ma.diff().rolling(5).mean()
    df["Trend_Slope"] = slope / prices.replace(0, np.nan)
    return df.fillna(0.0)

def _compute_robust_frac_diff(prices: pd.Series) -> pd.Series:
    """Wrapper to safely compute FFD features."""
    try:
        d_min = find_min_d(prices, max_d=1.0, step=0.05)
        return frac_diff_ffd(prices, d=d_min, thres=1e-4).rename(f"FracDiff_d{d_min:.2f}")
    except:
        return np.log(prices).diff().rename("FracDiff_LogRet").fillna(0.0)

# --- LABELING (TRIPLE BARRIER) ---

def generate_triple_barrier_labels(
    prices: pd.Series,
    volatility: pd.Series,
    horizon: int = TRIPLE_BARRIER_HORIZON,
    pt_mult: float = TRIPLE_BARRIER_PT_MULT,
    sl_mult: float = TRIPLE_BARRIER_SL_MULT,
) -> pd.Series:
    """
    Triple Barrier Method.
    Label 1 (Win): Hits Profit Target (Upper) first.
    Label 0 (Loss): Hits Stop Loss (Lower) first OR Time Limit.
    """
    labels = pd.Series(index=prices.index, dtype=float)
    
    # Barriers
    upper_barriers = prices * (1 + volatility * pt_mult)
    lower_barriers = prices * (1 - volatility * sl_mult)
    
    p_arr = prices.values
    ub_arr = upper_barriers.values
    lb_arr = lower_barriers.values
    n = len(p_arr)
    
    out_labels = np.full(n, np.nan)
    
    for i in range(n - horizon - 1):
        barrier_u = ub_arr[i]
        barrier_l = lb_arr[i]
        
        window = p_arr[i+1 : i+horizon+1]
        
        hit_pt = (window >= barrier_u)
        hit_sl = (window <= barrier_l)
        
        pt_idx = np.argmax(hit_pt) if np.any(hit_pt) else -1
        sl_idx = np.argmax(hit_sl) if np.any(hit_sl) else -1
        
        outcome = np.nan
        if pt_idx != -1 and (sl_idx == -1 or pt_idx < sl_idx):
            outcome = 1.0
        elif sl_idx != -1:
            outcome = 0.0
        out_labels[i] = outcome

    labels[:] = out_labels
    labels.iloc[-horizon:] = np.nan
    
    return labels.rename("Target")

# --- INSTITUTIONAL HELPERS ---

class FeatureNeutralizer(BaseEstimator, TransformerMixin):
    """
    Removes the dominant market factor (Beta) from features.
    """
    def __init__(self, proportion=0.65):
        self.proportion = proportion
        self.pca = PCA(n_components=1)

    def fit(self, X, y=None):
        try:
            if not np.any(np.isnan(X)):
                self.pca.fit(X)
        except:
            pass 
        return self

    def transform(self, X):
        try:
            if hasattr(self.pca, "components_"):
                market_factor = self.pca.transform(X) # Shape (n, 1)
                market_exposure = self.pca.inverse_transform(market_factor) 
                
                # Dynamic Neutralization: Remove 65% of Beta
                X_neutral = X - (self.proportion * market_exposure)
                return X_neutral
            else:
                return X
        except:
            return X

# --- FACTOR CONSTRUCTION ---

def _drop_constant_columns(df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    std = df.std(axis=0, skipna=True)
    keep = std[std > eps].index
    if len(keep) == 0:
        return df
    return df[keep]


FX_KEYWORDS = ("fx", "usd", "forex", "currency", "exchange")
REMIT_KEYWORDS = ("remit", "inflow", "foreign", "remittance")
RAIN_KEYWORDS = ("rain", "rainfall", "precip", "hydro", "monsoon")


def _build_macro_features(
    macro_series: Optional[pd.DataFrame],
    index: pd.DatetimeIndex,
    asset_returns: Optional[pd.Series] = None,
    sector_beta_series: Optional[pd.Series] = None,
    sector_name: Optional[str] = None,
) -> pd.DataFrame:
    if macro_series is None or macro_series.empty:
        return pd.DataFrame(index=index)
    macro_df_input = (
        pd.DataFrame(macro_series)
        if isinstance(macro_series, pd.DataFrame)
        else pd.DataFrame({"Macro_Input": macro_series})
    )
    macro_df_input = macro_df_input.reindex(index).ffill()
    combined = pd.DataFrame(index=index)

    def _macro_block(series: pd.Series, label: str) -> pd.DataFrame:
        block = pd.DataFrame(index=index)
        block[f"{label}_Z"] = calculate_rolling_zscore(series, window=252)
        block[f"{label}_Return_20d"] = series.pct_change(20).fillna(0.0)
        block[f"{label}_Momentum"] = calculate_volatility_adjusted_momentum(series, 60)
        block[f"{label}_Return_5d"] = series.pct_change(5).fillna(0.0)
        macro_ret = series.pct_change().fillna(0.0)
        beta_align = None
        if sector_beta_series is not None and not sector_beta_series.empty:
            beta_align = sector_beta_series.reindex(index).fillna(0.0)

        is_inflation = any(key in label.lower() for key in ("inflation", "cpi"))
        if asset_returns is not None and not asset_returns.empty:
            asset_ret = asset_returns.reindex(index).fillna(0.0)
            lookback = 126
            cov = asset_ret.rolling(lookback).cov(macro_ret)
            var = macro_ret.rolling(lookback).var().replace(0.0, np.nan)
            beta = (cov / var).fillna(0.0)
            corr = asset_ret.rolling(lookback).corr(macro_ret).fillna(0.0)
            spread = (asset_ret - macro_ret).fillna(0.0)
            block[f"{label}_Beta_Z"] = calculate_rolling_zscore(beta, window=lookback)
            block[f"{label}_Corr_Z"] = calculate_rolling_zscore(corr, window=lookback)
            block[f"{label}_Spread_Z"] = calculate_rolling_zscore(spread, window=lookback)
            block[f"{label}_AssetRet_Interaction"] = asset_ret * macro_ret
            if is_inflation:
                infl_momo = calculate_rolling_zscore(
                    (asset_ret * macro_ret).rolling(lookback).mean(), window=lookback
                )
                block[f"{label}_InflationCarry_Z"] = infl_momo
        if beta_align is not None:
            block[f"{label}_SectorBeta_Interaction"] = beta_align * macro_ret
            if is_inflation:
                block[f"{label}_InflationSensitivity_Z"] = calculate_rolling_zscore(
                    beta_align * macro_ret, window=126
                )
        lower_label = label.lower()
        if any(key in lower_label for key in FX_KEYWORDS):
            fx_change = series.pct_change().fillna(0.0)
            block[f"{label}_FX_Shock"] = calculate_rolling_zscore(fx_change, window=60)
            block[f"{label}_FX_Carry"] = calculate_rolling_zscore(
                (fx_change.rolling(30).mean() - fx_change.rolling(5).mean()).fillna(0.0),
                window=126,
            )
        if any(key in lower_label for key in REMIT_KEYWORDS):
            rem_trend = series.rolling(90).mean().pct_change().fillna(0.0)
            block[f"{label}_Remit_Trend_Z"] = calculate_rolling_zscore(rem_trend, window=126)
            if asset_returns is not None:
                block[f"{label}_Asset_Link"] = calculate_rolling_zscore(
                    asset_returns.rolling(60).corr(rem_trend), window=126
                )
        if any(key in lower_label for key in RAIN_KEYWORDS):
            rain_dev = (series - series.rolling(252).mean()).fillna(0.0)
            block[f"{label}_Hydro_Stress_Z"] = calculate_rolling_zscore(rain_dev, window=252)
            if sector_name == "Hydropower" and asset_returns is not None and not asset_returns.empty:
                proxy_prices = np.exp(asset_returns.cumsum())
                asset_mom = calculate_volatility_adjusted_momentum(proxy_prices, 60).reindex(index)
                aligned_series = series.reindex(index).ffill().fillna(0.0)
                roll_min = aligned_series.rolling(252).min()
                roll_max = aligned_series.rolling(252).max()
                macro_norm = (aligned_series - roll_min) / (roll_max - roll_min + 1e-6)
                block[f"{label}_Hydro_Interaction"] = (asset_mom.fillna(0.0) * macro_norm.fillna(0.5))
        if sector_name in ("Commercial Banks", "Development Banks") and any(
            key in lower_label for key in ("remit", "liquidity", "rate")
        ) and asset_returns is not None and not asset_returns.empty:
            vol_20 = asset_returns.rolling(20).std()
            vol_120 = asset_returns.rolling(120).std()
            vol_ratio = (vol_20 / vol_120.replace(0.0, np.nan)).clip(0.5, 2.0).reindex(index)
            aligned_series = series.reindex(index).ffill()
            block[f"{label}_Bank_Interaction"] = calculate_rolling_zscore(aligned_series, 126) / vol_ratio
        return block.fillna(0.0)

    for column in macro_df_input.columns:
        series = macro_df_input[column].astype(float)
        label = column if column.startswith("Macro_") else f"Macro_{column}"
        block = _macro_block(series, label)
        combined = pd.concat([combined, block], axis=1)

    return combined.fillna(0.0)

def _build_sector_features(
    prices: pd.Series,
    sector_series: Optional[pd.Series],
    index: pd.DatetimeIndex,
    horizon: int,
) -> pd.DataFrame:
    if sector_series is None or sector_series.empty:
        return pd.DataFrame(index=index)
    sector_aligned = pd.Series(sector_series).astype(float).reindex(index).ffill()
    df = pd.DataFrame(index=index)
    rel_price = (prices / sector_aligned.replace(0.0, np.nan)) - 1.0
    df["Sector_RelStrength_Z"] = calculate_rolling_zscore(rel_price.fillna(0.0), window=126)
    sector_ret = sector_aligned.pct_change(max(horizon, 5))
    asset_ret = prices.pct_change(max(horizon, 5))
    spread = (asset_ret - sector_ret).fillna(0.0)
    df["Sector_Spread_Z"] = calculate_rolling_zscore(spread, window=126)

    sector_vol = sector_aligned.pct_change().rolling(20).std()
    asset_vol = prices.pct_change().rolling(20).std()
    vol_spread = (asset_vol - sector_vol).fillna(0.0)
    df["Sector_Vol_Spread_Z"] = calculate_rolling_zscore(vol_spread, window=126)

    def _rolling_percent_rank(series: pd.Series, window: int = 252) -> pd.Series:
        def _percentile(values: pd.Series) -> float:
            arr = values.to_numpy()
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                return np.nan
            last = arr[-1]
            rank = (arr <= last).sum()
            return rank / len(arr)

        return series.rolling(window, min_periods=20).apply(_percentile, raw=False)

    df["Sector_Spread_PctRank"] = _rolling_percent_rank(spread)
    sector_mom = calculate_volatility_adjusted_momentum(sector_aligned, max(5, horizon)).fillna(0.0)
    asset_mom = calculate_volatility_adjusted_momentum(prices, max(5, horizon)).fillna(0.0)
    df["Sector_Momentum_Spread_Z"] = calculate_rolling_zscore(
        (asset_mom - sector_mom).fillna(0.0), window=126
    )

    asset_ret_full = prices.pct_change().fillna(0.0)
    sector_ret_full = sector_aligned.pct_change().fillna(0.0)
    lookback = 126
    cov = asset_ret_full.rolling(lookback).cov(sector_ret_full)
    var = sector_ret_full.rolling(lookback).var().replace(0.0, np.nan)
    beta = (cov / var).fillna(0.0)
    corr = asset_ret_full.rolling(lookback).corr(sector_ret_full).fillna(0.0)
    df["Sector_Beta_Raw"] = beta
    df["Sector_Corr_Raw"] = corr
    df["Sector_Beta_Z"] = calculate_rolling_zscore(beta, window=lookback)
    df["Sector_Corr_Z"] = calculate_rolling_zscore(corr, window=lookback)
    df["Sector_Trend_Interaction"] = df["Sector_Beta_Z"] * df["Sector_RelStrength_Z"]
    df["Sector_Corr_Interaction"] = df["Sector_Corr_Z"] * df["Sector_RelStrength_Z"]

    asset_dd = (prices / prices.cummax() - 1.0).fillna(0.0)
    sector_dd = (sector_aligned / sector_aligned.cummax() - 1.0).fillna(0.0)
    dd_spread = asset_dd - sector_dd
    df["Sector_Drawdown_Z"] = calculate_rolling_zscore(dd_spread, window=126)

    return df.fillna(0.0)


def build_factor_dataframe(
    price_df: pd.DataFrame,
    log_returns: pd.Series,
    long_window: int,
    horizon: int = TRIPLE_BARRIER_HORIZON,
    macro_series: Optional[pd.Series] = None,
    sector_series: Optional[pd.Series] = None,
    sector_name: Optional[str] = None,
) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return pd.DataFrame()
    price_df = align_to_trading_calendar(price_df)
    if price_df is None or price_df.empty:
        return pd.DataFrame()

    prices = price_df["Close"].astype(float)
    if log_returns is not None:
        log_returns = log_returns.reindex(prices.index).ffill().bfill()
    high = price_df.get("High", prices).astype(float)
    low = price_df.get("Low", prices).astype(float)
    vol = price_df.get("Volume", pd.Series(1.0, index=prices.index)).fillna(0.0)
    if sector_series is not None:
        if not isinstance(sector_series, pd.Series):
            sector_series = pd.Series(sector_series, index=price_df.index)
        sector_series = sector_series.reindex(prices.index).ffill()

    # Factors
    frac_diff = _compute_robust_frac_diff(prices)
    mom_slow = calculate_rolling_zscore(calculate_volatility_adjusted_momentum(prices, max(5, horizon))).rename(f"Sharpe_{horizon}d_Z")
    adx = calculate_rolling_zscore(calculate_adx_structure(high, low, prices)).rename("ADX_Z")
    eff_ratio = calculate_rolling_zscore(get_market_efficiency_ratio(prices, 20)).rename("Efficiency_Z")
    kyles = calculate_rolling_zscore(calculate_kyles_lambda(prices, vol), 126).rename("Liquidity_Z")
    smf = calculate_rolling_zscore(calculate_smart_money_flow(prices, high, low, vol), 126).rename("SmartMoney_Z")
    
    daily_vol = log_returns.rolling(20).std()
    vol_regime = calculate_rolling_zscore(daily_vol, 252).rename("Vol_Regime_Z")
    skew = log_returns.rolling(60).skew().fillna(0).rename("Roll_Skew")
    kurt = log_returns.rolling(60).kurt().fillna(0).rename("Roll_Kurt")

    regime_features = calculate_regime_features(prices, long_window)
    rsi = calculate_rsi(prices)
    volume_z = calculate_rolling_zscore(vol.rolling(20).mean().ffill(), window=60).rename("Volume_Z")
    range_pct = ((high - low) / prices.replace(0, np.nan)).rolling(20).mean().fillna(0.0)
    range_z = calculate_rolling_zscore(range_pct, window=60).rename("Range_Compression_Z")
    bb_squeeze = calculate_bb_squeeze(prices)
    mom_fast = calculate_rolling_zscore(
        calculate_volatility_adjusted_momentum(prices, max(3, horizon // 2)),
        window=60,
    ).rename(f"Sharpe_{max(3, horizon // 2)}d_Z")

    regime_z = calculate_rolling_zscore(daily_vol.rolling(126).mean(), window=252).rename("Regime_Intensity_Z")
    regime_tag = np.sign(regime_z).rename("Regime_Tag")
    reversion = calculate_mean_reversion_features(prices, window=max(20, horizon))

    atr_series = calculate_atr(price_df, period=14)
    atr_pct = (atr_series / prices.replace(0, np.nan)).fillna(0.0)
    atr_z = calculate_rolling_zscore(atr_pct, window=126).rename("ATR_Z")
    vol_short = log_returns.rolling(20).std()
    vol_long = log_returns.rolling(120).std()
    vol_ratio = (vol_short / vol_long.replace(0.0, np.nan)).clip(0.0, 5.0).fillna(0.0)
    vol_ratio_z = calculate_rolling_zscore(vol_ratio, window=126).rename("Vol_Ratio_Z")
    autocorr_60 = _rolling_autocorr(log_returns, window=60, lag=1).fillna(0.0).rename("Autocorr_60")
    autocorr_z = calculate_rolling_zscore(autocorr_60, window=252).rename("Autocorr_Z")
    rolling_max = prices.cummax()
    drawdown = (prices / rolling_max - 1.0).fillna(0.0).rename("Drawdown")
    drawdown_z = calculate_rolling_zscore(drawdown, window=252).rename("Drawdown_Z")

    factors = pd.concat(
        [
            frac_diff,
            mom_fast,
            mom_slow,
            adx,
            eff_ratio,
            kyles,
            smf,
            vol_regime,
            skew,
            kurt,
            regime_features,
            rsi,
            volume_z,
            range_z,
            bb_squeeze,
            reversion,
            regime_z,
            regime_tag,
            atr_z,
            vol_ratio_z,
            autocorr_z,
            drawdown_z,
        ],
        axis=1,
    )
    xsec_features = _cross_sectional_normalize(factors)
    if xsec_features is not None and not xsec_features.empty:
        factors = pd.concat([factors, xsec_features], axis=1)
    
    barrier_regime = _infer_barrier_regime(rsi, vol_ratio)
    pt_mult_dynamic, sl_mult_dynamic = _resolve_sector_barrier(sector_name, barrier_regime)

    dynamic_vol = daily_vol.fillna(0.02) * np.sqrt(max(horizon, 1))
    fwd_ret = prices.pct_change(horizon).shift(-horizon).rename("Fwd_Ret")
    rel_series = None
    sector_rank = None
    relative_target = None
    best_in_class = None
    if sector_series is not None:
        sector_prices = pd.Series(sector_series).astype(float).reindex(prices.index).ffill()
        sector_fwd = sector_prices.pct_change(horizon).shift(-horizon)
        if sector_fwd is not None:
            rel_series = (fwd_ret - sector_fwd).rename("Fwd_Relative")
        if rel_series is not None:
            sector_rank = rel_series.rank(pct=True).rename("SectorRank")
            relative_target = pd.Series(
                np.where(
                    sector_rank >= 0.8,
                    1.0,
                    np.where(sector_rank <= 0.2, 0.0, np.nan),
                ),
                index=sector_rank.index,
            ).rename("Target")
            best_in_class = relative_target.copy()
            best_in_class[:] = 0.0
            if not sector_rank.empty:
                threshold = sector_rank.quantile(0.8)
                best_in_class.loc[sector_rank >= threshold] = 1.0
    if relative_target is None:
        targets = generate_triple_barrier_labels(
            prices,
            dynamic_vol,
            horizon=horizon,
            pt_mult=pt_mult_dynamic,
            sl_mult=sl_mult_dynamic,
        )
    else:
        targets = relative_target

    zero_vol_mask = (vol <= 0)
    targets[zero_vol_mask] = np.nan

    sector_features = _build_sector_features(prices, sector_series, prices.index, horizon)
    sector_beta_series = (
        sector_features["Sector_Beta_Raw"] if "Sector_Beta_Raw" in sector_features else None
    )
    macro_features = _build_macro_features(
        macro_series.reindex(price_df.index).ffill() if macro_series is not None else None,
        prices.index,
        log_returns,
        sector_beta_series=sector_beta_series,
        sector_name=sector_name,
    )
    rel_cols = [c for c in [fwd_ret, rel_series, sector_rank] if isinstance(c, pd.Series)]
    payload_cols = [factors, sector_features, macro_features, targets, *rel_cols]
    data = pd.concat(payload_cols, axis=1, join="inner")

    leakage_cols = {"Target", "Fwd_Ret", "Fwd_Relative", "SectorRank"}
    feat_cols = [c for c in data.columns if c not in leakage_cols]
    features_only = data[feat_cols].fillna(0.0)
    features_only = _drop_constant_columns(features_only)
    cleaned = pd.concat([features_only, data[["Target", "Fwd_Ret"]]], axis=1)

    # Drop last horizon rows to prevent leakage
    drop_rows = horizon
    if drop_rows > 0 and len(cleaned) > drop_rows:
        cleaned = cleaned.iloc[:-drop_rows]
    return cleaned

def create_lagged_features(features: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    base = features.copy()
    cols = [c for c in base.columns if c not in ["Target", "Fwd_Ret"]]
    for lag in lags:
        shifted = base[cols].shift(lag).add_suffix(f"_LAG_{lag}")
        base = pd.concat([base, shifted], axis=1)
    return base 

def _train_secondary_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    X_latest: pd.DataFrame,
    name: str,
    min_samples: int = 80,
) -> Optional[dict]:
    if (
        X is None
        or X.empty
        or X_latest is None
        or X_latest.empty
        or len(X) < min_samples
        or y.nunique() < 2
    ):
        return None

    split_idx = max(int(len(X) * 0.8), len(X) - 40)
    if split_idx <= 20 or split_idx >= len(X):
        return None

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    try:
        model.fit(X_train, y_train)
    except Exception:
        return None

    try:
        latest_prob = float(model.predict_proba(X_latest)[0, 1])
    except Exception:
        latest_prob = None

    confidence = None
    if len(X_test) > 5 and y_test.nunique() > 1:
        try:
            preds = model.predict(X_test)
            confidence = accuracy_score(y_test, preds)
        except Exception:
            confidence = None

    if latest_prob is None:
        return None

    return {"name": name, "prob": latest_prob, "conf": confidence}


def _get_era_weights(index: pd.Index, legacy_cutoff: pd.Timestamp = LEGACY_CUTOFF) -> np.ndarray:
    if index is None or len(index) == 0:
        return np.ones(0)
    if not isinstance(index, pd.DatetimeIndex):
        return np.ones(len(index))
    mask = index < legacy_cutoff if legacy_cutoff is not None else np.zeros(len(index), dtype=bool)
    weights = np.ones(len(index))
    weights[mask] = 0.6
    return weights


def _calibrate_probabilities(oos_probs: pd.Series, y: pd.Series) -> Tuple[Optional[LogisticRegression], Optional[str]]:
    """Applies Platt-style calibration when we have enough out-of-sample data."""
    if oos_probs is None or oos_probs.empty:
        return None, "Calibration skipped (no OOS records)."
    aligned_y = y.loc[oos_probs.index]
    if aligned_y.nunique() < 2 or len(aligned_y) < 200:
        return None, "Calibration skipped (insufficient label diversity)."
    try:
        lr = LogisticRegression(max_iter=200, C=2.0, solver="lbfgs")
        lr.fit(oos_probs.values.reshape(-1, 1), aligned_y.values)
        return lr, "Applied logistic calibration on OOS folds."
    except Exception as exc:
        return None, f"Calibration failed: {exc}"


def _calibrate_value(cal_model: Optional[LogisticRegression], prob: float) -> float:
    if cal_model is None:
        return float(np.clip(prob, 0.0, 1.0))
    clipped = np.clip(prob, 1e-4, 1 - 1e-4)
    calibrated = cal_model.predict_proba(np.array([[clipped]]))[0, 1]
    return float(np.clip(calibrated, 0.0, 1.0))


def _compute_probability_scaler(
    avg_score: Optional[float],
    std_score: Optional[float],
    prob_edge: Optional[float],
) -> float:
    """Strength of conviction in the ensemble probability (0 -> neutral, 1 -> full)."""
    score_component = 0.0
    if avg_score is not None:
        score_component = max(0.0, min(1.0, (avg_score - 0.5) / 0.06))
    edge_component = 0.0
    if prob_edge is not None:
        edge_component = max(0.0, min(1.0, prob_edge / 0.03))
    strength = 0.55 * score_component + 0.45 * edge_component

    if (
        avg_score is not None
        and std_score is not None
        and avg_score > 0.58
        and std_score < 0.10
    ):
        strength = max(strength, 0.95)
    elif avg_score is not None and avg_score > 0.62:
        strength = max(strength, 0.75)
        if avg_score > 0.68:
            strength = max(strength, 0.90)
    if std_score is not None and std_score > 0:
        strength *= max(0.2, 1.0 - min(0.7, std_score * 3.0))
    return float(np.clip(strength, 0.0, 1.0))


def _apply_prob_cap(prob: float, cap: float = 0.02) -> float:
    deviation = prob - 0.5
    limited = np.sign(deviation) * min(abs(deviation), max(cap, 0.0))
    return float(0.5 + limited)


def _run_fallback_model(
    factors: Optional[pd.DataFrame],
    reason: str,
    horizon: int
) -> FactorModelResult:
    latest_factors = None
    if factors is not None and not factors.empty:
        cols = [c for c in factors.columns if c not in ["Target", "Fwd_Ret"]]
        if cols:
            latest_factors = factors.iloc[[-1]][cols].fillna(0.0).iloc[-1]
    warnings = [
        reason,
        "Dataset rejected for modeling; probability neutralized at 50%.",
        "Please collect additional history or relax horizon to proceed.",
    ]
    return FactorModelResult(
        probability=0.5,
        coefficients=None,
        latest_factors=latest_factors,
        contributions=None,
        history=None,
        message="Neutral probability (insufficient validated data).",
        accuracy_cv=None,
        accuracy_std=None,
        confusion=None,
        feature_importance_df=None,
        warnings=warnings,
        component_probs=None,
        component_confidence=None,
    )

# --- MODELING ---

def fit_signal_model(
    factors: pd.DataFrame,
    horizon: int = 30,
    cv_folds: int = 5,
    sector_name: Optional[str] = None,
    global_calibrator: Optional[LogisticRegression] = None,
) -> FactorModelResult:
    min_obs = max(150, horizon * 2)
    if factors is None or len(factors) < min_obs:
        return _run_fallback_model(factors, f"Insufficient Data (Need > {min_obs} observations)", horizon)
    
    # Create Lags
    df_lagged = create_lagged_features(factors, lags=[1, 5, 21])
    lag_feature_cols = [c for c in df_lagged.columns if c not in ["Target", "Fwd_Ret"]]
    cleaned_lag_features = pd.DataFrame(index=df_lagged.index)
    if lag_feature_cols:
        cleaned_lag_features = _drop_constant_columns(df_lagged[lag_feature_cols])
    df_lagged = pd.concat([cleaned_lag_features, df_lagged[["Target", "Fwd_Ret"]]], axis=1)
    if cleaned_lag_features.empty:
        return FactorModelResult(
            probability=0.5,
            coefficients=None,
            latest_factors=None,
            contributions=None,
            history=None,
            message="Insufficient feature variation after preprocessing.",
            accuracy_cv=None,
            accuracy_std=None,
            confusion=None,
            feature_importance_df=None,
            warnings=["All features were constant after preprocessing."],
        )
    
    # --- CRITICAL SPLIT ---
    train_df = df_lagged.dropna(subset=["Target"])
    latest_df = df_lagged.iloc[[-1]] 
    
    if train_df.empty or len(train_df) < min_obs:
         return _run_fallback_model(factors, "No Valid Training Data (fallback engaged)", horizon)

    X = train_df.drop(columns=["Target", "Fwd_Ret"])
    y = train_df["Target"]
    
    X_latest = latest_df.drop(columns=["Target", "Fwd_Ret"])

    if y.nunique() < 2:
        return FactorModelResult(0.0, None, X.iloc[-1], None, None, "No Winning Trades in History")

    def _infer_monotonic_direction(col: str) -> int:
        lower = col.lower()
        if any(keyword in lower for keyword in ("sharpe", "momentum", "trend", "strength")):
            return 1
        if any(keyword in lower for keyword in ("vol", "drawdown", "atr", "skew", "kurt")):
            return -1
        return 0

    monotonic_constraints = [_infer_monotonic_direction(c) for c in X.columns]
    preprocessing_steps = [
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", RobustScaler()),
        ("neutralizer", FeatureNeutralizer(proportion=0.55)),
    ]
    monolithic_model = HistGradientBoostingClassifier(
        loss='log_loss',
        learning_rate=0.03,
        max_iter=600,
        max_depth=3,
        min_samples_leaf=40,
        l2_regularization=1.0,
        max_leaf_nodes=31,
        random_state=42,
        monotonic_cst=monotonic_constraints,
    )
    pipeline = Pipeline(preprocessing_steps + [("model", monolithic_model)])

    # --- ROBUST PURGED CV ---
    n_samples = len(X)
    max_theoretical_folds = max(1, n_samples // 50)
    n_splits = max(3, min(cv_folds, max_theoretical_folds))

    emergency_cv = n_samples < 150
    if emergency_cv:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=3, shuffle=False)
        purge_gap = 0
    else:
        purge_gap = 5 if n_samples < 300 else 20
        cv = TimeSeriesSplit(n_splits=n_splits, gap=purge_gap)

    oos_probs = pd.Series(index=X.index, dtype=float)
    scores = []
    fold_stats: List[dict] = []
    baseline_logloss = math.log(2.0)
    min_train_required = 50
    min_test_required = 10
    class_floor = 0.10

    for fold_number, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        X_train_cv, y_train_cv = X.iloc[train_idx], y.iloc[train_idx]
        X_test_cv, y_test_cv = X.iloc[test_idx], y.iloc[test_idx]

        if not emergency_cv:
            embargo_target = max(int(MAX_FEATURE_LOOKBACK * 0.25), horizon)
            embargo = min(embargo_target, max(0, len(X_train_cv) - min_train_required))
            if embargo > 0 and len(X_train_cv) > embargo:
                X_train_cv = X_train_cv[:-embargo]
                y_train_cv = y_train_cv[:-embargo]

        fold_info = {
            "fold": fold_number,
            "train_start": str(X_train_cv.index[0]) if len(X_train_cv) else None,
            "train_end": str(X_train_cv.index[-1]) if len(X_train_cv) else None,
            "test_start": str(X_test_cv.index[0]) if len(X_test_cv) else None,
            "test_end": str(X_test_cv.index[-1]) if len(X_test_cv) else None,
            "train_size": int(len(X_train_cv)),
            "test_size": int(len(X_test_cv)),
        }

        if len(X_train_cv) < min_train_required or len(X_test_cv) < min_test_required:
            fold_info["status"] = "skipped"
            fold_info["skip_reason"] = "insufficient samples"
            fold_stats.append(fold_info)
            continue

        recency_curve = np.linspace(0.7, 1.3, len(y_train_cv))
        time_weights = recency_curve / np.sum(recency_curve)
        era_weights_cv = _get_era_weights(X_train_cv.index)

        train_pos_frac = float(y_train_cv.mean()) if len(y_train_cv) else np.nan
        test_pos_frac = float(y_test_cv.mean()) if len(y_test_cv) else np.nan
        fold_info["train_pos_frac"] = train_pos_frac
        fold_info["test_pos_frac"] = test_pos_frac
        if (
            np.isnan(train_pos_frac)
            or train_pos_frac < class_floor
            or train_pos_frac > 1 - class_floor
        ):
            fold_info["status"] = "skipped"
            fold_info["skip_reason"] = "train class imbalance"
            fold_stats.append(fold_info)
            continue
        if (
            np.isnan(test_pos_frac)
            or test_pos_frac < class_floor
            or test_pos_frac > 1 - class_floor
        ):
            fold_info["status"] = "skipped"
            fold_info["skip_reason"] = "test class imbalance"
            fold_stats.append(fold_info)
            continue

        classes = np.unique(y_train_cv)
        if len(classes) > 1:
            class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_cv)
            class_weight_dict = dict(zip(classes, class_weights))
            balance_weights = np.array([class_weight_dict[label] for label in y_train_cv])

            recency_boost = np.linspace(0.8, 1.2, len(y_train_cv))
            final_sample_weights = time_weights * balance_weights * recency_boost * era_weights_cv
        else:
            final_sample_weights = time_weights * era_weights_cv

        try:
            m = clone(pipeline)
            m.fit(X_train_cv, y_train_cv, model__sample_weight=final_sample_weights)

            probs = m.predict_proba(X_test_cv)[:, 1]
            oos_probs.iloc[test_idx] = probs

            binary_preds = (probs > 0.60).astype(int)
            if len(np.unique(y_test_cv)) > 1:
                acc_component = balanced_accuracy_score(y_test_cv, binary_preds)
                auc_component = roc_auc_score(y_test_cv, probs)
            else:
                acc_component = accuracy_score(y_test_cv, binary_preds)
                auc_component = None
            brier = np.mean((probs - y_test_cv.values) ** 2)
            try:
                logloss_val = log_loss(y_test_cv, np.clip(probs, 1e-4, 1 - 1e-4))
                loss_component = max(0.0, (baseline_logloss - logloss_val) / baseline_logloss)
            except Exception:
                logloss_val = None
                loss_component = 0.0
            if auc_component is not None:
                score = 0.6 * auc_component + 0.4 * loss_component
            else:
                score = 0.5 * acc_component + 0.5 * loss_component

            if not np.isnan(score):
                scores.append(score)
            fold_info.update({
                "accuracy": float(accuracy_score(y_test_cv, binary_preds)),
                "balanced_accuracy": float(acc_component),
                "brier": float(brier),
                "log_loss": float(logloss_val) if logloss_val is not None else None,
                "auc": float(auc_component) if auc_component is not None else None,
                "prob_edge": float(loss_component),
                "status": "ok",
            })
            fold_stats.append(fold_info)
        except Exception as exc:
            fold_info["status"] = "skipped"
            fold_info["skip_reason"] = f"fit error: {exc}"
            fold_stats.append(fold_info)
            continue
    valid_folds = sum(1 for f in fold_stats if f.get("status") == "ok")
    if valid_folds == 0:
        return _run_fallback_model(factors, "Insufficient valid folds after QC.", horizon)
    required_folds = 3 if n_splits >= 3 else 1
    if valid_folds < required_folds:
        return _run_fallback_model(factors, "Insufficient valid folds after QC.", horizon)

    # --- FINAL FIT & PREDICTION ---
    final_model = clone(pipeline)
    
    # Apply balancing to final fit as well
    classes_all = np.unique(y)
    era_weights_full = _get_era_weights(X.index)
    if len(classes_all) > 1:
        cw_all = compute_class_weight(class_weight='balanced', classes=classes_all, y=y)
        cw_dict_all = dict(zip(classes_all, cw_all))
        bw_all = np.array([cw_dict_all[label] for label in y])
        recency_all = np.linspace(0.8, 1.2, len(y))
        final_weights = (np.ones(len(y)) / max(1, horizon)) * bw_all * recency_all * era_weights_full
    else:
        final_weights = (np.ones(len(y)) / max(1, horizon)) * era_weights_full
    
    final_model.fit(X, y, model__sample_weight=final_weights)
    
    try:
        latest_prob = float(final_model.predict_proba(X_latest)[0, 1])
    except:
        latest_prob = 0.5

    try:
        X_clean = SimpleImputer(strategy="constant", fill_value=0.0).fit_transform(X)
        X_clean_df = pd.DataFrame(X_clean, columns=X.columns, index=X.index)
        corrs = X_clean_df.corrwith(y).abs().sort_values(ascending=False)
    except:
        corrs = pd.Series(0, index=X.columns)

    feat_imp = pd.DataFrame({"Feature": corrs.index, "Importance": corrs.values})
    
    if scores:
        avg_score = float(np.mean(scores))
        std_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    else:
        avg_score = 0.5
        std_score = 0.0
    
    def _avg_metric(key: str) -> Optional[float]:
        vals = [m[key] for m in fold_stats if key in m and m[key] is not None]
        return float(np.mean(vals)) if vals else None

    avg_accuracy_metric = _avg_metric("accuracy")
    avg_balanced_accuracy = _avg_metric("balanced_accuracy")
    avg_brier = _avg_metric("brier")
    avg_log_loss = _avg_metric("log_loss")
    avg_auc = _avg_metric("auc")
    avg_prob_edge = _avg_metric("prob_edge")

    component_probs = {}
    component_confidence = {}
    if latest_prob is not None:
        component_probs["Institutional GBDT"] = latest_prob
    if avg_score is not None:
        component_confidence["Institutional GBDT"] = avg_score

    cv_metrics = None
    prob_edge = None
    if avg_log_loss is not None:
        prob_edge = max(0.0, (baseline_logloss - avg_log_loss) / baseline_logloss)
    if avg_prob_edge is not None:
        prob_edge = avg_prob_edge

    metrics_dict = {
        "accuracy_thresh": avg_accuracy_metric,
        "balanced_accuracy": avg_balanced_accuracy,
        "brier": avg_brier,
        "log_loss": avg_log_loss,
        "auc": avg_auc,
        "prob_edge": prob_edge,
        "fold_scores": scores if scores else None,
        "fold_details": fold_stats if fold_stats else None,
    }
    if any(v is not None for v in metrics_dict.values()):
        cv_metrics = metrics_dict

    calibration_warning = None
    calibration_model = None
    oos_valid = oos_probs.dropna()
    if not oos_valid.empty:
        calibration_model, calibration_warning = _calibrate_probabilities(oos_valid, y)
    else:
        calibration_warning = "Calibration skipped (no OOS probabilities)."

    used_global_calibration = False
    safety_cap_applied = False
    if calibration_model is None and global_calibrator is not None:
        calibration_model = global_calibrator
        calibration_warning = "Local calibration failed; using global calibrator."
        used_global_calibration = True
    if calibration_model is None:
        calibration_warning = "Calibration unavailable; applied dampener."
        safety_cap_applied = True
    
    # Logistic ridge leg focusing on the most recent observations
    X_recent = X.tail(720)
    y_recent = y.loc[X_recent.index]
    ridge_pipeline = Pipeline(
        preprocessing_steps
        + [
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=0.35,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=800,
                ),
            )
        ]
    )
    logistic_leg = _train_secondary_model(
        ridge_pipeline,
        X_recent,
        y_recent,
        X_latest,
        "Ridge Logistic",
        min_samples=200,
    )
    if logistic_leg:
        component_probs[logistic_leg["name"]] = logistic_leg["prob"]
        if logistic_leg["conf"] is not None:
            component_confidence[logistic_leg["name"]] = logistic_leg["conf"]
    
    # Blend probabilities using confidence as weights
    blend_items = []
    for name, prob in component_probs.items():
        if prob is None:
            continue
        conf = component_confidence.get(name, None)
        weight = conf if conf is not None and conf > 0 else (0.06 if "Institutional" in name else 0.03)
        blend_items.append((prob, weight))
    if blend_items:
        total_weight = sum(w for _, w in blend_items)
        if total_weight > 0:
            blended_prob = sum(prob * weight for prob, weight in blend_items) / total_weight
        else:
            blended_prob = latest_prob
    else:
        blended_prob = latest_prob

    raw_prob = float(np.clip(blended_prob if blended_prob is not None else 0.5, 0.0, 1.0))
    if safety_cap_applied:
        blended_prob = 0.5 + (raw_prob - 0.5) * 0.5
    else:
        blended_prob = _calibrate_value(calibration_model, raw_prob)

    governance_flags: List[str] = []
    gating_warning = None
    prob_scaler = _compute_probability_scaler(avg_score, std_score, prob_edge)
    sector_relax_set = {"Hydropower", "Finance", "Development Banks", "Microfinance", "Investment"}
    if sector_name in sector_relax_set and prob_edge is not None and prob_edge >= 0.02:
        prob_scaler = max(prob_scaler, 0.45)
    if cv_metrics is None:
        cv_metrics = {}
    cv_metrics["prob_scaler"] = prob_scaler
    if prob_scaler < 1.0:
        blended_prob = 0.5 + (blended_prob - 0.5) * prob_scaler
        gating_warning = f"Probability shrink applied (scaler={prob_scaler:.2f})."
    cap_band = None
    if std_score is not None and std_score > 0.07:
        governance_flags.append(f"High CV volatility ({std_score:.2f}); capping probability band to ±2%.")
        cap_band = 0.02
    if prob_edge is None or prob_edge < 0.01:
        governance_flags.append("Probability edge < 1%; keeping signal close to neutral.")
        cap_band = 0.02 if cap_band is None else min(cap_band, 0.02)
    if cap_band is not None:
        blended_prob = _apply_prob_cap(blended_prob, cap_band)

    cv_msg = f"Purged Walk-Forward Score: {avg_score:.1%}"
    
    history_df = pd.DataFrame({
        "Probability": oos_probs, 
        "Realized Positive": y
    }).dropna()

    ensemble_warning = None
    if len(component_probs) > 1:
        ensemble_warning = f"Ensemble blend across {len(component_probs)} models."
    
    warnings_out = []
    if governance_flags:
        if cv_metrics is None:
            cv_metrics = {}
        cv_metrics["governance_flags"] = governance_flags
    if ensemble_warning:
        warnings_out.append(ensemble_warning)
    if calibration_warning:
        warnings_out.append(calibration_warning)
    if gating_warning:
        warnings_out.append(gating_warning)
    
    safe_cv = avg_score if avg_score is not None else 0.5
    safe_std = std_score if std_score is not None else 0.0
    safe_edge = prob_edge if prob_edge is not None else 0.0
    current_warnings = warnings_out if warnings_out else []

    model_tier, trust_score, governance_notes = classify_model_quality(
        safe_cv,
        safe_std,
        sector_name,
        safe_edge,
        current_warnings,
    )

    if governance_notes:
        if warnings_out is None:
            warnings_out = []
        warnings_out.extend(governance_notes)
    if trust_score == 0.0:
        blended_prob = 0.5
        if warnings_out is None:
            warnings_out = []
        warnings_out.append("Probability forced to 0.5 (governance quarantine).")

    return FactorModelResult(
        probability=blended_prob,
        coefficients=corrs,
        latest_factors=X_latest.iloc[-1],
        contributions=corrs,
        history=history_df,
        message=cv_msg,
        accuracy_cv=avg_score,
        accuracy_std=std_score,
        confusion=None,
        feature_importance_df=feat_imp,
        warnings=warnings_out or None,
        component_probs=component_probs or None,
        component_confidence=component_confidence or None,
        cv_metrics=cv_metrics,
        model_tier=model_tier,
        trust_score=trust_score,
    )

# --- VISUALIZATION ---

def build_factor_contribution_chart(contributions: pd.Series) -> go.Figure:
    if contributions is None or contributions.empty: return go.Figure()
    active = contributions.sort_values(ascending=False).head(10)
    fig = go.Figure(go.Bar(x=active.index, y=active.values, marker_color="#00bcd4"))
    fig.update_layout(title="Dominant Factors", template="plotly_dark", margin=dict(l=10,r=10,t=40,b=40))
    return fig

def build_signal_history_chart(history: pd.DataFrame) -> go.Figure:
    if history is None or history.empty: return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history["Probability"], mode="lines", name="Signal", line=dict(color="#7c4dff")))
    fig.add_trace(go.Scatter(x=history.index, y=history["Probability"].rolling(50).mean(), mode="lines", name="Trend", line=dict(color="#ffab00", dash="dash")))
    wins = history[(history["Probability"] > 0.6) & (history["Realized Positive"]==1)]
    fig.add_trace(go.Scatter(x=wins.index, y=wins["Probability"], mode="markers", name="Win", marker=dict(color="#00e676", size=6)))
    fig.update_layout(title="OOS Signal History", template="plotly_dark", yaxis_range=[0, 1.05], margin=dict(l=10,r=10,t=40,b=40))
    return fig

def calculate_performance_metrics(history: pd.DataFrame, log_returns: pd.Series, threshold: float = 0.55) -> Dict[str, float]:
    metrics = {"win_rate": 0.0, "win_loss_ratio": 0.0, "trade_count": 0, "kelly_fraction": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
    try:
        valid = history.dropna()
        trades = valid[valid["Probability"] > threshold]
        
        if not trades.empty:
            trade_rets = log_returns.reindex(trades.index).shift(-1)
            wins = trade_rets[trade_rets > 0]
            losses = trade_rets[trade_rets <= 0]
            
            avg_win = wins.mean() if not wins.empty else 0.02
            avg_loss = abs(losses.mean()) if not losses.empty else 0.02
            
            win_rate = len(wins) / len(trades)
            
            if avg_loss > 0:
                b = avg_win / avg_loss
            else:
                b = 1.0
                
            kelly = (win_rate * b - (1 - win_rate)) / b
            
            metrics.update({
                "trade_count": len(trades), 
                "win_rate": win_rate, 
                "win_loss_ratio": b, 
                "kelly_fraction": max(0.0, kelly),
                "avg_win": avg_win,
                "avg_loss": avg_loss
            })
    except: pass
    return metrics

__all__ = [
    "calculate_adx_structure",
    "calculate_volatility_adjusted_momentum",
    "calculate_bb_squeeze",
    "calculate_mean_reversion_features",
    "generate_triple_barrier_labels",
    "_compute_robust_frac_diff",
    "build_factor_dataframe",
    "create_lagged_features",
    "fit_signal_model",
    "build_factor_contribution_chart",
    "build_signal_history_chart",
    "calculate_performance_metrics",
]
def classify_model_quality(
    cv_score: float,
    cv_std: float,
    sector_name: Optional[str] = None,
    prob_edge: Optional[float] = None,
    warnings: Optional[List[str]] = None,
) -> Tuple[str, float, List[str]]:
    """
    NEPSE-tuned, *even less conservative* model quality classifier.

    Parameters
    ----------
    cv_score : float
        Purged walk-forward CV score (≈0.5 = random).
    cv_std : float
        Std. dev. of CV score across folds.
    sector_name : Optional[str]
        Sector label (for small nudges).
    prob_edge : Optional[float]
        Extra edge metric (e.g. from log-loss).
    warnings : Optional[List[str]]
        Pipeline warnings (used to detect calibration issues, etc.).

    Returns
    -------
    tier : {"Quarantine","Paper","Marginal","Silver","Gold"}
    trust_score : float in [0,1]
    notes : list[str]
    """
    notes: List[str] = []
    warnings = warnings or []

    # --- Basic sanity handling ----------------------------------------------
    if cv_score is None or (hasattr(math, "isnan") and math.isnan(cv_score)):
        return "Quarantine", 0.0, ["Missing CV score; quarantined."]

    if cv_std is None or (hasattr(math, "isnan") and math.isnan(cv_std)):
        cv_std = 0.20  # assume moderately noisy

    # Edge from CV and optional prob_edge
    cv_edge = max(0.0, (cv_score - 0.5))
    extra_edge = max(0.0, prob_edge or 0.0)
    edge = max(cv_edge, extra_edge)

    # --- Hard quarantine ONLY for truly bad / pathological regimes ----------
    # 1) Very low CV AND no meaningful edge
    if cv_score < 0.47 and edge < 0.01:
        notes.append("CV score below 0.47 with negligible edge; quarantined.")
        return "Quarantine", 0.0, notes

    # 2) Extremely unstable models
    if cv_std > 0.45 and cv_score < 0.50:
        notes.append("CV volatility above 0.45 with sub-random CV; quarantined.")
        return "Quarantine", 0.0, notes

    # Edge basically zero → still allowed, but only as tiny experimental tier
    if edge < 0.002:
        notes.append("Edge below 0.2 percentage points; effectively random.")
        return "Paper", 0.10, notes

    # --- Baseline tiering by CV score (very loose for NEPSE) ----------------
    tier = "Silver"
    trust = 0.5

    if 0.47 <= cv_score < 0.50:
        tier = "Paper"
        trust = 0.25
        notes.append("Near-random CV but not catastrophic; allow tiny sizing.")
    elif 0.50 <= cv_score < 0.515:
        tier = "Marginal"
        trust = 0.35
        notes.append("Weak but acceptable edge; starter tier.")
    elif 0.515 <= cv_score < 0.53:
        tier = "Silver"
        trust = 0.55
        notes.append("Moderate edge; default production tier.")
    elif 0.53 <= cv_score < 0.57:
        tier = "Gold"
        trust = 0.75
        notes.append("Strong edge across folds; high-confidence model.")
    else:
        tier = "Gold"
        trust = 0.85
        notes.append("Very strong CV score; candidate for heavier sizing.")

    # --- Volatility-aware adjustments ---------------------------------------
    if cv_std < 0.08:
        trust += 0.05
        notes.append("Very stable across folds; boosting trust.")
    elif cv_std > 0.25:
        trust -= 0.10
        notes.append("High CV volatility; trimming trust.")
    if cv_std > 0.32:
        trust -= 0.10
        notes.append("Very high CV volatility; further trimming trust.")

    # --- Edge-based trimming / boost ----------------------------------------
    if edge < 0.005:
        trust = min(trust, 0.40)
        notes.append("Edge is small; capping trust at 0.40.")
    elif edge > 0.03:
        trust += 0.05
        notes.append("Unusually large edge; boosting trust slightly.")

    # --- Sector-aware nudging (NEPSE-specific intuition) --------------------
    sec = (sector_name or "").lower()

    if "hydro" in sec:
        trust = min(trust, 0.65)
        notes.append("Hydropower: capping trust at 0.65 due to structural noise.")
    elif "bank" in sec:
        trust += 0.05
        notes.append("Banking sector: structurally more liquid; nudging trust up.")
    elif "micro" in sec or "microfinance" in sec:
        trust = min(trust, 0.60)
        notes.append("Microfinance: idiosyncratic risk; capping trust at 0.60.")
    elif "investment" in sec or "holding" in sec:
        trust = min(trust, 0.55)
        notes.append("Investment / holding companies: conservative cap.")

    # --- Calibration-related downgrades -------------------------------------
    if any("Calibr" in w for w in warnings) and tier == "Gold":
        tier = "Silver"
        trust = min(trust, 0.85)
        notes.append("Downgraded due to calibration issues.")

    # Final clamp
    trust = float(max(0.0, min(1.0, trust)))
    return tier, trust, notes
