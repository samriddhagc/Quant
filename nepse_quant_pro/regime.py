import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture 

from .config import PERIODS_PER_YEAR
from .types import RegimeResult

def detect_regime(
    log_returns: pd.Series,
    method: str = "Gaussian Mixture", 
    rolling_window: int = 60,
) -> RegimeResult:
    """
    Robust Regime Detection using Gaussian Mixture Models (GMM).
    Identifies 'Hidden States' of volatility (Calm vs Stressed) 
    without relying on arbitrary rolling quantiles.
    """
    if log_returns.empty:
        return RegimeResult("Unknown", 0.0, method, "Insufficient data.", None, None)

    # 1. Feature Engineering: Annualized Rolling Volatility
    rolling_vol = log_returns.rolling(rolling_window).std() * math.sqrt(PERIODS_PER_YEAR)
    rolling_vol = rolling_vol.dropna()
    
    if len(rolling_vol) < 100:
        # Fallback to simple logic if not enough data for GMM
        current_vol = float(rolling_vol.iloc[-1]) if not rolling_vol.empty else 0.0
        return RegimeResult("Uncertain", 0.0, "Fallback", "Not enough data for GMM.", rolling_vol, None)

    # 2. Fit Gaussian Mixture Model (2 Components: Calm vs Stressed)
    # We reshape data for sklearn: [[vol1], [vol2], ...]
    X = rolling_vol.values.reshape(-1, 1)
    
    try:
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
        gmm.fit(X)
        
        # 3. Identify which cluster is "Stressed"
        # The cluster with the higher Mean Volatility is the 'Stressed' one.
        means = gmm.means_.flatten()
        stressed_idx = np.argmax(means)
        calm_idx = np.argmin(means)
        
        # 4. Predict Current State
        current_vol = X[-1].reshape(1, -1)
        probs = gmm.predict_proba(current_vol)[0] # [prob_cluster_0, prob_cluster_1]
        
        prob_stressed = probs[stressed_idx]
        prob_calm = probs[calm_idx]
        
        # 5. Define Regime based on Probability
        if prob_stressed > 0.60:
            regime = "Stressed"
            confidence = prob_stressed
        elif prob_calm > 0.60:
            regime = "Calm"
            confidence = prob_calm
        else:
            regime = "Neutral" # Transition zone
            confidence = 1.0 - abs(prob_stressed - 0.5) * 2

        # Thresholds for visualization (using the learned means)
        lower = means[calm_idx]
        upper = means[stressed_idx]
        
        info = (
            f"GMM Detected: {regime} (Conf: {confidence*100:.1f}%). "
            f"Cluster Means: Calm={means[calm_idx]*100:.1f}%, Stressed={means[stressed_idx]*100:.1f}%"
        )
        
        return RegimeResult(regime, confidence * 100, "Gaussian Mixture", info, rolling_vol, None)

    except Exception as e:
        # Graceful fallback if GMM fails mathematically
        return RegimeResult("Error", 0.0, "GMM Failed", str(e), rolling_vol, None)

def detect_market_regime(price_df: pd.DataFrame) -> RegimeResult:
    """
    Wrapper for full DataFrames (Scanner compatibility).
    Calculates returns and runs regime detection.
    """
    if price_df is None or "Close" not in price_df.columns:
        return RegimeResult("Unknown", 0.0, "Invalid Data", "Missing Close column", None, None)
        
    prices = price_df["Close"].astype(float)
    # Simple log return calc
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    return detect_regime(log_returns)

def build_regime_chart(rolling_vol: pd.Series, lower: float, upper: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol,
            mode="lines",
            name="Rolling Annualized Vol",
            line=dict(color="#00bcd4", width=2),
        )
    )
    # Visualizing the GMM 'Centers' roughly as thresholds
    if lower and upper:
        fig.add_hline(y=lower, line=dict(color="#81c784", dash="dash"), annotation_text="Calm Mean")
        fig.add_hline(y=upper, line=dict(color="#e57373", dash="dash"), annotation_text="Stressed Mean")
    
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=40),
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
    )
    return fig

__all__ = ["detect_regime", "detect_market_regime", "build_regime_chart"]