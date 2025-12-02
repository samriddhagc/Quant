# SA Risk Lab (Streamlit Dashboard)

An educational, single-page **risk/quant dashboard** for NEPSE stocks,
built with **Streamlit, Plotly, and Pandas**. It supports multi-asset inputs,
Monte Carlo risk, factor signals, regime awareness, and a utility-based decision engine.

## Sidebar Controls (Quick Reference)

Data
- Upload NEPSE CSV (Date, Close): primary price data; uses `nepse_stock.csv` if none uploaded.
- Optional Macro Proxy CSV: optional macro series to feed factor model (Date, Value).

Trend & Windows
- Short Window (SMA): lookback for fast moving average.
- Long Window (SMA): lookback for slow moving average.

Monte Carlo & Horizon
- Monte Carlo Simulations: number of simulated paths (more = smoother but slower).
- Time Horizon (Days): forecast horizon for MC, risk, and ER/ES scaling.

Volatility & Regime
- Volatility Model: choose constant historical σ, EWMA, or GARCH (if arch installed).
- Regime Detection: Rolling Volatility (default) or Markov Switching (if enabled).
- Shock Distribution: Normal or Student-t (fat tails); Student-t widens tails.

Stress & Failure
- Failure Threshold (%): price drop level to measure failure probability.
- Custom Stress Shock (%): instantaneous price shock applied before MC.
- Custom Stress Vol Multiplier: scales volatility under stress.

Decision & Risk Appetite
- Risk Aversion (λ): how much you penalize tail risk in utility U = ER – λ×ES. Larger λ means you hate losses more and need higher ER to act; smaller λ makes the model more aggressive.
- ER Shrinkage α: how much you pull the drift estimate toward a prior (risk-free). α=0 uses pure sample drift; α=0.1 mixes 90% sample + 10% prior; higher α stabilizes noisy series but can understate trend.
- Annual Risk-Free Rate: R_f used as the prior mean for drift and in CAPM (baseline required return).
- Probability Threshold (%): minimum blended gain probability; if prob < threshold, the trade is unfavorable regardless of U.
- ES Confidence Level (%): tail confidence for ES/CVaR; higher = harsher tails (more conservative ES).

Asset Selection
- Primary asset (single-asset analysis): main ticker for single-asset view.
- Select assets for multi-asset analysis: used for PCA/portfolio and comparisons.

## Key Outputs
- Signal Direction: Bullish / Bearish / Neutral / Caution based on utility and probabilities.
- Position Recommendation: suggested allocation size (%) from the utility score.
- Utility U: U = ER_blend – λ×ES (annualized ER and ES, adjusted by regime thresholds). Positive U with enough probability ⇒ Bullish; negative U ⇒ Bearish if probability is low.
- Probability of Gain: blended MC/factor probability (if factor model available).
- Regime: Calm / Neutral / Stressed with confidence.
- Historical μ/σ (annual): drift/vol from returns (shrunk drift shown separately in rationale).
- Drift components: Recent shrunk drift (short window, shrinkage α toward R_f) and CAPM ER (Rf + β×(E[Rm]–Rf)). ER_blend combines these (default 60/40) for more stable expectation.

## Tabs (Summary)
1) Decision Summary: utility-based decision, rationale, probabilities, regime.
2) Single Asset Analysis: price with SMAs and Golden/Death Cross markers.
3) Signal Model: factor contributions, coefficients, probability history.
4) Monte Carlo & Risk: paths, VaR/CVaR, stress tests, failure probabilities.
5) PCA & Hidden Factors: variance explained and loadings for selected assets.
6) Portfolio Optimizer (Markowitz): equal-weight, min-var, tangency comparisons.
7) Beta & Hedge Ratio: OLS beta/alpha/R² and hedged vs unhedged curves.
8) Downside Risk & Stress: drawdowns, Sortino, downside vol, simple stress checks.

## How to Run

1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your NEPSE CSV is named `nepse_stock.csv` and placed in the same directory,
   or upload via the sidebar file uploader when the app is running.

4. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

5. A browser tab should open automatically. If not, visit the URL shown in the terminal
   (typically `http://localhost:8501`).

## Notes

- This dashboard is intended for **statistical & educational** purposes only.
- The Monte Carlo/utility logic is simplified; do not use for production trading decisions.
