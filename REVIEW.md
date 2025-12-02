# NEPSE Quant Pro - Code Review & Recommendations

## Overview
The application is a sophisticated quantitative analysis dashboard built with Streamlit. It features advanced risk modeling (Monte Carlo with Jump Diffusion, Regime Detection, etc.) and a clean modular structure. However, to make it "realistically aligned" with NEPSE (Nepal Stock Exchange), several specific market nuances need to be addressed.

## Key Findings & Recommendations

### 1. Data Quality & Corporate Action Adjustments (Critical)
**Current State:** The app reads raw "Close" prices from a CSV.
**Issue:** NEPSE listed companies frequently issue Bonus Shares and Right Shares. A 100% right share issuance causes the price to drop by ~50% overnight. Without adjustment, your risk models (especially volatility and jump diffusion) will interpret this as a massive market crash, leading to incorrect risk estimates and signal generation.
**Recommendation:**
- **Implement Adjusted Close Logic:** You need a mechanism to adjust historical prices.
    - *Option A:* Require users to upload "Adjusted Close" data.
    - *Option B (Better):* Create a module that takes a separate "Corporate Actions" CSV (Date, Type, Ratio) and calculates adjusted prices on the fly.
    - *Formula:* $P_{adj} = \frac{P_{close} + (RightPrice \times RightRatio)}{1 + BonusRatio + RightRatio}$

### 2. Realistic Transaction Costs
**Current State:** The app uses a flat `transaction_cost_pct` (default 0.10%).
**Issue:** NEPSE has a tiered commission structure, plus fixed fees.
- **Broker Commission:**
    - Up to Rs 50,000: 0.40%
    - Rs 50,000 - Rs 5 Lakhs: 0.37%
    - Rs 5 Lakhs - Rs 20 Lakhs: 0.34%
    - Rs 20 Lakhs - Rs 1 Crore: 0.30%
    - Above Rs 1 Crore: 0.27%
- **SEBON Fee:** 0.015%
- **DP Charge:** Rs 25 per transaction (flat).
**Recommendation:**
- Replace the simple percentage slider with a `calculate_transaction_cost(amount, shares)` function that implements the tiered structure and fixed fees. This will significantly impact the profitability of smaller trades or high-frequency strategies.

### 3. Taxation (Capital Gains Tax)
**Current State:** No tax calculation is visible in the returns analysis.
**Issue:** In Nepal, CGT is deducted at source (by the broker) but varies by holding period:
- **Short Term (< 365 days):** 7.5%
- **Long Term (> 365 days):** 5.0%
**Recommendation:**
- Add a "Taxation" module. When calculating "Net Profit" or backtesting strategies, apply the appropriate CGT rate based on the holding period of the simulated trade. This is crucial for comparing "Trading" (Short Term) vs "Investing" (Long Term) strategies.

### 4. Market Microstructure & Circuit Breakers
**Current State:** The simulations (GBM, Jump Diffusion) assume continuous trading or standard volatility.
**Issue:** NEPSE has specific circuit breakers that halt trading:
- **Index-based:** 4%, 5%, 6% halts.
- **Stock-based:** +/- 10% daily price limit.
**Recommendation:**
- **Price Limits:** In your Monte Carlo simulations, cap the maximum daily move to +/- 10%. This prevents the model from predicting unrealistic single-day gains/losses (e.g., a 20% single-day jump is impossible in NEPSE).
- **Trading Hours:** Ensure any intraday logic (if added later) respects the 11:00 AM - 3:00 PM window and the Pre-open session (10:30 - 10:45 AM).

### 5. Sector-Specific Analysis
**Current State:** Generic asset analysis.
**Issue:** NEPSE is heavily skewed towards Hydropower and Banking/Microfinance. These sectors behave very differently. Hydros are often "High Beta" and volatile, while Commercial Banks are "Low Beta" and dividend-heavy.
**Recommendation:**
- Add a "Sector" dropdown in the UI.
- Implement sector-specific risk defaults. For example, if "Hydropower" is selected, default to a higher volatility assumption or a different regime detection sensitivity.

### 6. Calendar & Holidays
**Current State:** `NEPSE_TRADING_DAYS = 240` in `config.py`.
**Issue:** Nepal has a unique holiday calendar (public holidays, festivals) that doesn't align with international markets.
**Recommendation:**
- Ensure your annualized metrics (Sharpe, Volatility) consistently use `240` (or the exact average for the last few years). The current config is good, but verify it matches the actual data density.

### 7. Beta Calculation Reference
**Current State:** Beta is likely calculated against the uploaded file or a default.
**Issue:** Beta should be calculated against the **NEPSE Alpha Index** or the broad **NEPSE Index**.
**Recommendation:**
- Ensure the user always has the NEPSE Index loaded as the "Benchmark" for Beta calculations. You might want to bundle a `nepse_index_history.csv` to always be available as the default benchmark, regardless of what stock the user uploads.

## Summary of Immediate Action Items
1.  **Modify `nepse_quant_pro/config.py`**: Add constants for Tax Rates and Commission Tiers.
2.  **Update `nepse_quant_pro/returns.py`**: Add a `calculate_net_return` function that subtracts the exact commissions and taxes.
3.  **Update `nepse_quant_pro/risk_engine.py`**: Clip simulated daily returns to +/- 10% to respect NEPSE circuit breakers.
