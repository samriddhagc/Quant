import pandas as pd
import numpy as np
from nepse_quant_pro.data_io import get_dynamic_data
from nepse_quant_pro.returns import compute_log_returns, estimate_mu_sigma
from nepse_quant_pro.risk_engine import run_canonical_engine, decision_engine
from nepse_quant_pro.config import PERIODS_PER_YEAR

# --- CONFIGURATION ---
CAPITAL = 1_000_000  # Example Portfolio Size (NPR)
RISK_AVERSION = 0.5  # Lambda (0.5 is standard/aggressive, 2.0 is conservative)
MAX_POS_SIZE = 0.20  # Max 20% in one stock

def main():
    # 1. Load the "Gold Tier" Signals
    print("🚀 Loading Alpha Signals...")
    results = pd.read_csv("cv_batch_results.csv")
    
    # Filter for the "Tradeable" universe we identified (CV Score > 0.52)
    candidates = results[results['cv_score'] > 0.52].copy()
    print(f"Found {len(candidates)} candidates for Risk Analysis.\n")

    portfolio_allocations = []

    for _, row in candidates.iterrows():
        symbol = row['symbol']
        # Use CV Score as the raw probability because 'probability' col is capped by safety rules
        # If CV score is high (e.g. 0.63), we want to use that edge.
        ai_probability = row['cv_score'] 
        
        print(f"Analyzing {symbol} (Signal Strength: {ai_probability:.1%})...")

        # 2. Fetch Data for Risk Engine
        try:
            df = get_dynamic_data(symbol)
            if df is None or df.empty:
                continue
            
            # 3. Calculate Inputs for GBM
            log_rets = compute_log_returns(df['Close'])
            mu_sigma = estimate_mu_sigma(log_rets)
            current_price = df['Close'].iloc[-1]
            
            # 4. Run Monte Carlo (The Risk Engine)
            # We use a 20-day horizon to match your new "Swing" logic
            mc_result = run_canonical_engine(
                current_price=current_price,
                sigma_daily=mu_sigma['daily_sigma'],
                er_annual=mu_sigma['annual_mu'],
                rf_annual=0.06,
                horizon_days=20,  # MATCHING YOUR NEW HORIZON
                sims=5000,
                es_conf_level=0.95,
                distribution="Student-t (fat-tailed)",
                student_df=8,
                return_generation_method="GBM"
            )

            # 5. INTEGRATION POINT: Fuse Signal + Risk
            # We pass the 'ai_probability' explicitly to the decision engine
            direction, size, edge, fused_prob, reasons = decision_engine(
                er_blended=mc_result['er_blended'],
                expected_shortfall=mc_result['es_annual'],
                prob_gain=mc_result['prob_gain'], # MC's probability (usually ~50%)
                terminal_returns=mc_result['terminal_returns'],
                sim_kurtosis=mc_result['sim_kurtosis'],
                risk_aversion_lambda=RISK_AVERSION,
                ai_raw_prob=ai_probability,       # <--- HERE IS THE FUSION
                kelly_cap=MAX_POS_SIZE
            )

            if direction == "Bullish" and size > 0.0:
                shares = int((CAPITAL * size) / current_price)
                portfolio_allocations.append({
                    "Symbol": symbol,
                    "Signal": f"{ai_probability:.1%}",
                    "Kelly Size": f"{size:.1%}",
                    "Value (NPR)": f"{size * CAPITAL:,.0f}",
                    "Shares": shares,
                    "Expected Edge": f"{edge:.2%}",
                    "Downside Risk (ES)": f"{mc_result['es_annual']:.1%}"
                })
        
        except Exception as e:
            print(f"Skipping {symbol}: {e}")

    # 6. Output Final Portfolio
    if portfolio_allocations:
        final_df = pd.DataFrame(portfolio_allocations)
        final_df = final_df.sort_values(by="Value (NPR)", ascending=False)
        print("\n🏆 FINAL OPTIMIZED PORTFOLIO 🏆")
        print(final_df.to_string(index=False))
        final_df.to_csv("buy_orders.csv", index=False)
        print("\nSaved orders to buy_orders.csv")
    else:
        print("No trades passed the Risk Gate.")

if __name__ == "__main__":
    main()