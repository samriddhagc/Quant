import argparse
import os
import pandas as pd
from datetime import datetime
from nepse_quant_pro.data_io import get_dynamic_data

# --- CONFIGURATION ---
PORTFOLIO_FILE = "paper_portfolio.csv"
PNL_LOG_FILE = "paper_pnl_log.csv"
INPUT_ORDERS_FILE = "buy_orders.csv"

# --- NEPSE FEE CALCULATOR ---
class NepseFees:
    """
    Exact NEPSE Fee Structure implementation.
    """
    @staticmethod
    def calculate_broker_commission(amount: float) -> float:
        if amount <= 2500:
            return 10.0
        elif amount <= 50000:
            return amount * 0.0036
        elif amount <= 500000:
            return amount * 0.0033
        elif amount <= 2000000:
            return amount * 0.0031
        elif amount <= 10000000:
            return amount * 0.0027
        else:
            return amount * 0.0024

    @staticmethod
    def calculate_total_fees(qty: int, price: float, is_buy: bool = True) -> dict:
        amount = qty * price
        
        # 1. Broker Commission
        commission = NepseFees.calculate_broker_commission(amount)
        
        # 2. SEBON Fee (0.015%)
        sebon_fee = amount * 0.00015
        
        # 3. DP Charge (Rs 25 per stock per transaction)
        dp_charge = 25.0
        
        total_fees = commission + sebon_fee + dp_charge
        
        return {
            "commission": commission,
            "sebon": sebon_fee,
            "dp_charge": dp_charge,
            "total": total_fees
        }

    @staticmethod
    def calculate_tax(net_gain: float, holding_period_days: int = 20) -> float:
        """
        Capital Gains Tax (CGT).
        < 1 Year (365 days): 7.5%
        >= 1 Year: 5%
        """
        if net_gain <= 0:
            return 0.0
        
        tax_rate = 0.075 if holding_period_days < 365 else 0.05
        return net_gain * tax_rate

# --- ACTIONS ---

def initialize_portfolio():
    """Reads buy_orders.csv and creates the initial portfolio with buy costs."""
    if not os.path.exists(INPUT_ORDERS_FILE):
        print(f"❌ Error: {INPUT_ORDERS_FILE} not found. Run the strategy first.")
        return

    print(f"📥 Loading orders from {INPUT_ORDERS_FILE}...")
    orders = pd.read_csv(INPUT_ORDERS_FILE)
    
    portfolio_records = []
    total_investment = 0.0
    total_buy_fees = 0.0

    today = datetime.now().strftime("%Y-%m-%d")

    print("\n--- EXECUTING BUY ORDERS (PAPER) ---")
    print(f"{'Symbol':<10} {'Qty':<8} {'Price':<10} {'Amount':<15} {'Fees':<10}")
    print("-" * 60)

    for _, row in orders.iterrows():
        symbol = row['Symbol']
        
        # Parse Qty and Price (Handle string formatting if present)
        try:
            qty = int(str(row['Shares']).replace(',', ''))
            # We assume the 'Value (NPR)' was based on the latest price
            # Let's fetch the real latest price to be accurate
            df_data = get_dynamic_data(symbol)
            if df_data is None or df_data.empty:
                print(f"⚠️ Warning: Could not fetch price for {symbol}. Skipping.")
                continue
            
            buy_price = float(df_data['Close'].iloc[-1])
        except Exception as e:
            print(f"Error parsing data for {symbol}: {e}")
            continue

        amount = qty * buy_price
        fees = NepseFees.calculate_total_fees(qty, buy_price, is_buy=True)
        
        total_cost = amount + fees['total']
        
        portfolio_records.append({
            "Entry_Date": today,
            "Symbol": symbol,
            "Quantity": qty,
            "Buy_Price": buy_price,
            "Buy_Amount": amount,
            "Buy_Fees": fees['total'],
            "Total_Cost_Basis": total_cost # Includes fees
        })
        
        total_investment += total_cost
        total_buy_fees += fees['total']
        
        print(f"{symbol:<10} {qty:<8} {buy_price:<10.2f} {amount:<15,.2f} {fees['total']:<10.2f}")

    # Save
    pd.DataFrame(portfolio_records).to_csv(PORTFOLIO_FILE, index=False)
    print("-" * 60)
    print(f"✅ Portfolio initialized in {PORTFOLIO_FILE}")
    print(f"💰 Total Capital Deployment: Rs. {total_investment:,.2f}")
    print(f"💸 Total Buy Fees Paid:      Rs. {total_buy_fees:,.2f}")


def update_pnl():
    """Fetches live prices and calculates Net P&L (Mark-to-Market)."""
    if not os.path.exists(PORTFOLIO_FILE):
        print(f"❌ Error: {PORTFOLIO_FILE} not found. Run --action buy first.")
        return

    portfolio = pd.read_csv(PORTFOLIO_FILE)
    today = datetime.now().strftime("%Y-%m-%d")
    
    print(f"\n--- DAILY P&L UPDATE ({today}) ---")
    print(f"{'Symbol':<10} {'Buy Price':<10} {'LTP':<10} {'Chg%':<8} {'Net Profit':<12}")
    print("-" * 60)

    daily_stats = []
    total_current_value = 0.0
    total_net_profit = 0.0

    for idx, row in portfolio.iterrows():
        symbol = row['Symbol']
        qty = row['Quantity']
        buy_price = row['Buy_Price']
        buy_fees = row['Buy_Fees']
        cost_basis = row['Total_Cost_Basis']
        
        # 1. Fetch Live Price
        try:
            df_data = get_dynamic_data(symbol)
            if df_data is None or df_data.empty:
                print(f"{symbol:<10} {'ERROR':<10}")
                continue
            ltp = float(df_data['Close'].iloc[-1])
        except:
            ltp = buy_price # Fallback
        
        # 2. Calculate Sell Scenario (Hypothetical)
        sell_amount = qty * ltp
        sell_fees = NepseFees.calculate_total_fees(qty, ltp, is_buy=False)['total']
        
        # 3. Calculate Taxes
        # Net Gain for Tax = (Sell Amount - Sell Fees) - (Buy Amount + Buy Fees)
        # Note: Tax is on the *Gain*, not the total value.
        taxable_gain = (sell_amount - sell_fees) - cost_basis
        
        cgt_tax = 0.0
        if taxable_gain > 0:
            cgt_tax = NepseFees.calculate_tax(taxable_gain, holding_period_days=20)
            
        # 4. Net Profit (In Hand)
        # Net Profit = Sell Amount - Sell Fees - Tax - Cost Basis
        net_profit = sell_amount - sell_fees - cgt_tax - cost_basis
        
        change_pct = (ltp - buy_price) / buy_price * 100
        
        total_current_value += sell_amount
        total_net_profit += net_profit
        
        print(f"{symbol:<10} {buy_price:<10.2f} {ltp:<10.2f} {change_pct:^8.2f}% {net_profit:<12.2f}")
        
        daily_stats.append({
            "Date": today,
            "Symbol": symbol,
            "LTP": ltp,
            "Net_Profit": net_profit,
            "Tax_Paid": cgt_tax
        })

    # Log to history
    log_df = pd.DataFrame(daily_stats)
    if not os.path.exists(PNL_LOG_FILE):
        log_df.to_csv(PNL_LOG_FILE, index=False)
    else:
        log_df.to_csv(PNL_LOG_FILE, mode='a', header=False, index=False)

    print("-" * 60)
    print(f"📊 Portfolio Value: Rs. {total_current_value:,.2f}")
    print(f"💵 Total Net P&L:   Rs. {total_net_profit:,.2f}")
    if total_net_profit > 0:
        print("🚀 STATUS: PROFITABLE")
    else:
        print("🔻 STATUS: LOSS")

def main():
    parser = argparse.ArgumentParser(description="NEPSE Paper Trading Tracker")
    parser.add_argument("--action", choices=["buy", "update"], required=True, help="Use 'buy' to initialize or 'update' to track P&L.")
    args = parser.parse_args()

    if args.action == "buy":
        initialize_portfolio()
    elif args.action == "update":
        update_pnl()

if __name__ == "__main__":
    main()