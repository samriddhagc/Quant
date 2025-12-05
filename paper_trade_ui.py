import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import time
from datetime import datetime

# Import your existing tracker logic to reuse the 'update_pnl' function
try:
    from paper_trade_tracker import update_pnl, PORTFOLIO_FILE, PNL_LOG_FILE
except ImportError:
    st.error("Could not import 'paper_trade_tracker.py'. Make sure it is in the same directory.")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="NEPSE Paper Trade Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stMetric {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
st.sidebar.title("🛠 Controls")
if st.sidebar.button("🔄 Refresh Live Prices", type="primary"):
    with st.spinner("Fetching latest NEPSE data (Pre-Tax P&L)..."):
        try:
            update_pnl() 
            st.sidebar.success("Updated successfully!")
            time.sleep(1) # Small pause to ensure CSV writes complete
            st.rerun()    # Reload the UI with new data
        except Exception as e:
            st.sidebar.error(f"Update failed: {e}")

st.sidebar.markdown("---")
st.sidebar.info(
    "This dashboard reads from `paper_portfolio.csv` and `paper_pnl_log.csv`. "
    "Use the **Refresh** button to trigger the calculation engine."
)

# --- DATA LOADING ---
def load_data():
    if not os.path.exists(PORTFOLIO_FILE):
        return None, None
    
    portfolio_df = pd.read_csv(PORTFOLIO_FILE)
    
    # Load Logs if available
    if os.path.exists(PNL_LOG_FILE):
        log_df = pd.read_csv(PNL_LOG_FILE)
    else:
        log_df = pd.DataFrame()
        
    return portfolio_df, log_df

# --- MAIN DASHBOARD ---
st.title("📈 Paper Trading Command Center")

portfolio_df, log_df = load_data()

if portfolio_df is None or portfolio_df.empty:
    st.warning("⚠️ No portfolio found. Please run `python paper_trade_tracker.py --action buy` to initialize your positions.")
    st.stop()

# --- 1. LIVE P&L CALCULATION ---
if not log_df.empty:
    # Get the latest entry for each symbol based on Date/Index
    latest_status = log_df.groupby("Symbol").last().reset_index()
    
    # Merge with portfolio (Using new column names: Unrealized_PnL, Est_Tax_Liability)
    merged_df = pd.merge(
        portfolio_df, 
        latest_status[['Symbol', 'LTP', 'Unrealized_PnL', 'Est_Tax_Liability']], 
        on='Symbol', 
        how='left'
    )
    
    # Fill NaN for stocks that might not be in the log yet
    merged_df['Unrealized_PnL'] = merged_df['Unrealized_PnL'].fillna(0.0)
    merged_df['LTP'] = merged_df['LTP'].fillna(merged_df['Buy_Price'])
    
    total_invested = merged_df['Total_Cost_Basis'].sum()
    total_unrealized = merged_df['Unrealized_PnL'].sum()
    
    # Current Equity = Invested Capital + Unrealized P&L
    current_equity = total_invested + total_unrealized
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("💰 Total Equity", f"Rs. {current_equity:,.2f}")
    col2.metric("📉 Total Invested", f"Rs. {total_invested:,.2f}")
    
    col3.metric(
        "🚀 Unrealized P&L", 
        f"Rs. {total_unrealized:,.2f}", 
        delta=f"{(total_unrealized/total_invested)*100:.2f}%" if total_invested > 0 else "0%",
        help="Net of Fees, Pre-Tax"
    )
    
    # Show Tax as a "Liability" separate from P&L
    total_est_tax = merged_df['Est_Tax_Liability'].sum()
    col4.metric(
        "🏛️ Potential Tax", 
        f"Rs. {total_est_tax:,.2f}", 
        help="Estimated liability if you sold everything today."
    )

    st.markdown("---")

    # --- 2. CHARTS SECTION ---
    chart_col1, chart_col2 = st.columns([2, 1])
    
    with chart_col1:
        st.subheader("📊 Performance History (Unrealized)")
        if not log_df.empty and 'Unrealized_PnL' in log_df.columns:
            # Group by Date
            daily_equity = log_df.groupby("Date")['Unrealized_PnL'].sum().reset_index()
            
            fig_perf = px.line(
                daily_equity, 
                x='Date', 
                y='Unrealized_PnL', 
                markers=True,
                title="Cumulative Unrealized P&L",
                line_shape='spline'
            )
            fig_perf.update_traces(line_color='#00CC96', line_width=3)
            fig_perf.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.info("Run 'Refresh Live Prices' at least once to populate the chart.")

    with chart_col2:
        st.subheader("🍰 Asset Allocation")
        # Allocation based on Current Value (LTP * Qty)
        merged_df['Current_Value'] = merged_df['Quantity'] * merged_df['LTP']
        
        fig_pie = px.pie(
            merged_df, 
            values='Current_Value', 
            names='Symbol', 
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig_pie.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- 3. DETAILED HOLDINGS ---
    st.subheader("📜 Current Holdings")
    
    display_df = merged_df[[
        'Symbol', 'Entry_Date', 'Quantity', 'Buy_Price', 'LTP', 
        'Total_Cost_Basis', 'Current_Value', 'Unrealized_PnL', 'Est_Tax_Liability'
    ]].copy()
    
    display_df['Return %'] = ((display_df['LTP'] - display_df['Buy_Price']) / display_df['Buy_Price']) * 100
    
    st.dataframe(
        display_df.style.format({
            "Buy_Price": "Rs. {:.2f}",
            "LTP": "Rs. {:.2f}",
            "Total_Cost_Basis": "Rs. {:,.2f}",
            "Current_Value": "Rs. {:,.2f}",
            "Unrealized_PnL": "Rs. {:,.2f}",
            "Est_Tax_Liability": "Rs. {:,.2f}",
            "Return %": "{:.2f}%"
        }).background_gradient(subset=['Unrealized_PnL'], cmap='RdYlGn', vmin=-5000, vmax=5000),
        use_container_width=True,
        height=400
    )

else:
    st.info("No log data available yet. Click 'Refresh Live Prices' in the sidebar.")