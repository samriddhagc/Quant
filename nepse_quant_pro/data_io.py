from io import StringIO
from typing import List, Optional
import pandas as pd
import streamlit as st
import requests
import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
# Ensure .database module is available for init_db, etc.
from .database import init_db, get_latest_date, save_to_db, load_from_db

# Initialize DB on first import
init_db()

# --- OLD CSV LOGIC (Preserved) ---
def load_csv(file) -> Optional[pd.DataFrame]:
    try:
        if isinstance(file, str):
            return pd.read_csv(file)
        stringio = StringIO(file.getvalue().decode("utf-8"))
        return pd.read_csv(stringio)
    except Exception as exc:
        st.error(f"Error loading CSV: {exc}")
        return None

def extract_price_matrix(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        raise ValueError("Input DataFrame is empty.")
    df = raw_df.copy()
    df.columns = [c.strip() for c in df.columns]
    
    date_candidates = [c for c in df.columns if "date" in c.lower()]
    date_col = "Date" if "Date" in df.columns else (date_candidates[0] if date_candidates else df.columns[0])
    
    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.sort_values("Date").set_index("Date")
    
    price_data = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col == date_col: continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notnull().sum() > 0:
            price_data[col] = numeric
            
    # [ROBUSTNESS FIX] Fill Volume NaNs with 0 before final dropna
    if 'Volume' in price_data.columns:
        price_data['Volume'] = price_data['Volume'].fillna(0)
            
    return price_data.dropna(how="all").ffill().dropna()

def load_macro_series(file) -> Optional[pd.Series]:
    if file is None: return None
    macro_df = load_csv(file)
    if macro_df is None or macro_df.empty: return None
    date_col = macro_df.columns[0]
    val_col = macro_df.columns[1]
    macro_df[date_col] = pd.to_datetime(macro_df[date_col])
    return macro_df.set_index(date_col)[val_col]

# --- NEW SMART FETCH LOGIC (Corrected for Continuity) ---

def fetch_chunk(symbol, start_ts, end_ts):
    """Fetches a single chunk from the API."""
    url = "https://merolagani.com/handlers/TechnicalChartHandler.ashx"
    params = {
        "type": "get_advanced_chart", "symbol": symbol, "resolution": "1D",
        "rangeStartDate": int(start_ts), "rangeEndDate": int(end_ts),
        "isAdjust": "1", "currencyCode": "NPR"
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://merolagani.com/CompanyDetail.aspx?symbol=" + symbol,
        "X-Requested-With": "XMLHttpRequest"
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        data = r.json()
        if "t" not in data or len(data["t"]) == 0: return pd.DataFrame()
        return pd.DataFrame({
            "Date": pd.to_datetime(data["t"], unit="s"),
            "Open": data["o"], "High": data["h"], "Low": data["l"], "Close": data["c"], "Volume": data["v"]
        })
    except:
        return pd.DataFrame()

def fetch_live_candle(symbol):
    """Scrapes the live price (LTP) from the main page."""
    url = "https://merolagani.com/CompanyDetail.aspx?symbol=" + symbol
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        ltp = soup.select_one("#ctl00_ContentPlaceHolder1_CompanyDetail1_lblMarketPrice")
        if ltp:
            price = float(ltp.text.replace(",", ""))
            return pd.DataFrame([{
                "Date": pd.to_datetime(datetime.now().date()),
                "Open": price, "High": price, "Low": price, "Close": price, "Volume": 0
            }])
    except:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner="Syncing Market Data...")
def get_dynamic_data(symbol):
    """
    ROBUST DATA SYNCHRONIZATION LOGIC.
    Performs a single, maximum-range fetch for stability.
    """
    symbol = symbol.upper().strip()
    last_date = get_latest_date(symbol)
    today = datetime.now().date()
    new_data = pd.DataFrame()
    
    # --- STEP 1: Initial Full Fetch (The FIX for Fragmentation) ---
    if last_date is None:
        # Fetch up to ~10 years of history to cover multiple regimes
        start_of_time = datetime.now() - timedelta(days=365 * 10)
        
        # Use a single, large range fetch for maximum stability
        st.spinner("Fetching full 10-year history...")
        new_data = fetch_chunk(symbol, start_of_time.timestamp(), time.time())
        
    # --- STEP 2: Incremental Update ---
    elif last_date < today:
        st.spinner(f"Fetching updates since {last_date}...")
        start_ts = pd.Timestamp(last_date).timestamp()
        new_data = fetch_chunk(symbol, start_ts, time.time())
        
    # --- STEP 3: Live Candle (Today's Price) ---
    live_df = fetch_live_candle(symbol)
    
    if not new_data.empty:
        new_data = pd.concat([new_data, live_df])
        new_data = new_data.drop_duplicates(subset=["Date"], keep="last")
        save_to_db(new_data, symbol)
    elif live_df.empty and last_date is None:
        st.error(f"Could not fetch any data for {symbol}.")
        return pd.DataFrame()
    
    # Return the entire history from the database (guaranteed alignment)
    return load_from_db(symbol)

__all__ = ["load_csv", "extract_price_matrix", "load_macro_series", "get_dynamic_data"]
