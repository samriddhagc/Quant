import sqlite3
import pandas as pd
import os

# Database file will be created in your project root
DB_FILE = "nepse_market_data.db"

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    return conn

def init_db():
    """Creates the table if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            symbol TEXT,
            date DATE,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, date)
        )
    ''')
    conn.commit()
    conn.close()

def get_latest_date(symbol):
    """Returns the most recent date we have for a symbol, or None."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(date) FROM stock_prices WHERE symbol = ?", (symbol,))
    result = cursor.fetchone()[0]
    conn.close()
    return pd.to_datetime(result).date() if result else None

def save_to_db(df, symbol):
    """Saves new data to SQLite, ignoring duplicates."""
    if df.empty: return
    
    conn = get_db_connection()
    df = df.copy()
    df["symbol"] = symbol
    # Ensure date is string YYYY-MM-DD for SQLite
    df["date"] = pd.to_datetime(df["Date"]).dt.strftime('%Y-%m-%d')
    
    # Rename columns to match DB schema
    df_to_save = df[["symbol", "date", "Open", "High", "Low", "Close", "Volume"]]
    df_to_save.columns = ["symbol", "date", "open", "high", "low", "close", "volume"]
    
    # Efficient Upsert (Insert or Ignore)
    cursor = conn.cursor()
    cursor.executemany('''
        INSERT OR REPLACE INTO stock_prices (symbol, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', df_to_save.values.tolist())
    
    conn.commit()
    conn.close()

def load_from_db(symbol):
    """Loads full history for a symbol."""
    conn = get_db_connection()
    query = """
        SELECT date as Date, open as Open, high as High, low as Low, close as Close, volume as Volume 
        FROM stock_prices 
        WHERE symbol = ? 
        ORDER BY date ASC
    """
    df = pd.read_sql(query, conn, params=(symbol,))
    conn.close()
    
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    return df