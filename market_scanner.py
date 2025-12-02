import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

# --- CITADEL-LEVEL CLEANING ---
# 1. Suppress Streamlit "No Runtime" Warnings (Harmless noise)
logging.getLogger('streamlit.runtime.caching.cache_data_api').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime.scriptrunner.script_runner').setLevel(logging.ERROR)

# 2. Suppress Pandas/Numpy FutureWarnings
warnings.filterwarnings("ignore")

from nepse_quant_pro.artifacts import ArtifactSelector
from nepse_quant_pro.data_io import get_dynamic_data
from nepse_quant_pro.features_robust import (
    get_shannon_entropy,
    get_market_efficiency_ratio
)
from nepse_quant_pro.sectors import SECTOR_LOOKUP

# --- EXPANDED WATCHLIST (Approx 200+ Active Scripts) ---
ALL_SYMBOLS = [
    # Commercial Banks
    "NABIL", "NICA", "EBL", "GBIME", "PCBL", "SANIMA", "SBI", "SCB", "ADBL", "NBL",
    "PRVU", "NIMB", "LSL", "KBL", "MBL", "SBL", "CZBIL", "HBL", "NMB", "RBB",

    # Development Banks
    "MNBBL", "GBBL", "JBBL", "LBBL", "MLBL", "SHINE", "KSBBL", "EDBL", "MDB",
    "SADBL", "SAPDBL", "SINDU", "KRBL", "GRDBL", "CORBL",

    # Finance
    "ICFC", "RLFL", "GFCL", "MFIL", "PFL", "CFCL", "BFC", "GUFL", "JFL", "MPFL",
    "NFS", "PROFL", "SFCL", "SIFC", "UFL",

    # Hydropower
    "API", "UPPER", "CHCL", "HIDCL", "AKPL", "RHPL", "NHPC", "SHPC", "BPCL", "NGPL",
    "AHPC", "AKJCL", "BARUN", "BNHC", "CHL", "DHPL", "GHL", "HDHPC", "HURJA", "JOSHI",
    "KKHC", "KPCL", "LEC", "MBJC", "MEN", "MHNL", "MKJC", "NBJCL", "NJPL", "PMHPL",
    "PPCL", "RADHI", "RFPL", "RHPC", "RRHP", "RSDC", "SAHAS", "SHEL", "SIKLES",
    "SJCL", "SMH", "SMJC", "SPDL", "SPL", "SSHL", "TPC", "UNHPL", "USHEC", "MKHC",

    # Manufacturing / Others
    "SHIVM", "SONA", "GCIL", "HDL", "BNT", "UNL", "STC",

    # Insurance
    "NLIC", "LICN", "ALICL", "SICL", "NIL", "IGI", "RBCL", "NICL", "NLG", "PRIN",
    "PIC", "PICL", "UIC", "EIC", "SIC", "LGIL", "SJLIC", "SNLI", "CLI", "LIL",

    # Microfinance
    "CBBL", "DDBL", "FOWAD", "GMFBS", "JSLBB", "KMCDB", "LLBS", "MERO", "MLBBL",
    "MLBS", "MSLB", "NICLBSL", "NUBL", "RSDC", "SABSL", "SDLBSL", "SKBBL", "SMB",
    "SMATA", "SMFDB", "SWBBL", "VLBS", "WNLB",

    # Investment / Others
    "NIFRA", "NRIC", "CIT", "HIDCL", "HRL", "NRN",
]

# Deduplicate and sort
SYMBOLS = sorted(set(ALL_SYMBOLS))

CV_SELECTOR = ArtifactSelector("cv_batch")
# Prefer GBM artifacts, fall back to bootstrap then legacy namespace
RISK_SELECTORS = [
    ArtifactSelector(ns) for ns in ("risk_gbm", "risk_bootstrap", "risk_job")
]


def _load_risk_artifact(symbol: str):
    """Return the first available risk artifact for the symbol."""
    for selector in RISK_SELECTORS:
        artifact = selector.load(symbol)
        if artifact:
            artifact.setdefault("engine", selector.namespace)
            return artifact
    return None


def fetch_data_task(ticker: str):
    """Worker function for I/O threading (network bound)."""
    try:
        df = get_dynamic_data(ticker)
        return ticker, df
    except Exception:
        return ticker, None


def analyze_task(ticker: str, df: pd.DataFrame):
    """
    Worker function for CPU processing.
    PIVOT: Replaced Directional Prediction with Regime & Relative Ranking.
    """
    try:
        if df is None or df.empty or len(df) < 60:
            return None

        # --- 1. REGIME IDENTIFICATION (Governance Layer) ---
        # Shannon Entropy: Measures noise. > 0.75 implies Random Walk (Unpredictable).
        entropy_series = get_shannon_entropy(df['Close'], window=60)
        current_entropy = entropy_series.iloc[-1]

        # Efficiency Ratio (Kaufman): Measures trend strength vs noise.
        # High (>0.3) = Trending, Low (<0.2) = Choppy/Mean Reverting.
        er_series = get_market_efficiency_ratio(df['Close'], window=30)
        current_er = er_series.iloc[-1]

        # --- 2. LOAD ALPHA SIGNAL (AI Model) ---
        cv_artifact = CV_SELECTOR.load(ticker)
        if not cv_artifact:
            return None

        meta = cv_artifact.get("meta", {})
        cv_score = meta.get("cv_score")
        
        if cv_score is None:
            return None

        # --- 3. HARD REGIME GATE (The "Kill Switch") ---
        # If the market is purely random (High Entropy) AND has no trend structure (Low Efficiency),
        # we reject the trade regardless of what the ML model says.
        # This filters out "Fake Alpha" in choppy markets.
        if current_entropy > 0.82 and current_er < 0.25:
            return None

        # --- 4. RISK & SIZING METRICS ---
        risk_artifact = _load_risk_artifact(ticker)
        es_95 = -0.05 # Default safe fallback
        if risk_artifact:
            tails = risk_artifact.get("tails", {})
            if tails.get("es_95"):
                es_95 = tails.get("es_95")

        current_price = float(df["Close"].iloc[-1])
        sector = SECTOR_LOOKUP.get(ticker, "Others")

        return {
            "Symbol": ticker,
            "Price": current_price,
            "Sector": sector,
            "CV_Score": cv_score,          # Raw probability (0-1)
            "Entropy": current_entropy,    # Market noise (Lower is better)
            "Efficiency": current_er,      # Trend strength (Higher is better)
            "ES_95": abs(es_95),           # Downside risk (Lower is better)
            "Updated": cv_artifact.get("updated_at", "N/A"),
        }

    except Exception as e:
        logging.error("Scanner fail on %s: %s", ticker, e)
        return None


def parallel_scan(symbols, io_workers: int | None = None) -> pd.DataFrame:
    """
    Two-stage parallel scan with Sector-Relative Ranking.
    """
    num_symbols = len(symbols)
    if num_symbols == 0:
        print("⚠️ No symbols supplied.")
        return pd.DataFrame()

    # Worker counts
    if io_workers is None:
        io_workers = min(64, max(8, num_symbols))

    print(f"📡 Phase 1: Fetching data for {num_symbols} stocks...")
    data_map: dict[str, pd.DataFrame] = {}

    # Phase 1: I/O Bound
    with ThreadPoolExecutor(max_workers=io_workers) as executor:
        futures = {executor.submit(fetch_data_task, sym): sym for sym in symbols}
        for future in tqdm(as_completed(futures), total=num_symbols, desc="Downloading"):
            sym, df = future.result()
            if df is not None:
                data_map[sym] = df

    print(f"\n✅ {len(data_map)} symbols ready for Regime Analysis.")

    if not data_map:
        return pd.DataFrame()

    print(f"🧠 Phase 2: Calculating Regime Metrics & Relative Alpha...")
    results: list[dict] = []

    for sym in tqdm(data_map.keys(), desc="Processing"):
        res = analyze_task(sym, data_map[sym])
        if res:
            results.append(res)

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # --- 5. CROSS-SECTIONAL RELATIVE RANKING ---
    # We do not trust absolute probabilities (e.g., 0.51 vs 0.52).
    # We trust relative strength within a sector.
    
    # Calculate Sector Statistics
    sector_stats = df.groupby('Sector')['CV_Score'].agg(['mean', 'std'])
    
    def calculate_z_score(row):
        sec_mean = sector_stats.loc[row['Sector'], 'mean']
        sec_std = sector_stats.loc[row['Sector'], 'std']
        if pd.isna(sec_std) or sec_std == 0:
            return 0.0
        return (row['CV_Score'] - sec_mean) / sec_std

    df['Relative_Alpha'] = df.apply(calculate_z_score, axis=1)

    # --- 6. COMPOSITE SCORING (The "Quant" Score) ---
    # Formula:
    # Score = (Relative_Alpha * 1.0) + (Efficiency * 2.0) - (Entropy * 2.0)
    # Rationale: We reward stocks outperforming peers, reward smooth trends, and penalize noise.
    
    df['Final_Score'] = (
        (df['Relative_Alpha'] * 1.0) + 
        (df['Efficiency'] * 2.0) - 
        (df['Entropy'] * 2.0)
    )

    # Formatting for display
    df['CV_Score'] = (df['CV_Score'] * 100).round(1)
    df['Efficiency'] = df['Efficiency'].round(2)
    df['Entropy'] = df['Entropy'].round(2)
    df['Relative_Alpha'] = df['Relative_Alpha'].round(2)
    df['Final_Score'] = df['Final_Score'].round(2)
    df['ES_95'] = (df['ES_95'] * 100).round(1)

    # Sort by the Composite Score
    df = df.sort_values(by="Final_Score", ascending=False)

    return df


if __name__ == "__main__":
    print("--- INSTITUTIONAL REGIME SCANNER (Relative Alpha) ---")
    df = parallel_scan(SYMBOLS)

    if not df.empty:
        # Filter for top opportunities (Positive Score)
        # We prefer assets with positive composite scores
        top_picks = df[df['Final_Score'] > 0.0].copy()
        
        print("\n🏆 TOP STRUCTURAL OPPORTUNITIES (Ranked by Regime Quality) 🏆")
        # Display key columns
        cols = ["Symbol", "Sector", "Price", "Final_Score", "Efficiency", "Entropy", "Relative_Alpha", "CV_Score"]
        print(top_picks[cols].to_string(index=False))
        
        # Save results
        top_picks.to_csv("daily_alpha_ranking.csv", index=False)
        print(f"\n💾 Found {len(top_picks)} structural opportunities. Saved to daily_alpha_ranking.csv")
    else:
        print("\n⚠️ No stocks passed the Regime Gate. Market is likely High Entropy (Random Walk).")