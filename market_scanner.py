import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from nepse_quant_pro.returns import compute_log_returns
from nepse_quant_pro.risk_engine import james_stein_drift
from nepse_quant_pro.config import (
    RISK_AVERSION_LAMBDA,
    RECENT_DRIFT_WINDOW,
)
import numpy as np

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

MAX_POSITION_SIZE = 0.30


def _load_risk_artifact(symbol: str):
    """Return the first available risk artifact for the symbol."""
    for selector in RISK_SELECTORS:
        artifact = selector.load(symbol)
        if artifact:
            artifact.setdefault("engine", selector.namespace)
            return artifact
    return None


def _approx_terminal_returns(dist_payload: dict | None) -> np.ndarray | None:
    if not dist_payload:
        return None
    bins = dist_payload.get("bins") or []
    counts = dist_payload.get("counts") or []
    if len(bins) < 2 or len(counts) != len(bins) - 1:
        return None
    centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(counts))]
    expanded = np.repeat(centers, counts).astype(float)
    if expanded.size == 0:
        return None
    return expanded


def fetch_data_task(ticker: str):
    """Worker function for I/O threading (network bound)."""
    try:
        df = get_dynamic_data(ticker)
        return ticker, df
    except Exception:
        return ticker, None


def _classify_direction(probability: float) -> str:
    """Map raw probability to a directional label."""
    if probability is None:
        return "Neutral"
    if probability >= 0.60:
        return "Bullish"
    if probability <= 0.40:
        return "Bearish"
    return "Neutral"


def analyze_task(ticker: str, df: pd.DataFrame):
    """
    Worker function for CPU processing.
    NEW BEHAVIOR: no live training. We consume artifacts produced by the batch jobs.
    """
    try:
        if df is None or df.empty:
            return None

        cv_artifact = CV_SELECTOR.load(ticker)
        if not cv_artifact:
            logging.debug("No CV artifact for %s", ticker)
            return None

        meta = cv_artifact.get("meta", {})
        cv_score = meta.get("cv_score")
        ai_prob = meta.get("probability")
        prob_edge = meta.get("prob_edge", 0.0)

        if cv_score is None or ai_prob is None:
            logging.debug("Incomplete CV artifact for %s", ticker)
            return None

        if cv_score < 0.53:
            return None

        log_returns = compute_log_returns(df["Close"])
        recent_returns = log_returns.tail(max(RECENT_DRIFT_WINDOW, 60))
        if recent_returns.empty or len(recent_returns) < 30:
            return None

        js_drift = james_stein_drift(recent_returns, annualized=True)

        risk_artifact = _load_risk_artifact(ticker)
        if not risk_artifact:
            return None

        returns = _approx_terminal_returns(risk_artifact.get("distribution"))
        if returns is None or returns.size < 20:
            return None

        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        if wins.size == 0 or losses.size == 0:
            return None

        payoff_ratio = float(np.mean(wins) / abs(np.mean(losses)))
        payoff_ratio = max(payoff_ratio, 1e-3)

        prob_gain = float(max(min(ai_prob, 0.995), 0.005))
        kelly_fraction = prob_gain - (1 - prob_gain) / payoff_ratio
        fractional_kelly = max(0.0, kelly_fraction * 0.5)
        fractional_kelly = min(fractional_kelly, MAX_POSITION_SIZE)

        direction = _classify_direction(ai_prob)
        if direction == "Neutral" or fractional_kelly <= 0:
            return None

        es_95 = None
        tails = risk_artifact.get("tails", {}) if risk_artifact else {}
        if tails:
            es_95 = tails.get("es_95")

        ai_scalar = (ai_prob - 0.5) * 2.0
        blended_er = js_drift * (1 + 0.5 * ai_scalar)
        downside = abs(es_95) if es_95 is not None else 0.40
        risk_aversion = RISK_AVERSION_LAMBDA or 0.50
        utility_score = blended_er - (risk_aversion * downside)

        current_price = float(df["Close"].iloc[-1])

        kelly_pct = fractional_kelly * 100
        es_val = abs(es_95) if es_95 is not None else None

        return {
            "Symbol": ticker,
            "Price": current_price,
            "Signal": direction,
            "Alloc %": round(kelly_pct, 1),
            "CV Score": round(cv_score * 100, 1),
            "AI Prob": round(ai_prob * 100, 1),
            "MC Prob": None,
            "Utility": round(utility_score * 100, 2),
            "Regime": risk_artifact.get("engine", "Artifact"),
            "ES 95": es_val,
            "VaR 95": None,
            "JS Drift": round(js_drift * 100, 2),
            "Payoff": round(payoff_ratio, 2),
            "Kelly Raw": round(kelly_fraction * 100, 2),
            "Updated": cv_artifact.get("updated_at", "N/A"),
        }

    except Exception as e:
        logging.error("Scanner fail on %s: %s", ticker, e)
        return None


def parallel_scan(symbols, io_workers: int | None = None) -> pd.DataFrame:
    """
    Two-stage parallel scan with AI Integration.
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

    print(f"\n✅ {len(data_map)} symbols ready for AI analysis.")

    if not data_map:
        return pd.DataFrame()

    print(f"🧠 Phase 2: Loading artifacts & applying Citadel Gate...")
    results: list[dict] = []

    for sym in tqdm(data_map.keys(), desc="Processing"):
        res = analyze_task(sym, data_map[sym])
        if res:
            results.append(res)

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("--- INSTITUTIONAL MARKET SCANNER (CV > 53%) ---")
    df = parallel_scan(SYMBOLS)

    if not df.empty:
        # Sort by Utility (Best Risk-Adjusted Bet) to find the "Best"
        # Since we already filtered for CV > 53, everything here is "Valid Alpha"
        top_picks = (
            df.sort_values(by="Utility", ascending=False)
              .reset_index(drop=True)
        )
        
        print("\n🏆 TOP ALPHA OPPORTUNITIES (CV Verified) 🏆")
        print(top_picks.to_string(index=False))
        
        # Save results
        top_picks.to_csv("daily_top_picks.csv", index=False)
        print(f"\n💾 Found {len(top_picks)} valid opportunities. Saved to daily_top_picks.csv")
    else:
        print("\n⚠️ No stocks passed the 'Citadel Gate' (CV > 53%). Market condition is likely poor/noisy.")
