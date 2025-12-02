import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)
os.environ["STREAMLIT_LOG_LEVEL"] = "error"

from nepse_quant_pro.data_io import get_dynamic_data
from nepse_quant_pro.returns import compute_log_returns
from nepse_quant_pro.factors import (
    build_factor_dataframe,
    fit_signal_model,
    calculate_performance_metrics
)
from nepse_quant_pro.sectors import SECTOR_GROUPS, SECTOR_LOOKUP
from nepse_quant_pro.macro_io import (
    get_cached_forex_series,
    build_world_bank_macro_panel,
    load_world_bank_macro,
    build_imf_macro_panel,
    load_imf_macro,
)
from nepse_quant_pro.store import JobStore
from nepse_quant_pro.job_queue import JobQueue

# -------------------------------
# CONFIG
# -------------------------------
MIN_HISTORY = 250      # lowered to get more results
BACKTEST_HORIZON = 20  # prediction window
IO_WORKERS = 24        # aggressive parallel I/O
CPU_WORKERS = 24       # process everything concurrently (mac-friendly)
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2.0
SNAPSHOT_EVERY = 25
MIN_CONFIDENCE = 0.52
_MACRO_FOREX_SERIES = None
_WORLD_BANK_MACRO = load_world_bank_macro()
_IMF_MACRO = load_imf_macro()


def _get_macro_series():
    global _MACRO_FOREX_SERIES
    if _MACRO_FOREX_SERIES is None:
        _MACRO_FOREX_SERIES = get_cached_forex_series("USD")
    return _MACRO_FOREX_SERIES

# -------------------------------
# WATCHLIST (~35 liquid symbols)
# -------------------------------
SYMBOLS = sorted(SECTOR_LOOKUP.keys())
ARTIFACT_ROOT = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))

# -------------------------------
# PHASE 1 — FAST FETCH
# -------------------------------
def fetch_data(ticker):
    last_error = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            df = get_dynamic_data(ticker)
            if df is None or df.empty:
                last_error = "No Data"
            elif len(df) < MIN_HISTORY:
                return ticker, None, f"Too Short ({len(df)})"
            else:
                return ticker, df, "OK"
        except Exception as e:
            last_error = str(e)
        time.sleep(RETRY_DELAY * attempt)
    return ticker, None, f"Error Fetch: {last_error or 'Unknown'}"


# -------------------------------
# PHASE 2 — SUPER-FAST ANALYSIS
# -------------------------------
def analyze(ticker, df, sector_series=None):
    try:
        sector_name = SECTOR_LOOKUP.get(ticker)
        log_returns = compute_log_returns(df["Close"])
        sector_series = (
            sector_series.reindex(df.index).ffill()
            if sector_series is not None and not sector_series.empty
            else None
        )
        macro_components = []
        forex_series = _get_macro_series()
        if forex_series is not None:
            macro_components.append(
                pd.DataFrame(
                    {"Macro_FX_USD": forex_series.reindex(df.index).ffill()},
                    index=df.index,
                )
            )
        world_bank_panel = build_world_bank_macro_panel(
            df.index, macro_df=_WORLD_BANK_MACRO
        )
        if world_bank_panel is not None:
            macro_components.append(world_bank_panel)
        imf_panel = build_imf_macro_panel(df.index, macro_df=_IMF_MACRO)
        if imf_panel is not None:
            macro_components.append(imf_panel)
        macro_series = (
            pd.concat(macro_components, axis=1).sort_index() if macro_components else None
        )

        factors = build_factor_dataframe(
            df,
            log_returns,
            long_window=80,         # optimized for speed
            horizon=BACKTEST_HORIZON,
            macro_series=macro_series,
            sector_series=sector_series,
            sector_name=sector_name,
        )

        if factors.empty:
            return None, "No Factors"

        model = fit_signal_model(
            factors,
            horizon=BACKTEST_HORIZON,
            sector_name=sector_name,
        )
        if model.history is None or model.history.empty:
            return None, "Model Failed"

        prob = model.probability if model.probability is not None else 0.5
        conf = model.accuracy_cv if model.accuracy_cv is not None else 0.0
        if conf < MIN_CONFIDENCE and prob < 0.55:
            return None, "Low Confidence Ensemble"

        metrics = calculate_performance_metrics(
            model.history,
            log_returns,
            threshold=0.60
        )

        ai_daily_edge = (
            metrics.get("avg_win", 0) * metrics.get("win_rate", 0)
            - metrics.get("avg_loss", 0) * (1 - metrics.get("win_rate", 0))
        )

        result = {
            "Symbol": ticker,
            "Probability": round(prob * 100, 1),
            "Confidence": round(conf * 100, 1) if conf else None,
            "Win Rate": round(metrics["win_rate"] * 100, 1),
            "Trades": metrics["trade_count"],
            "AI Daily Return": ai_daily_edge,
            "Market Return": log_returns.mean(),
            "Beats Market": ai_daily_edge > log_returns.mean(),
            "Profit Factor": round(metrics.get("win_loss_ratio", 0), 2),
            "Components": json.dumps(model.component_probs or {}),
        }

        return result, "OK"

    except Exception as e:
        return None, f"Error Analysis: {str(e)}"


# -------------------------------
# MAIN ENGINE — ULTIMATE FAST MODE
# -------------------------------
def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (float, int, bool, str)):
        return value
    if isinstance(value, (pd.Series, list, tuple)):
        return list(value)
    if pd.isna(value):
        return None
    return value


def _read_symbols_file(path: Optional[str]) -> List[str]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Symbols file not found: {file_path}")
    try:
        df = pd.read_csv(file_path)
        for col in ("Symbol", "symbol", "Ticker", "ticker"):
            if col in df.columns:
                return df[col].dropna().astype(str).str.strip().tolist()
        first_col = df.columns[0]
        return df[first_col].dropna().astype(str).str.strip().tolist()
    except Exception:
        with open(file_path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]


def _notify_slack(webhook: Optional[str], message: str):
    if not webhook:
        return
    try:
        resp = requests.post(webhook, json={"text": message}, timeout=5)
        resp.raise_for_status()
    except Exception as exc:
        logging.warning("Slack notification failed: %s", exc)


def ultra_fast_backtest(symbols, io_workers, cpu_workers, snapshot_path: Path):
    logging.info("📡 Fetching %s symbols (Ultra Fast Mode)...", len(symbols))

    # --- FETCH ALL IN PARALLEL ---
    fetch_results = {}
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        futures = {pool.submit(fetch_data, sym): sym for sym in symbols}
        for fut in tqdm(as_completed(futures), total=len(symbols), desc="Downloading"):
            sym, df, status = fut.result()
            fetch_results[sym] = (df, status)

    valid_data = {s: d for s, (d, status) in fetch_results.items() if status == "OK"}
    skip_info = {}

    for sym, (_, reason) in fetch_results.items():
        if reason != "OK":
            skip_info[reason] = skip_info.get(reason, 0) + 1

    logging.info("✅ Valid datasets: %s", len(valid_data))

    def build_sector_benchmarks(data_by_symbol):
        benchmarks = {}
        for sector, members in SECTOR_GROUPS.items():
            closes = []
            for sym in members:
                df = data_by_symbol.get(sym)
                if df is None or df.empty or "Close" not in df.columns:
                    continue
                close = df["Close"].astype(float).rename(sym)
                closes.append(close)
            if closes:
                joined = pd.concat(closes, axis=1).sort_index()
                benchmarks[sector] = joined.mean(axis=1, skipna=True).rename(f"{sector}_Benchmark")
        return benchmarks

    sector_benchmarks = build_sector_benchmarks(valid_data)

    # --- ANALYZE ALL IN PARALLEL ---
    results = []
    with ThreadPoolExecutor(max_workers=cpu_workers) as pool:
        futures = {}
        for sym, df in valid_data.items():
            sector_name = SECTOR_LOOKUP.get(sym)
            sector_series = sector_benchmarks.get(sector_name)
            futures[pool.submit(analyze, sym, df, sector_series)] = sym
        for idx, fut in enumerate(tqdm(as_completed(futures), total=len(valid_data), desc="Backtesting"), start=1):
            res, reason = fut.result()
            if res:
                results.append(res)
                if len(results) % SNAPSHOT_EVERY == 0:
                    pd.DataFrame(results).to_csv("bulk_backtest_snapshot.csv", index=False)
                    pd.DataFrame(results).to_csv(snapshot_path, index=False)
            else:
                skip_info[reason] = skip_info.get(reason, 0) + 1

    return pd.DataFrame(results), skip_info


# -------------------------------
# MAIN
# -------------------------------
# -------------------------------
# MAIN
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Bulk backtest scraper job.")
    parser.add_argument("--symbols", nargs="+", help="Explicit list of tickers to run.")
    parser.add_argument("--symbols-file", help="CSV/text file with tickers.")
    parser.add_argument("--limit", type=int, default=len(SYMBOLS), help="Limit number of tickers.")
    parser.add_argument("--io-workers", type=int, default=IO_WORKERS, help="Parallel fetch workers.")
    parser.add_argument("--cpu-workers", type=int, default=CPU_WORKERS, help="Parallel analysis workers.")
    parser.add_argument("--artifacts-dir", default=os.environ.get("ARTIFACTS_DIR", "artifacts"), help="Artifact directory.")
    parser.add_argument("--job-name", default="bulk_backtest", help="Job name for artifacts/logs.")
    parser.add_argument("--slack-webhook", default=None, help="Optional Slack webhook URL.")
    return parser.parse_args()


def resolve_symbols(args) -> List[str]:
    if args.symbols:
        picks = [sym.upper().strip() for sym in args.symbols]
    elif args.symbols_file:
        picks = [sym.upper().strip() for sym in _read_symbols_file(args.symbols_file)]
    else:
        picks = SYMBOLS.copy()
    if args.limit and len(picks) > args.limit:
        picks = picks[: args.limit]
    return picks


def _execute_bulk(args):
    store = JobStore(namespace=args.job_name, root=Path(args.artifacts_dir).expanduser().resolve())
    log_file = store.path / "job.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Starting bulk backtest job=%s run_id=%s", args.job_name, store.run_id)
    store.write_metadata(vars(args), name="params.json")
    status_payload = {
        "run_id": store.run_id,
        "artifact_path": str(store.path),
    }

    symbols = resolve_symbols(args)
    if not symbols:
        logging.error("No symbols resolved. Exiting.")
        _notify_slack(args.slack_webhook, "Bulk backtest aborted: no symbols provided.")
        return 1, status_payload

    snapshot_path = store.path / "snapshot.csv"
    df, debug = ultra_fast_backtest(
        symbols,
        io_workers=max(1, args.io_workers),
        cpu_workers=max(1, args.cpu_workers),
        snapshot_path=snapshot_path,
    )

    logging.info("📊 FINAL REPORT")

    if not df.empty:
        # 1. Calculate Summary Stats
        total_stocks = len(df)
        beating_market_count = df['Beats Market'].sum()
        percent_beating = (beating_market_count / total_stocks) * 100

        logging.info("✅ Total Results: %s", total_stocks)
        logging.info("🏆 Beats Market: %s (%.1f%%)", beating_market_count, percent_beating)

        # 2. FILTER: Keep only stocks that beat the market
        winners = df[df["Beats Market"] == True].sort_values("Win Rate", ascending=False)

        if not winners.empty:
            logging.info("🚀 Stocks beating market (%s):\n%s", len(winners), winners[[
                "Symbol", "Win Rate", "Trades", "Profit Factor", "AI Daily Return"
            ]].to_string(index=False))
        else:
            logging.warning("⚠️ No stocks beat the market in this run.")

        # 3. Save to CSV
        df.to_csv("full_market_backtest.csv", index=False)
        summary_parquet = store.write_dataframe(df, name="summary.parquet")
        df.to_csv(store.path / "summary.csv", index=False)
        for record in df.to_dict("records"):
            store.write_symbol_record(record.get("Symbol", "UNKNOWN"), record)
        logging.info("💾 Saved to full_market_backtest.csv and %s", summary_parquet)
        _notify_slack(
            args.slack_webhook,
            f"Bulk backtest complete: {total_stocks} symbols. Summary: {summary_parquet}",
        )

    else:
        logging.error("❌ No results generated.")
        store.write_metadata(debug, name="skip_summary.json")
        _notify_slack(args.slack_webhook, "Bulk backtest produced no results.")

    logging.info("🔍 Debug Summary (Skipped):")
    for reason, count in debug.items():
        logging.info("   • %s: %s", reason, count)
    store.write_metadata(debug, name="skip_summary.json")
    return 0, status_payload


def main():
    args = parse_args()
    from datetime import datetime, timezone

    queue = JobQueue(namespace=args.job_name)
    try:
        with queue.job_lock():
            exit_code, status_payload = _execute_bulk(args)
    except RuntimeError as exc:
        print(str(exc))
        return 1
    completed_at = datetime.now(timezone.utc).isoformat()
    status_payload = status_payload or {}
    status_payload.setdefault("completed_at", completed_at)
    run_id = status_payload.get("run_id") or completed_at
    if exit_code == 0:
        queue.record_completion(run_id, status_payload)
    else:
        queue.record_failure(run_id, status_payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
