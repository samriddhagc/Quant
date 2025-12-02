import argparse
import json
import logging
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from nepse_quant_pro.data_io import get_dynamic_data
from nepse_quant_pro.returns import compute_log_returns
from nepse_quant_pro.factors import build_factor_dataframe, fit_signal_model
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
from nepse_quant_pro.data_quality import (
    align_to_trading_calendar,
    compute_symbol_health,
    SymbolHealth,
    resample_event_bars,
)
from nepse_quant_pro.horizons import resolve_symbol_horizon
from nepse_quant_pro.validation import run_post_cv_validation
from nepse_quant_pro.version import MODEL_VERSION

DEFAULT_SYMBOLS = sorted(SECTOR_LOOKUP.keys())


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if pd.isna(value):
        return None
    return value


def _series_payload(series):
    if series is None:
        return None
    try:
        items = series.items()
    except Exception:
        return None
    payload = []
    for idx, val in items:
        payload.append({"name": str(idx), "value": _json_safe(val)})
    return payload or None


def _df_payload(df: Optional[pd.DataFrame]):
    if df is None or df.empty:
        return None
    records = []
    for _, row in df.iterrows():
        entry = {}
        for col, val in row.items():
            entry[str(col)] = _json_safe(val)
        records.append(entry)
    return records or None


def _history_payload(history_df: Optional[pd.DataFrame]) -> Optional[list]:
    if history_df is None or history_df.empty:
        return None
    df = history_df.copy().reset_index().rename(columns={"index": "Date"})
    payload = []
    for _, row in df.iterrows():
        entry = {"date": _json_safe(row.get("Date"))}
        for col, val in row.items():
            if col == "Date":
                continue
            entry[str(col).lower()] = _json_safe(val)
        payload.append(entry)
    return payload or None


def _simulate_probability_backtest(
    history_df: Optional[pd.DataFrame],
    log_returns: pd.Series,
    threshold: float = 0.6,
) -> Optional[dict]:
    if history_df is None or history_df.empty:
        return None
    df = history_df.copy()
    prob_col = None
    for col in df.columns:
        if "prob" in col.lower():
            prob_col = col
            break
    if prob_col is None:
        return None
    aligned = pd.DataFrame(index=df.index)
    aligned["prob"] = df[prob_col].astype(float)
    aligned = aligned.join(log_returns.rename("log_ret"), how="inner").dropna()
    if aligned.empty:
        return None
    aligned["signal"] = (aligned["prob"] >= threshold).astype(int)
    aligned["strat_ret"] = aligned["signal"] * aligned["log_ret"].shift(-1).fillna(0.0)
    aligned["equity"] = np.exp(aligned["strat_ret"].cumsum())
    equity_payload = [
        {"date": _json_safe(idx), "equity": _json_safe(val)}
        for idx, val in aligned["equity"].items()
    ]
    total_return = float(aligned["equity"].iloc[-1] - 1.0)
    days = len(aligned)
    cagr = (aligned["equity"].iloc[-1]) ** (252 / max(days, 1)) - 1.0 if days > 0 else 0.0
    drawdown = aligned["equity"] / aligned["equity"].cummax() - 1.0
    max_dd = float(drawdown.min())
    win_rate = float((aligned["strat_ret"] > 0).mean())
    return {
        "entry_threshold": threshold,
        "equity_curve": equity_payload,
        "stats": {
            "total_return": total_return,
            "cagr": cagr,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
        },
    }


def _build_symbol_artifact(
    summary: dict,
    history_payload: Optional[list],
    backtest_payload: Optional[dict],
    extras: Optional[dict] = None,
    warnings: Optional[list] = None,
    message: Optional[str] = None,
) -> dict:
    warning_list = []
    if warnings:
        warning_list.extend(warnings)
    summary_warning = summary.get("warning")
    if summary_warning:
        warning_list.append(summary_warning)
    meta = {
        "probability": summary.get("probability"),
        "cv_score": summary.get("cv_score"),
        "cv_std": summary.get("cv_std"),
        "prob_edge": summary.get("prob_edge"),
        "message": message or summary_warning,
        "warnings": warning_list,
    }
    artifact = {
        "symbol": summary.get("symbol"),
        "updated_at": datetime.utcnow().isoformat(),
        "meta": meta,
        "signal_history": history_payload,
        "probability_backtest": backtest_payload,
    }
    if extras:
        for key, val in extras.items():
            artifact[key] = val
    return artifact


def _notify_slack(webhook: Optional[str], message: str):
    if not webhook:
        return
    try:
        resp = requests.post(webhook, json={"text": message}, timeout=5)
        resp.raise_for_status()
    except Exception as exc:
        logging.warning("Slack notification failed: %s", exc)


def _read_symbols_from_file(path: Path) -> List[str]:
    try:
        df = pd.read_csv(path)
        for col in ("Symbol", "symbol", "SYMBOL", "Ticker", "ticker"):
            if col in df.columns:
                values = df[col].dropna().astype(str).str.strip().tolist()
                if values:
                    return values
        first_col = df.columns[0]
        return df[first_col].dropna().astype(str).str.strip().tolist()
    except Exception:
        with open(path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]


def _iter_symbols(args: argparse.Namespace) -> Iterable[str]:
    if args.symbols:
        for sym in args.symbols:
            yield sym.upper().strip()
    elif args.symbols_file:
        for sym in _read_symbols_from_file(Path(args.symbols_file)):
            yield sym.upper().strip()
    elif args.randomize:
        universe = DEFAULT_SYMBOLS.copy()
        random.shuffle(universe)
        for sym in universe[: args.random_count]:
            yield sym
    else:
        for sym in DEFAULT_SYMBOLS:
            yield sym


def _build_sector_benchmarks(
    data_by_symbol: Dict[str, pd.DataFrame]
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
    benchmarks: Dict[str, pd.Series] = {}
    sector_indices: Dict[str, pd.Series] = {}
    for sector, members in SECTOR_GROUPS.items():
        closes = []
        for sym in members:
            df = data_by_symbol.get(sym)
            if df is None or df.empty or "Close" not in df.columns:
                continue
            closes.append(df["Close"].astype(float).rename(sym))
        if not closes:
            continue
        joined = pd.concat(closes, axis=1).sort_index()
        if joined.empty:
            continue
        bench = joined.mean(axis=1, skipna=True)
        sector_indices[sector] = bench
        for sym in members:
            if sym in data_by_symbol:
                benchmarks[sym] = bench
    return benchmarks, sector_indices


def _compute_sector_liquidity_stats(
    data_by_symbol: Dict[str, pd.DataFrame]
) -> Tuple[Dict[str, dict], dict]:
    """
    Computes dynamic sector thresholds but applies ABSOLUTE CAPS to prevent
    rejecting high-quality assets purely because they are in a high-volume sector.
    """
    def _collect_stats(df: pd.DataFrame) -> Optional[dict]:
        if df is None or df.empty or "Close" not in df.columns or "Volume" not in df.columns:
            return None
        closes = df["Close"].astype(float)
        volumes = df["Volume"].astype(float)
        if closes.empty or volumes.empty:
            return None
        median_volume = float(volumes.median()) if volumes.notna().any() else None
        liquidity = (closes * volumes).replace([np.inf, -np.inf], np.nan)
        median_liquidity = float(liquidity.median()) if liquidity.notna().any() else None
        zero_pct = float((volumes.fillna(0.0) <= 0.0).mean()) if len(volumes) else None
        return {
            "median_volume": median_volume,
            "median_liquidity": median_liquidity,
            "zero_volume_pct": zero_pct,
        }

    def _quantile(values: List[Optional[float]], pct: float, fallback: float) -> float:
        vals = [v for v in values if v is not None and not np.isnan(v)]
        if not vals:
            return fallback
        return float(np.percentile(vals, pct))

    global_samples = []
    sector_samples: Dict[str, List[dict]] = {sector: [] for sector in SECTOR_GROUPS}
    for sym, df in data_by_symbol.items():
        stats = _collect_stats(df)
        if stats is None:
            continue
        sector = SECTOR_LOOKUP.get(sym)
        if sector in sector_samples:
            sector_samples[sector].append(stats)
        global_samples.append(stats)

    # --- QUANT SAFEGUARDS (NEPSE SPECIFIC) ---
    # --- QUANT SAFEGUARDS (NEPSE SPECIFIC) ---
    ABSOLUTE_MIN_VOL = 100.0       # Lowered from 200
    ABSOLUTE_MIN_LIQ = 20_000.0    # Lowered from 50k
    
    # Ceiling: Relaxed significantly for NEPSE
    # OLD: 1_000_000.0 
    # NEW: 200_000.0 (2 Lakhs)
    ABSOLUTE_CAP_LIQ_REQ = 200_000.0 
    
    ABSOLUTE_MAX_ZERO_PCT = 0.50   # Allow more zero-volume days

    global_volume = _quantile([s["median_volume"] for s in global_samples], 25, 1000.0)
    global_liquidity = _quantile([s["median_liquidity"] for s in global_samples], 25, 300_000.0)
    global_zero = _quantile([s["zero_volume_pct"] for s in global_samples], 70, 0.35)
    
    default_thresholds = {
        "min_median_volume": max(global_volume, ABSOLUTE_MIN_VOL),
        "min_median_liquidity": max(global_liquidity, ABSOLUTE_MIN_LIQ),
        "max_zero_volume_pct": max(global_zero, ABSOLUTE_MAX_ZERO_PCT),
    }

    sector_thresholds: Dict[str, dict] = {}
    for sector, samples in sector_samples.items():
        if not samples:
            sector_thresholds[sector] = default_thresholds
            continue
            
        # Calculate dynamic sector stats
        vol_thresh = _quantile([s["median_volume"] for s in samples], 25, default_thresholds["min_median_volume"])
        liq_thresh = _quantile([s["median_liquidity"] for s in samples], 25, default_thresholds["min_median_liquidity"])
        zero_thresh = _quantile([s["zero_volume_pct"] for s in samples], 70, default_thresholds["max_zero_volume_pct"])
        
        # --- APPLY SAFETY CAPS (THE FIX) ---
        
        # 1. Volume: Dynamic, but capped at 2,000 to save high-priced stocks (SCB)
        final_vol = min(vol_thresh, 2000.0)  
        final_vol = max(final_vol, ABSOLUTE_MIN_VOL) 
        
        # 2. Liquidity: Dynamic, but capped at 1M to save Blue Chips (HBL, NTC)
        final_liq = min(liq_thresh, ABSOLUTE_CAP_LIQ_REQ)
        final_liq = max(final_liq, ABSOLUTE_MIN_LIQ)
        
        # 3. Zero Volume: Never be stricter than 45%
        final_zero = max(zero_thresh, ABSOLUTE_MAX_ZERO_PCT)

        sector_thresholds[sector] = {
            "min_median_volume": final_vol,
            "min_median_liquidity": final_liq,
            "max_zero_volume_pct": final_zero,
        }

    return sector_thresholds, default_thresholds


def _sector_relative_signal(
    prices: pd.Series,
    sector_series: Optional[pd.Series],
    lookback: int = 63,
) -> Optional[dict]:
    if sector_series is None or prices is None or prices.empty:
        return None
    sector_aligned = pd.Series(sector_series).astype(float).reindex(prices.index).ffill()
    if sector_aligned.isna().all():
        return None
    spread = prices.pct_change(lookback) - sector_aligned.pct_change(lookback)
    rolling_mean = spread.rolling(252).mean()
    rolling_std = spread.rolling(252).std().replace(0.0, np.nan)
    zscore = (spread - rolling_mean) / rolling_std
    zscore = zscore.replace([np.inf, -np.inf], np.nan).dropna()
    if zscore.empty:
        return None
    latest_z = float(zscore.iloc[-1])
    prob = float(np.clip(0.5 + np.tanh(latest_z) * 0.2, 0.0, 1.0))
    sizing_multiplier = float(np.clip(1 + latest_z / 4.0, 0.5, 1.5))
    confidence = float(min(1.0, abs(latest_z) / 2.0))
    return {
        "prob": prob,
        "zscore": latest_z,
        "sizing_multiplier": sizing_multiplier,
        "confidence": confidence,
    }


def _detect_probability_drift(history_df: Optional[pd.DataFrame], window: int = 120, threshold: float = 0.10) -> Optional[str]:
    if history_df is None or history_df.empty or "Probability" not in history_df.columns:
        return None
    rolling = history_df["Probability"].rolling(window).mean().dropna()
    if rolling.empty:
        return None
    recent = float(rolling.iloc[-1])
    baseline = float(history_df["Probability"].mean())
    if np.isnan(recent) or np.isnan(baseline):
        return None
    if abs(recent - baseline) >= threshold:
        return f"Probability drift alert: recent avg {recent:.2f} vs long-term {baseline:.2f}."
    return None


def _process_symbol(payload) -> dict:
    (
        symbol,
        df,
        sector_series,
        forex_series,
        world_bank_macro,
        imf_macro,
        sector_name,
        horizon,
        cv_folds,
        health_snapshot,
    ) = payload
    try:
        if df is None or df.empty or "Close" not in df.columns:
            raise ValueError("No price data.")
        df = df.sort_index()
        log_returns = compute_log_returns(df["Close"])
        if sector_series is not None and not sector_series.empty:
            sector_series = sector_series.reindex(df.index).ffill()
        macro_components = []
        if forex_series is not None:
            macro_components.append(
                pd.DataFrame({"Macro_FX_USD": forex_series.reindex(df.index).ffill()}, index=df.index)
            )
        if world_bank_macro is not None and not world_bank_macro.empty:
            wb_panel = build_world_bank_macro_panel(df.index, macro_df=world_bank_macro)
            if wb_panel is not None:
                macro_components.append(wb_panel)
        if imf_macro is not None and not imf_macro.empty:
            imf_panel = build_imf_macro_panel(df.index, macro_df=imf_macro)
            if imf_panel is not None:
                macro_components.append(imf_panel)
        macro_series = (
            pd.concat(macro_components, axis=1).sort_index() if macro_components else None
        )
        if macro_series is not None and not macro_series.empty:
            macro_series = macro_series.reindex(df.index).ffill().bfill()
        
        # Ensure factor windows don't exceed data length or become too short relative to horizon
        long_window = max(26, min(156, len(df) // 4))
        
        factors = build_factor_dataframe(
            price_df=df,
            log_returns=log_returns,
            long_window=long_window,
            horizon=horizon,
            macro_series=macro_series,
            sector_series=sector_series,
            sector_name=sector_name,
        )
        model_result = fit_signal_model(
            factors,
            horizon=horizon,
            cv_folds=cv_folds,
            sector_name=sector_name,
        )
        summary = {
            "symbol": symbol,
            "cv_score": model_result.accuracy_cv,
            "cv_std": model_result.accuracy_std,
            "probability": model_result.probability,
            "prob_edge": (
                model_result.cv_metrics.get("prob_edge")
                if model_result.cv_metrics
                else None
            ),
            "warning": "; ".join(model_result.warnings or []),
        }
        history_payload = _history_payload(model_result.history)
        backtest_payload = _simulate_probability_backtest(model_result.history, log_returns)

        # Increase slippage assumption for post-CV validation
        # Default is usually 30bps (0.3%). For better realism, we can optionally pass a higher value here.
        # We'll stick to default for now as the main fix was the horizon calc, but keep it in mind.
        validation_info = run_post_cv_validation(
            model_result.history,
            log_returns,
            df["Close"],
            df.get("Volume"),
        )
        validation_equity = validation_info.get("equity") if validation_info else None
        if validation_info and validation_info.get("metrics"):
            model_result.validation_metrics = validation_info["metrics"]
        if validation_info and not validation_info.get("passed", True):
            reason = validation_info.get("reason") or "Post-validation failed."
            model_result.disabled_reason = reason
            model_result.probability = 0.5
            model_result.warnings = (model_result.warnings or []) + [reason]

        sector_signal = _sector_relative_signal(df["Close"], sector_series)
        if sector_signal:
            component_probs = model_result.component_probs or {}
            component_probs["Sector Relative"] = sector_signal["prob"]
            model_result.component_probs = component_probs
            component_confidence = model_result.component_confidence or {}
            component_confidence["Sector Relative"] = sector_signal["confidence"]
            model_result.component_confidence = component_confidence
            model_result.position_multiplier = sector_signal["sizing_multiplier"]

        drift_alert = _detect_probability_drift(model_result.history)
        if drift_alert:
            model_result.warnings = (model_result.warnings or []) + [drift_alert]

        if "SectorRank" in factors:
            model_result.sector_rank_snapshot = factors["SectorRank"].tail(1).T.squeeze()

        extras = {
            "cv_metrics": _json_safe(model_result.cv_metrics),
            "coefficients": _series_payload(getattr(model_result, "coefficients", None)),
            "latest_factors": _series_payload(getattr(model_result, "latest_factors", None)),
            "feature_importance": _df_payload(getattr(model_result, "feature_importance_df", None)),
            "contributions": _series_payload(getattr(model_result, "contributions", None)),
            "component_probs": _json_safe(getattr(model_result, "component_probs", None)),
            "component_confidence": _json_safe(getattr(model_result, "component_confidence", None)),
            "symbol_health": health_snapshot,
            "validation_metrics": _json_safe(getattr(model_result, "validation_metrics", None)),
            "validation_equity": validation_equity,
            "disabled_reason": model_result.disabled_reason,
            "position_multiplier": model_result.position_multiplier,
            "cross_sectional_signal": _json_safe(sector_signal),
            "drift_alert": drift_alert,
            "sector_rank_snapshot": _series_payload(getattr(model_result, "sector_rank_snapshot", None)),
            "model_tier": model_result.model_tier,
            "trust_score": model_result.trust_score,
        }
        artifact = _build_symbol_artifact(
            summary,
            history_payload,
            backtest_payload,
            extras=extras,
            warnings=model_result.warnings,
            message=getattr(model_result, "message", None),
        )
        return {"summary": summary, "artifact": artifact}
    except Exception as exc:
        summary = {
            "symbol": symbol,
            "cv_score": None,
            "cv_std": None,
            "probability": None,
            "prob_edge": None,
            "warning": f"ERROR: {exc}",
        }
        artifact = _build_symbol_artifact(summary, None, None)
        return {"summary": summary, "artifact": artifact}


def _fetch_symbol_data(symbol: str):
    try:
        df = get_dynamic_data(symbol)
        status = "OK" if df is not None and not df.empty else "EMPTY"
        return symbol, df, status
    except Exception as exc:
        return symbol, None, f"Error fetch: {exc}"


def _run_batch(args):
    store = JobStore(namespace=args.job_name, root=Path(args.artifacts_dir).expanduser().resolve())
    log_file = store.path / "job.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Starting CV batch job name=%s run_id=%s", args.job_name, store.run_id)
    store.write_metadata({"model_version": MODEL_VERSION, **vars(args)}, name="params.json")
    status_payload = {
        "run_id": store.run_id,
        "artifact_path": str(store.path),
        "model_version": MODEL_VERSION,
    }

    cpu_count = os.cpu_count() or 1
    if args.workers <= 0:
        args.workers = cpu_count
    else:
        args.workers = max(1, min(args.workers, cpu_count))
    logging.info("Using %s worker processes out of %s CPU cores.", args.workers, cpu_count)

    symbols: List[str] = []
    for sym in _iter_symbols(args):
        if len(symbols) >= args.limit:
            break
        symbols.append(sym)

    if not symbols:
        msg = "No symbols to process. Provide inputs or increase --random-count/--limit."
        logging.error(msg)
        _notify_slack(args.slack_webhook, f"CV batch failed: {msg}")
        return 1, status_payload

    macro_fx_series = get_cached_forex_series("USD")
    world_bank_macro = load_world_bank_macro()
    imf_macro = load_imf_macro()

    logging.info("Fetching %s symbols for benchmarks...", len(symbols))
    fetch_results = {}
    fetch_workers = max(1, min(len(symbols), args.workers * 2))
    with ThreadPoolExecutor(max_workers=fetch_workers) as executor:
        futures = {executor.submit(_fetch_symbol_data, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym, df, status = future.result()
            fetch_results[sym] = (df, status)

    valid_data = {sym: df for sym, (df, status) in fetch_results.items() if status == "OK"}
    if not valid_data:
        logging.error("No valid datasets; aborting.")
        rows = [
            {
                "symbol": sym,
                "cv_score": None,
                "cv_std": None,
                "probability": None,
                "prob_edge": None,
                "warning": status,
            }
            for sym, (_, status) in fetch_results.items()
        ]
        df = pd.DataFrame(rows)
        df.to_csv("cv_batch_results.csv", index=False)
        summary_path = store.write_dataframe(df, name="summary.parquet")
        df.to_csv(store.path / "summary.csv", index=False)
        for record in rows:
            store.write_symbol_record(
                record.get("symbol", "UNKNOWN"),
                _build_symbol_artifact(record, None, None),
            )
        logging.info("Saved minimal summary to %s and cv_batch_results.csv", summary_path)
        _notify_slack(
            args.slack_webhook,
            f"CV batch finished with no valid datasets. Summary stored at {summary_path}",
        )
        return 1, status_payload

    aligned_data = {}
    for sym, df in valid_data.items():
        try:
            aligned = align_to_trading_calendar(df)
        except Exception as exc:
            logging.warning("Alignment failed for %s: %s", sym, exc)
            continue
        if aligned is not None and not aligned.empty:
            aligned_data[sym] = aligned

    sector_thresholds, default_thresholds = _compute_sector_liquidity_stats(aligned_data)

    normalized_data = {}
    health_payloads = {}
    health_failures = []
    for sym, df in aligned_data.items():
        sector = SECTOR_LOOKUP.get(sym)
        thresholds = sector_thresholds.get(sector, default_thresholds)
        try:
            health = compute_symbol_health(sym, df, **thresholds)
        except Exception as exc:
            health = SymbolHealth(
                symbol=sym,
                trading_days=0,
                missing_close_pct=1.0,
                flat_close_pct=1.0,
                median_volume=0.0,
                median_liquidity=0.0,
                zero_volume_pct=1.0,
                first_trade=None,
                healthy=False,
                reasons=[f"Health gate error: {exc}"],
                tier="none",
            )
        health_payloads[sym] = health.as_dict()
        if health.healthy:
            normalized_data[sym] = df
        else:
            health_failures.append((sym, health))

    if not normalized_data:
        rows = [
            {
                "symbol": sym,
                "cv_score": None,
                "cv_std": None,
                "probability": None,
                "prob_edge": None,
                "warning": "; ".join(health.reasons),
            }
            for sym, health in health_failures
        ]
        df = pd.DataFrame(rows)
        df.to_csv("cv_batch_results.csv", index=False)
        summary_path = store.write_dataframe(df, name="summary.parquet")
        df.to_csv(store.path / "summary.csv", index=False)
        for record in rows:
            store.write_symbol_record(
                record.get("symbol", "UNKNOWN"),
                _build_symbol_artifact(record, None, None),
            )
        logging.error("All symbols rejected by health gate. Summary stored at %s", summary_path)
        _notify_slack(
            args.slack_webhook,
            f"CV batch halted: all symbols rejected by health gate. Summary: {summary_path}",
        )
        return 1, status_payload

    model_data = {}
    model_failures = []
    for sym, df in normalized_data.items():
        sector = SECTOR_LOOKUP.get(sym)
        if sector in ("Commercial Banks", "Life Insurance", "Non-Life Insurance"):
            sigma_price = 0.8
            sigma_volume = 0.8
        elif sector in ("Hydropower", "Microfinance"):
            sigma_price = 0.5
            sigma_volume = 0.5
        else:
            sigma_price = 0.6
            sigma_volume = 0.6
        event_df = resample_event_bars(df, price_sigma=sigma_price, volume_sigma=sigma_volume)
        if event_df is None or event_df.empty or len(event_df) < max(120, args.horizon // 2):
            model_failures.append(sym)
            continue
        model_data[sym] = event_df

    if not model_data:
        logging.error("Weekly resample left no symbols; aborting.")
        return 1, status_payload

    sector_benchmarks, sector_indices = _build_sector_benchmarks(model_data)

    tasks = []
    for sym, df in model_data.items():
        # --- CRITICAL FIX: REMOVE THE '// 5' DIVISOR ---
        # OLD: horizon = max(1, resolve_symbol_horizon(sym, args.horizon) // 5)
        # NEW: Respect the actual horizon argument so we don't train on 12-day noise when we want 60-day trend.
        horizon = resolve_symbol_horizon(sym, args.horizon)
        
        tasks.append(
            (
                sym,
                df,
                sector_benchmarks.get(sym),
                macro_fx_series,
                world_bank_macro,
                imf_macro,
                SECTOR_LOOKUP.get(sym),
                horizon,
                args.cv_folds,
                health_payloads.get(sym),
            )
        )

    records = []
    artifacts = {}
    fetch_errors = [
        {"symbol": sym, "warning": status}
        for sym, (_, status) in fetch_results.items()
        if status != "OK"
    ]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_symbol, task): task[0] for task in tasks}
        for idx, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                result = future.result()
                summary = result.get("summary", {})
                artifact = result.get("artifact")
            except Exception as exc:
                summary = {
                    "symbol": symbol,
                    "cv_score": None,
                    "cv_std": None,
                    "probability": None,
                    "prob_edge": None,
                    "warning": f"FUTURE ERROR: {exc}",
                }
                artifact = _build_symbol_artifact(summary, None, None)
            logging.info("[%s/%s] Finished %s", idx, len(tasks), symbol)
            records.append(summary)
            artifacts[summary.get("symbol")] = artifact

    for err in fetch_errors:
        summary = {
            "symbol": err["symbol"],
            "cv_score": None,
            "cv_std": None,
            "probability": None,
            "prob_edge": None,
            "warning": err["warning"],
        }
        records.append(summary)
        artifacts[summary["symbol"]] = _build_symbol_artifact(summary, None, None)

    for sym in model_failures:
        summary = {
            "symbol": sym,
            "cv_score": None,
            "cv_std": None,
            "probability": None,
            "prob_edge": None,
            "warning": "Weekly resample insufficient for modeling.",
        }
        records.append(summary)
        artifacts[sym] = _build_symbol_artifact(summary, None, None)

    for sym, health in health_failures:
        summary = {
            "symbol": sym,
            "cv_score": None,
            "cv_std": None,
            "probability": None,
            "prob_edge": None,
            "warning": f"Health gate rejection: {'; '.join(health.reasons)}",
        }
        records.append(summary)
        artifacts[sym] = _build_symbol_artifact(summary, None, None)

    df = pd.DataFrame(records)
    if df.empty:
        logging.error("No symbols processed.")
        _notify_slack(args.slack_webhook, "CV batch produced no symbols.")
        return 1, status_payload
    df.to_csv("cv_batch_results.csv", index=False)
    summary_parquet = store.write_dataframe(df, name="summary.parquet")
    df.to_csv(store.path / "summary.csv", index=False)
    for symbol, artifact in artifacts.items():
        store.write_symbol_record(symbol or "UNKNOWN", artifact)
    logging.info("Saved CV batch summary to %s and cv_batch_results.csv", summary_parquet)
    _notify_slack(
        args.slack_webhook,
        f"CV batch complete: {len(df)} symbols processed. Summary: {summary_parquet}",
    )
    return 0, status_payload


def main():
    parser = argparse.ArgumentParser(
        description="Batch-check CV metrics for many NEPSE symbols (multiprocess, max CPU)."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="List of ticker symbols (space separated).",
    )
    parser.add_argument(
        "--symbols-file",
        help="CSV or newline-delimited file containing ticker symbols.",
    )
    parser.add_argument(
        "--random-count",
        type=int,
        default=25,
        help="When --randomize is set, randomly sample this many from the default universe (default: 25).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum number of symbols to process (default: 25).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=60,
        help="Triple barrier horizon used during factor modeling.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds for the institutional model.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of concurrent worker processes. Default 0 = use os.cpu_count().",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomly sample symbols from the default universe instead of taking the first N.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=os.environ.get("ARTIFACTS_DIR", "artifacts"),
        help="Directory where job artifacts (logs, per-symbol JSON) will be written.",
    )
    parser.add_argument(
        "--job-name",
        default="cv_batch",
        help="Logical job name used when creating artifact subdirectories.",
    )
    parser.add_argument(
        "--slack-webhook",
        default=None,
        help="Optional Slack webhook URL for notifications.",
    )
    args = parser.parse_args()

    queue = JobQueue(namespace=args.job_name)
    completed_at = datetime.utcnow().isoformat()
    try:
        with queue.job_lock():
            exit_code, status_payload = _run_batch(args)
    except RuntimeError as exc:
        logging.error("Job lock failed: %s", exc)
        return 1
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