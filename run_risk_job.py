import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from nepse_quant_pro.data_io import get_dynamic_data
from nepse_quant_pro.returns import compute_log_returns
from nepse_quant_pro.risk_engine import run_monte_carlo_paths
from nepse_quant_pro.store import JobStore
from nepse_quant_pro.job_queue import JobQueue


def _notify_slack(webhook: str, message: str):
    if not webhook:
        return
    try:
        resp = requests.post(webhook, json={"text": message}, timeout=5)
        resp.raise_for_status()
    except Exception as exc:
        logging.warning("Slack notification failed: %s", exc)


def _build_risk_artifact(symbol: str, paths: np.ndarray, returns: np.ndarray) -> dict:
    horizon = paths.shape[0] - 1
    percentiles = [5, 25, 50, 75, 95]
    fan_chart = []
    for day in range(horizon + 1):
        row = {"day": day}
        slice_vals = paths[day, :]
        for p in percentiles:
            row[f"p{p}"] = float(np.percentile(slice_vals, p))
        fan_chart.append(row)
    hist, bins = np.histogram(returns, bins=40)
    var95 = float(np.percentile(returns, 5))
    var99 = float(np.percentile(returns, 1))
    es95 = float(returns[returns <= var95].mean())
    es99 = float(returns[returns <= var99].mean())
    return {
        "symbol": symbol,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "distribution": {
            "bins": bins.tolist(),
            "counts": hist.tolist(),
        },
        "expected_return": float(np.mean(returns)),
        "tails": {
            "var_95": var95,
            "var_99": var99,
            "es_95": es95,
            "es_99": es99,
        },
        "fan_chart": fan_chart,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Monte Carlo risk job.")
    parser.add_argument("--symbols", nargs="+", help="Symbols to process.")
    parser.add_argument("--symbols-file", help="CSV or txt file with symbols.")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--horizon-days", type=int, default=240)
    parser.add_argument("--sims", type=int, default=100_000)
    parser.add_argument(
        "--engine",
        choices=["bootstrap", "gbm"],
        default="bootstrap",
        help="Return generation engine: block bootstrap (default) or GBM",
    )
    parser.add_argument("--artifacts-dir", default=os.environ.get("ARTIFACTS_DIR", "artifacts"))
    parser.add_argument("--job-name", default="risk_job")
    parser.add_argument("--slack-webhook", default=None)
    return parser.parse_args()


def resolve_symbols(args) -> list:
    if args.symbols:
        picks = [s.upper().strip() for s in args.symbols]
    elif args.symbols_file:
        path = Path(args.symbols_file)
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            df = pd.read_csv(path)
            col = df.columns[0]
            picks = df[col].dropna().astype(str).str.strip().tolist()
        except Exception:
            with open(path, "r", encoding="utf-8") as handle:
                picks = [line.strip() for line in handle if line.strip()]
    else:
        picks = []
    if args.limit and len(picks) > args.limit:
        picks = picks[: args.limit]
    return picks


def _execute_risk(args):
    store = JobStore(namespace=args.job_name, root=Path(args.artifacts_dir).expanduser().resolve())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(store.path / "job.log"), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Risk job started run_id=%s", store.run_id)
    store.write_metadata(vars(args), name="params.json")
    status_payload = {
        "run_id": store.run_id,
        "artifact_path": str(store.path),
    }
    symbols = resolve_symbols(args)
    if not symbols:
        logging.error("No symbols supplied.")
        _notify_slack(args.slack_webhook, "Risk job aborted: no symbols.")
        return 1, status_payload
    summary_rows = []
    
    engine_label = "BlockBootstrap" if args.engine == "bootstrap" else "Growth (GBM)"

    for idx, sym in enumerate(symbols, start=1):
        try:
            df = get_dynamic_data(sym)
            if df is None or df.empty or "Close" not in df.columns:
                raise ValueError("No price data")
            df = df.sort_index()
            
            # Compute Returns
            log_ret = compute_log_returns(df["Close"])
            
            # --- [UPGRADE] NEPSE CIRCUIT BREAKER LOGIC ---
            # Cap daily returns at +/- 10% to prevent unrealistic black swans
            # This makes the Block Bootstrap simulation respect market structure.
            log_ret = log_ret.clip(lower=-0.10, upper=0.10)
            
            # Parameters for backup (though Bootstrap ignores mu/sigma, we calculate for logging)
            mu_daily = float(log_ret.mean())
            sigma_daily = float(log_ret.std())
            annual_mu = mu_daily * 252
            annual_sigma = sigma_daily * np.sqrt(252)
            
            # --- [UPGRADE] BLOCK BOOTSTRAP ENGINE ---
            # Using real historical blocks instead of theoretical normal distribution
            paths = run_monte_carlo_paths(
                current_price=float(df["Close"].iloc[-1]),
                annual_mu=annual_mu,
                annual_sigma=annual_sigma,
                days=args.horizon_days,
                sims=args.sims,
                return_generation_method=engine_label,
                hist_returns=log_ret if args.engine == "bootstrap" else None,
            )

            returns = paths[-1, :] / paths[0, 0] - 1.0
            artifact = _build_risk_artifact(sym, paths, returns)
            artifact["engine"] = engine_label
            store.write_symbol_record(sym, artifact)
            
            summary_rows.append(
                {
                    "symbol": sym,
                    "engine": engine_label,
                    "var_95": artifact["tails"]["var_95"],
                    "var_99": artifact["tails"]["var_99"],
                    "es_95": artifact["tails"]["es_95"],
                    "es_99": artifact["tails"]["es_99"],
                }
            )
            logging.info("[%s/%s] Completed %s (%s)", idx, len(symbols), sym, engine_label)
            
        except Exception as exc:
            logging.warning("[%s/%s] Failed %s: %s", idx, len(symbols), sym, exc)
            summary_rows.append(
                {
                    "symbol": sym,
                    "engine": engine_label,
                    "var_95": None,
                    "var_99": None,
                    "es_95": None,
                    "es_99": None,
                    "warning": str(exc),
                }
            )
            store.write_symbol_record(
                sym,
                {"symbol": sym, "error": str(exc), "updated_at": datetime.now(timezone.utc).isoformat()},
            )
            
    summary = pd.DataFrame(summary_rows)
    store.write_dataframe(summary, name="summary.parquet")
    summary.to_csv(store.path / "summary.csv", index=False)
    _notify_slack(args.slack_webhook, f"Risk job complete: {len(symbols)} symbols, run_id={store.run_id}")
    return 0, status_payload


def main():
    args = parse_args()
    from datetime import datetime, timezone

    queue = JobQueue(namespace=args.job_name)
    try:
        with queue.job_lock():
            exit_code, status_payload = _execute_risk(args)
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
