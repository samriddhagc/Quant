import json
import os
import math
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from nepse_quant_pro.config import (
    PERIODS_PER_YEAR,
    NO_TRADE_UTILITY_THRESHOLD,
    MAX_DRAWDOWN_THRESHOLD,
)
from nepse_quant_pro.data_io import (
    load_csv,
    extract_price_matrix,
    load_macro_series,
    get_dynamic_data,
)

from nepse_quant_pro.returns import (
    compute_features,
    compute_log_returns,
    summarize_returns,
    estimate_mu_sigma,
    compute_ewma_vol,
    fit_garch_vol,
    compute_autocorrelation,
    calculate_atr,
    calculate_mean_atr,
    get_daily_volatility_class,
    check_data_liquidity,
    get_asset_class,
)
from nepse_quant_pro.regime import detect_regime, build_regime_chart
from nepse_quant_pro.factors import (
    build_factor_contribution_chart,
    build_signal_history_chart,
    calculate_performance_metrics,
)
from nepse_quant_pro.artifacts import (
    ArtifactSelector,
    artifact_to_factor_model,
    load_run_metadata,
)
from nepse_quant_pro.sectors import SECTOR_LOOKUP
from nepse_quant_pro.pca_factors import run_pca
from nepse_quant_pro.beta_hedge import compute_beta, build_hedged_returns, estimate_capm_er
from nepse_quant_pro.risk_engine import (
    run_monte_carlo_paths,
    compute_var_cvar,
    run_canonical_engine,
    decision_engine,
    get_dynamic_trading_params,
    james_stein_drift,
)
from nepse_quant_pro.downside import (
    compute_downside_risk,
    compute_downside_stats,
    build_drawdown_figure,
    compute_drawdowns,
)
from nepse_quant_pro.portfolio import (
    estimate_mean_cov,
    compute_portfolio_stats,
    min_variance_portfolio,
    tangency_portfolio,
    get_hrp_weights,
    black_litterman_posterior,
)
from nepse_quant_pro.visuals import (
    build_trend_figure,
    build_returns_histogram,
    build_qq_plot,
    build_mc_paths_figure,
    build_volatility_figure,
)
from nepse_quant_pro.job_queue import JobQueue

JOB_STATUS_ROOT = Path(os.environ.get("JOB_ROOT", "jobs")).expanduser().resolve()
RISK_ENGINE_OPTIONS = OrderedDict(
    [
        (
            "GBM (Parametric)",
            {
                "namespace": "risk_gbm",
                "return_method": "GBM",
                "engine_arg": "gbm",
            },
        ),
        (
            "Block Bootstrap (Historical)",
            {
                "namespace": "risk_bootstrap",
                "return_method": "BlockBootstrap",
                "engine_arg": "bootstrap",
            },
        ),
    ]
)
DEFAULT_RISK_ENGINE_LABEL = next(iter(RISK_ENGINE_OPTIONS))

st.set_page_config(page_title="SA Risk Lab", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    .stMetric { background-color: #111827 !important; border-radius: 0.5rem; padding: 0.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _df_to_payload(df: pd.DataFrame) -> str:
    return df.to_json(date_format="iso", orient="split")


def _series_to_payload(series: pd.Series) -> str:
    return series.to_json(date_format="iso", orient="split")


def _payload_to_df(payload: str) -> pd.DataFrame:
    return pd.read_json(payload, orient="split")


def _payload_to_series(payload: str) -> pd.Series:
    return pd.read_json(payload, orient="split", typ="series")


def _latest_job_event(namespace: str, status: str = "completed") -> Optional[dict]:
    folder = JOB_STATUS_ROOT / namespace / status
    if not folder.exists():
        return None
    files = sorted(folder.glob("job_*.json"))
    if not files:
        return None
    latest = files[-1]
    try:
        with open(latest, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        data = {}
    data.setdefault("run_id", latest.stem.replace("job_", ""))
    data.setdefault("path", str(latest))
    return data


def _render_job_status(container, namespace: str, label: str):
    status = _latest_job_event(namespace)
    if status:
        run_id = status.get("artifact_run_id") or status.get("run_id")
        completed = status.get("completed_at") or status.get("timestamp")
        container.caption(f"{label}: last run {completed} (run {run_id})")
    else:
        container.caption(f"{label}: no completed jobs yet.")


def _submit_job_request(namespace: str, payload: dict):
    queue = JobQueue(namespace=namespace)
    queue.submit_request(payload)


def _format_run_metadata(label: str, metadata: Optional[dict], keys: Optional[List[str]] = None) -> str:
    if not metadata:
        return f"{label}: metadata unavailable."
    ordered_keys = keys or []
    parts = []
    for key in ordered_keys:
        if key in metadata:
            parts.append(f"{key}={metadata[key]}")
    if not parts:
        for key, value in metadata.items():
            parts.append(f"{key}={value}")
            if len(parts) >= 4:
                break
    return f"{label}: " + ", ".join(parts)


def _job_control_panel(default_symbol: Optional[str], all_symbols: Optional[List[str]] = None):
    expander = st.sidebar.expander("Job Queue Controls", expanded=False)
    with expander:
        _render_job_status(expander, "cv_batch", "CV Batch")
        with st.form("cv_batch_form"):
            st.write("Request a new CV batch run.")
            symbol_options = all_symbols or []
            default_selection = (
                [default_symbol] if default_symbol and default_symbol in symbol_options else []
            )
            cv_selected = st.multiselect(
                "Select symbols (from loaded data)",
                options=symbol_options,
                default=default_selection,
                key="cv_batch_symbol_select",
            )
            cv_manual = st.text_input(
                "Additional symbols (comma separated)",
                value="",
                key="cv_batch_symbols_manual",
                help="Optional comma-separated tickers if not in the dropdown.",
            )
            cv_limit = st.number_input(
                "Max symbols",
                min_value=1,
                max_value=500,
                value=25,
                step=1,
                key="cv_batch_limit",
            )
            submitted = st.form_submit_button("Queue CV Batch Job")
            if submitted:
                manual_list = [
                    sym.strip().upper()
                    for sym in cv_manual.replace("\n", ",").split(",")
                    if sym.strip()
                ]
                symbols = [
                    sym.upper()
                    for sym in cv_selected
                ]
                if manual_list:
                    symbols.extend(manual_list)
                payload = {
                    "symbols": symbols,
                    "limit": cv_limit,
                    "requested_at": datetime.now(timezone.utc).isoformat(),
                    "requested_via": "streamlit",
                }
                _submit_job_request("cv_batch", payload)
                st.success("CV batch job request submitted.")

        _render_job_status(expander, "bulk_bt", "Bulk Backtest")
        with st.form("bulk_bt_form"):
            st.write("Request a bulk backtest run.")
            bt_selected = st.multiselect(
                "Select symbols (from loaded data)",
                options=symbol_options,
                default=default_selection,
                key="bulk_bt_symbol_select",
            )
            bt_manual = st.text_input(
                "Additional symbols (comma separated)",
                value="",
                key="bulk_bt_symbols_manual",
            )
            bt_limit = st.number_input(
                "Max symbols",
                min_value=1,
                max_value=300,
                value=25,
                step=1,
                key="bulk_bt_limit",
            )
            submitted_bt = st.form_submit_button("Queue Bulk Backtest Job")
            if submitted_bt:
                manual_list = [
                    sym.strip().upper()
                    for sym in bt_manual.replace("\n", ",").split(",")
                    if sym.strip()
                ]
                symbols = [sym.upper() for sym in bt_selected]
                if manual_list:
                    symbols.extend(manual_list)
                payload = {
                    "symbols": symbols,
                    "limit": bt_limit,
                    "requested_at": datetime.now(timezone.utc).isoformat(),
                    "requested_via": "streamlit",
                }
                _submit_job_request("bulk_bt", payload)
                st.success("Bulk backtest job request submitted.")

        for label, engine_meta in RISK_ENGINE_OPTIONS.items():
            _render_job_status(expander, engine_meta["namespace"], f"Risk Engine ({label})")
        with st.form("risk_job_form"):
            st.write("Request a Monte Carlo risk run.")
            risk_selected = st.multiselect(
                "Select symbols (from loaded data)",
                options=symbol_options,
                default=default_selection,
                key="risk_symbol_select",
            )
            risk_manual = st.text_input(
                "Additional symbols (comma separated)",
                value="",
                key="risk_job_symbols_manual",
            )
            engine_selector = st.selectbox(
                "Risk engine",
                options=list(RISK_ENGINE_OPTIONS.keys()),
                index=list(RISK_ENGINE_OPTIONS.keys()).index(
                    st.session_state.get("risk_engine_label", DEFAULT_RISK_ENGINE_LABEL)
                ),
                key="risk_job_engine_choice",
            )
            horizon = st.number_input(
                "Horizon (days)",
                min_value=30,
                max_value=720,
                value=240,
                step=10,
                key="risk_job_horizon",
            )
            sims = st.number_input(
                "Simulations",
                min_value=500,
                max_value=500000,
                value=100000,
                step=1000,
                key="risk_job_sims",
            )
            submitted_risk = st.form_submit_button("Queue Risk Job")
            if submitted_risk:
                manual_list = [
                    sym.strip().upper()
                    for sym in risk_manual.replace("\n", ",").split(",")
                    if sym.strip()
                ]
                symbols = [sym.upper() for sym in risk_selected]
                if manual_list:
                    symbols.extend(manual_list)
                engine_meta = RISK_ENGINE_OPTIONS[engine_selector]
                payload = {
                    "symbols": symbols,
                    "horizon_days": int(horizon),
                    "sims": int(sims),
                    "engine": engine_meta["engine_arg"],
                    "requested_at": datetime.now(timezone.utc).isoformat(),
                    "requested_via": "streamlit",
                }
                _submit_job_request(engine_meta["namespace"], payload)
                st.success(f"Risk job request submitted for {engine_selector}.")

def _format_percent(value: Optional[float], digits: int = 2, already_percent: bool = True) -> str:
    if value is None:
        return "N/A"
    scaled = value if already_percent else value * 100.0
    return f"{scaled:.{digits}f}%"


def _build_distribution_chart(distribution: Optional[dict]) -> Optional[go.Figure]:
    if not distribution:
        return None
    bins = distribution.get("bins")
    counts = distribution.get("counts")
    if not bins or not counts or len(bins) != len(counts) + 1:
        return None
    centers = [(bins[i] + bins[i + 1]) / 2.0 for i in range(len(counts))]
    widths = [bins[i + 1] - bins[i] for i in range(len(counts))]
    fig = go.Figure(
        data=[
            go.Bar(
                x=centers,
                y=counts,
                width=widths,
                marker_color="#636EFA",
                opacity=0.7,
            )
        ]
    )
    fig.update_layout(
        title="Terminal Return Distribution",
        xaxis_title="Return",
        yaxis_title="Frequency",
        template="plotly_dark",
    )
    return fig


def _build_fan_chart(fan_chart: Optional[List[dict]]) -> Optional[go.Figure]:
    if not fan_chart:
        return None
    fan_df = pd.DataFrame(fan_chart)
    if fan_df.empty or "day" not in fan_df.columns:
        return None
    x = fan_df["day"]
    fig = go.Figure()
    if {"p95", "p5"}.issubset(fan_df.columns):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=fan_df["p95"],
                line=dict(color="rgba(99,110,250,0.4)", width=0),
                name="p95",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=fan_df["p5"],
                line=dict(color="rgba(99,110,250,0.0)"),
                fill="tonexty",
                fillcolor="rgba(99,110,250,0.2)",
                name="p5",
            )
        )
    if {"p75", "p25"}.issubset(fan_df.columns):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=fan_df["p75"],
                line=dict(color="rgba(239,85,59,0.4)", width=0),
                name="p75",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=fan_df["p25"],
                line=dict(color="rgba(239,85,59,0.0)"),
                fill="tonexty",
                fillcolor="rgba(239,85,59,0.2)",
                name="p25",
            )
        )
    if "p50" in fan_df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=fan_df["p50"],
                line=dict(color="#00CC96", width=2),
                name="Median",
            )
        )
    fig.update_layout(
        title="Monte Carlo Fan Chart",
        xaxis_title="Day",
        yaxis_title="Price",
        template="plotly_dark",
    )
    return fig


@st.cache_data(show_spinner=False)
def cached_canonical_engine(
    current_price: float,
    sigma_daily: float,
    er_recent_annual: float,
    er_capm_annual: float,
    shrinkage_factor: float,
    capm_weight: float,
    rf_annual: float,
    horizon_days: int,
    sims: int,
    es_conf_level: float,
    distribution: str,
    student_df: int,
    return_generation_method: str,
    rho: float,
    garch_payload: Optional[str],
    kappa: float,
    theta: float,
    lambda_jump: float,
    jump_mean: float,
    jump_std: float,
    hist_returns_payload: Optional[str],
):
    garch_vol = _payload_to_series(garch_payload) if garch_payload is not None else None
    hist_returns = (
        _payload_to_series(hist_returns_payload) if hist_returns_payload is not None else None
    )
    return run_canonical_engine(
        current_price=current_price,
        sigma_daily=sigma_daily,
        er_annual=er_recent_annual,
        er_recent_annual=er_recent_annual,
        er_capm_annual=er_capm_annual,
        shrinkage_factor=shrinkage_factor,
        capm_weight=capm_weight,
        rf_annual=rf_annual,
        horizon_days=horizon_days,
        sims=sims,
        es_conf_level=es_conf_level,
        distribution=distribution,
        student_df=student_df,
        return_generation_method=return_generation_method,
        rho=rho,
        garch_vol=garch_vol,
        kappa=kappa,
        theta=theta,
        lambda_jump=lambda_jump,
        jump_mean=jump_mean,
        jump_std=jump_std,
        hist_returns=hist_returns,
    )

def main():
    st.title("SA Risk Lab")
    st.markdown("### Quantitative Risk Analysis (Institutional-Grade)")

    # Sidebar ----------------------------------------------------------------
    st.sidebar.header("Data & Inputs")

    # --- DATA SOURCE SELECTION ---
    data_source = st.sidebar.radio("Data Mode", ["Live Database", "Upload CSV"], index=0)
    raw_df = None
    uploaded_file = None

    # Macro file (available in both modes)
    macro_file = st.sidebar.file_uploader("Optional Macro Proxy CSV", type=["csv"])
    st.sidebar.markdown("If no file is uploaded, the app uses `nepse_stock.csv` locally.")

    # Initialize session state for symbol if not present
    if "active_symbol" not in st.session_state:
        st.session_state["active_symbol"] = "SHIVM"

    if data_source == "Live Database":
        default_symbol = st.session_state["active_symbol"]
        try:
            top_picks = pd.read_csv("daily_top_picks.csv")
            if not top_picks.empty and "Symbol" in top_picks.columns:
                st.sidebar.markdown("---")
                st.sidebar.write("🏆 **Scanner Top Picks**")
                selected_pick = st.sidebar.selectbox(
                    "Load from Scanner", 
                    ["Select..."] + top_picks["Symbol"].astype(str).tolist(),
                    key="scanner_select",
                )
                if selected_pick != "Select...":
                    default_symbol = selected_pick
        except Exception:
            pass

        symbol_input = st.sidebar.text_input(
            "Enter Symbol", value=default_symbol, help="e.g. NABIL, NICA"
        ).upper()

        fetch_clicked = st.sidebar.button("Fetch Data", type="primary")

        if fetch_clicked:
            st.session_state["active_symbol"] = symbol_input

        if 'selected_pick' in locals() and selected_pick != "Select..." and selected_pick != st.session_state["active_symbol"]:
            st.session_state["active_symbol"] = selected_pick

        target_symbol = st.session_state["active_symbol"]

        if target_symbol:
            try:
                raw_df_live = get_dynamic_data(target_symbol)
                if raw_df_live is not None and not raw_df_live.empty:
                    st.sidebar.success(f"Ready: {target_symbol} ({len(raw_df_live)} rows)")
                    raw_df = raw_df_live.reset_index()
                else:
                    st.sidebar.error(f"No data found for {target_symbol}")
            except Exception as e:
                st.sidebar.error(f"Error loading {target_symbol}: {e}")

    else:
        uploaded_file = st.sidebar.file_uploader("Upload NEPSE CSV (Date, Close)", type=["csv"])
        if uploaded_file is not None:
            raw_df = load_csv(uploaded_file)
        else:
            uploaded_file = None
            st.sidebar.info("Please upload a file or switch to Live Database mode.")

    st.sidebar.markdown("---")
    short_window = st.sidebar.number_input("Short Window (SMA)", min_value=5, max_value=200, value=50, step=1)
    long_window = st.sidebar.number_input("Long Window (SMA)", min_value=20, max_value=400, value=200, step=5)

    st.sidebar.markdown("---")
    mc_sims = st.sidebar.number_input(
        "Monte Carlo Simulations", min_value=500, max_value=500000, value=2000, step=500
    )
    mc_horizon = st.sidebar.number_input(
        "Time Horizon (Days)", min_value=30, max_value=1260, value=252, step=10
    )

    # --- MONTE CARLO ENGINE (PRODUCTION SOURCED) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Monte Carlo Engine")
    engine_labels = list(RISK_ENGINE_OPTIONS.keys())
    default_engine = st.session_state.get("risk_engine_label", DEFAULT_RISK_ENGINE_LABEL)
    if default_engine not in engine_labels:
        default_engine = DEFAULT_RISK_ENGINE_LABEL
    mc_model_type = st.sidebar.selectbox(
        "Model Process",
        options=engine_labels,
        index=engine_labels.index(default_engine),
    )
    st.session_state["risk_engine_label"] = mc_model_type
    selected_risk_engine = RISK_ENGINE_OPTIONS[mc_model_type]

    # Legacy parameters retained for compatibility with run_canonical_engine signature
    distribution_choice = "Normal"
    student_df = 8
    kappa = 0.0
    theta_input = 0.0
    lambda_jump = 0.0
    jump_mean = 0.0
    jump_std = 0.0

    st.sidebar.markdown("---")
    regime_method = st.sidebar.selectbox("Regime Detection", ["Gaussian Mixture", "Rolling Volatility"])

    st.sidebar.markdown("---")
    failure_threshold_pct = st.sidebar.slider(
        "Failure Threshold (%)", min_value=-50, max_value=-5, value=-20, step=1
    )
    custom_shock_pct = st.sidebar.slider(
        "Custom Stress Shock (%)", min_value=-40, max_value=40, value=-10, step=1
    )
    custom_vol_multiplier = st.sidebar.slider(
        "Custom Stress Vol Multiplier", min_value=0.5, max_value=3.0, value=1.5, step=0.1
    )

    st.sidebar.markdown("---")
    risk_aversion_lambda = st.sidebar.slider(
        "Risk Aversion (λ)", min_value=0.25, max_value=5.0, value=0.5, step=0.1
    )

    # --- UPDATED ER BLENDING INPUTS ---
    st.sidebar.markdown("**Expected Return Blending**")

    # James-Stein robust shrinkage
    shrinkage_mode = st.sidebar.checkbox(
        "Use James-Stein Shrinkage?",
        value=True,
        help=(
            "If enabled, the recent historical drift is replaced by a James-Stein "
            "robust estimate before blending with CAPM."
        ),
    )

    shrinkage_factor = st.sidebar.slider(
        "Statistical Shrinkage (History → Rf)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="0 = Pure History, 1 = Pure Risk-Free",
    )
    capm_weight = st.sidebar.slider(
        "Model Weight (History vs CAPM)",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="0 = History/Shrinkage only, 1 = CAPM only",
    )
    rf_annual = st.sidebar.slider(
        "Annual Risk-Free Rate", min_value=0.0, max_value=0.1, value=0.05, step=0.005
    )
    st.sidebar.caption(
        "Production view uses artifact run parameters. Adjust settings via job queue or use the Research Sandbox for ad-hoc what-if analysis."
    )

    # [SNIPER MODE] Default raised to 65%
    prob_threshold = (
        st.sidebar.slider("Probability Threshold (%)", min_value=40, max_value=90, value=65, step=1) / 100
    )
    es_conf_level = (
        st.sidebar.slider("ES Confidence Level (%)", min_value=80, max_value=99, value=90, step=1) / 100
    )

    with st.sidebar.expander("Diagnostics & Thresholds", expanded=False):
        factor_cv_folds = st.number_input(
            "Factor CV Folds", min_value=2, max_value=10, value=5, step=1
        )
        mdd_threshold = (
            st.slider(
                "Max Drawdown Threshold (%)",
                min_value=-80,
                max_value=-5,
                value=int(MAX_DRAWDOWN_THRESHOLD * 100),
                step=1,
            )
            / 100
        )
        transaction_cost_pct = st.slider(
            "Transaction Cost (round-trip %)", min_value=0.05, max_value=1.0, value=0.10, step=0.05
        )
        transaction_cost = transaction_cost_pct / 100

        # --- SLIPPAGE & LIQUIDITY ---
        st.markdown("**Realism Settings**")
        slippage_pct_input = st.slider(
            "Slippage per Trade (%)", min_value=0.0, max_value=4.0, value=0.5, step=0.1
        )
        slippage_decimal = slippage_pct_input / 100.0
        enable_liquidity_filter = st.checkbox("Enable Liquidity Filter (Vol > 100)", value=True)

    if "tsl_multiplier_slider" not in st.session_state:
        st.session_state["tsl_multiplier_slider"] = 3.0
    if "asset_type_code" not in st.session_state:
        st.session_state["asset_type_code"] = "emerging_stock"
    if "regime_window" not in st.session_state:
        st.session_state["regime_window"] = 200

    trailing_stop_mult = st.session_state["tsl_multiplier_slider"]
    asset_type_code = st.session_state["asset_type_code"]
    regime_window = st.session_state["regime_window"]

    # Data loading ------------------------------------------------------------
    if raw_df is None:
        st.info("👈 Select a stock or upload data to begin.")
        return

    multi_prices_df: Optional[pd.DataFrame] = None
    analysis_symbol_prefill = None
    if data_source == "Live Database":
        active_symbol = st.session_state.get("active_symbol")
        if not active_symbol:
            st.error("Active symbol missing for live data.")
            return
        live_df = raw_df.copy()
        if "Date" in live_df.columns:
            live_df["Date"] = pd.to_datetime(live_df["Date"], errors="coerce")
            live_df = live_df.dropna(subset=["Date"]).set_index("Date")
        close_col = None
        for col in live_df.columns:
            if col.lower() == "close":
                close_col = col
                break
        if close_col is None:
            st.error("Critical: 'Close' column missing in live data.")
            return
        series = pd.to_numeric(live_df[close_col], errors="coerce").dropna()
        if series.empty:
            st.error("Live data close series is empty.")
            return
        multi_prices_df = pd.DataFrame({active_symbol: series})
        analysis_symbol_prefill = active_symbol
    else:
        try:
            multi_prices_df = extract_price_matrix(raw_df)
        except Exception as exc:
            st.error(f"Error preparing price matrix: {exc}")
            return

    available_tickers = list(multi_prices_df.columns)
    if not available_tickers:
        st.error("No valid asset columns found in the dataset.")
        return

    st.sidebar.markdown("---")
    primary_asset = st.sidebar.selectbox(
        "Primary asset (single-asset analysis)", options=available_tickers
    )
    if analysis_symbol_prefill is None:
        analysis_symbol_prefill = primary_asset
    job_symbol_options = sorted(SECTOR_LOOKUP.keys())
    _job_control_panel(analysis_symbol_prefill, job_symbol_options)
    default_multi_selection = (
        available_tickers[:3] if len(available_tickers) >= 3 else available_tickers
    )
    selected_tickers = st.sidebar.multiselect(
        "Select assets for multi-asset analysis",
        options=available_tickers,
        default=default_multi_selection,
    )
    if not selected_tickers:
        selected_tickers = [primary_asset]

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("Run Analysis")

    # Persist last run inputs
    if "stored_inputs" not in st.session_state:
        st.session_state["stored_inputs"] = None

    candidate_inputs = {
        "raw_df": raw_df,
        "macro_series": load_macro_series(macro_file),
        "short_window": short_window,
        "long_window": long_window,
        "mc_sims": mc_sims,
        "mc_horizon": mc_horizon,
        "mc_model_type": mc_model_type,
        "distribution_choice": distribution_choice,
        "student_df": student_df,
        "regime_method": regime_method,
        "failure_threshold_pct": failure_threshold_pct,
        "custom_shock_pct": custom_shock_pct,
        "custom_vol_multiplier": custom_vol_multiplier,
        "risk_aversion_lambda": risk_aversion_lambda,
        "shrinkage_factor": shrinkage_factor,
        "capm_weight": capm_weight,
        "rf_annual": rf_annual,
        "prob_threshold": prob_threshold,
        "es_conf_level": es_conf_level,
        "factor_cv_folds": factor_cv_folds,
        "mdd_threshold": mdd_threshold,
        "transaction_cost": transaction_cost,
        "trailing_stop_mult": trailing_stop_mult,
        "asset_type_code": asset_type_code,
        "regime_window": regime_window,
        "primary_asset": primary_asset,
        "analysis_symbol": analysis_symbol_prefill,
        "selected_tickers": selected_tickers,
        "multi_prices_df": multi_prices_df,
        "kappa": kappa,
        "theta_input": theta_input,
        "lambda_jump": lambda_jump,
        "jump_mean": jump_mean,
        "jump_std": jump_std,
        # Store new settings
        "slippage_pct_input": slippage_pct_input,
        "enable_liquidity_filter": enable_liquidity_filter,
        "shrinkage_mode": shrinkage_mode,
    }

    if run_button:
        st.session_state["stored_inputs"] = candidate_inputs

    stored = st.session_state.get("stored_inputs")
    if stored is None:
        st.info("Adjust parameters in the sidebar and click **Run Analysis** to start.")
        return

    # Use stored snapshot for all downstream calculations
    macro_series = stored["macro_series"]
    short_window = stored["short_window"]
    long_window = stored["long_window"]
    mc_sims = stored["mc_sims"]
    mc_horizon = stored["mc_horizon"]
    mc_model_type = stored["mc_model_type"]
    distribution_choice = stored["distribution_choice"]
    student_df = stored["student_df"]
    regime_method = stored["regime_method"]
    failure_threshold_pct = stored["failure_threshold_pct"]
    custom_shock_pct = stored["custom_shock_pct"]
    custom_vol_multiplier = stored["custom_vol_multiplier"]
    risk_aversion_lambda = stored["risk_aversion_lambda"]
    shrinkage_factor = stored["shrinkage_factor"]
    capm_weight = stored["capm_weight"]
    rf_annual = stored["rf_annual"]
    prob_threshold = stored["prob_threshold"]
    es_conf_level = stored["es_conf_level"]
    factor_cv_folds = stored["factor_cv_folds"]
    mdd_threshold = stored["mdd_threshold"]
    transaction_cost = stored.get("transaction_cost", 0.001)
    trailing_stop_mult = stored.get("trailing_stop_mult", trailing_stop_mult)
    asset_type_code = stored.get("asset_type_code", asset_type_code)
    regime_window = stored.get("regime_window", regime_window)
    primary_asset = stored["primary_asset"]
    analysis_symbol = stored.get("analysis_symbol", primary_asset)
    selected_tickers = stored["selected_tickers"]
    multi_prices_df = stored["multi_prices_df"]

    # Retrieve new settings
    slippage_pct_input = stored.get("slippage_pct_input", 0.5)
    slippage_decimal = slippage_pct_input / 100.0
    enable_liquidity_filter = stored.get("enable_liquidity_filter", True)
    shrinkage_mode = stored.get("shrinkage_mode", False)

    # Retrieve advanced params
    kappa = stored.get("kappa", 0.0)
    theta_input = stored.get("theta_input", 0.0)
    lambda_jump = stored.get("lambda_jump", 0.0)
    jump_mean = stored.get("jump_mean", 0.0)
    jump_std = stored.get("jump_std", 0.0)

    price_df = multi_prices_df[[primary_asset]].rename(columns={primary_asset: "Close"})

    # Handle theta default (current price)
    current_price_live = float(price_df["Close"].iloc[-1])
    theta_val = theta_input if theta_input > 0 else current_price_live

    if price_df.shape[0] < long_window + 50:
        st.warning(
            f"Limited history ({price_df.shape[0]} rows) for long window ({long_window}). "
            "Consider more data for stable estimates."
        )

    # Core computations -------------------------------------------------------
    features = compute_features(price_df, short_window, long_window)
    log_returns = compute_log_returns(price_df["Close"])
    if log_returns.empty:
        st.error("Unable to compute log returns. Provide more price history.")
        return

    # --- Trend Score for Decision Engine ---
    if not features.empty:
        latest_feat = features.iloc[-1]
        trend_flag = bool(latest_feat.get("Signal", 0))
        trend_score = 1.0 if trend_flag else -1.0
    else:
        trend_score = 0.0

    # --- STEP 1: ASSET CLASSIFICATION ---
    daily_vol_ann = get_daily_volatility_class(log_returns)
    liquidity_data = check_data_liquidity(price_df)

    symbol_for_class = primary_asset if primary_asset is not None else "default"
    asset_class = get_asset_class(symbol_for_class, daily_vol_ann, liquidity_data)
    dynamic_params = get_dynamic_trading_params(asset_class)

    # --- DYNAMIC ATR SLIDER ---
    st.sidebar.markdown("---")
    with st.sidebar.expander("Dynamic ATR Trailing Stop", expanded=False):
        suggested_sl_mult = float(dynamic_params.get("sl_mult", 3.0))
        trailing_stop_mult = st.slider(
            f"Trailing SL Multiplier (x ATR) - Suggested: {suggested_sl_mult:.1f}",
            min_value=1.0,
            max_value=10.0,
            value=suggested_sl_mult,
            step=0.1,
            key="tsl_multiplier_slider",
        )
        st.write(f"Entry Threshold: **{dynamic_params['prob_thresh']:.1%}**")
        st.write(
            f"Min Exposure: **{dynamic_params['min_exposure']:.1%}** (for {asset_class})"
        )
        st.write(f"Regime MA: **{dynamic_params['regime_ma']}** Days")
    st.session_state["asset_type_code"] = asset_class
    st_session_regime = dynamic_params.get("regime_ma", regime_window)
    st.session_state["regime_window"] = st_session_regime
    asset_type_code = asset_class
    regime_window = st_session_regime

    atr_input = pd.DataFrame(index=price_df.index)
    atr_input["Close"] = price_df["Close"]
    abs_returns = price_df["Close"].pct_change().abs().fillna(0.0)
    if "High" in price_df.columns and "Low" in price_df.columns:
        atr_input["High"] = price_df["High"]
        atr_input["Low"] = price_df["Low"]
    else:
        atr_input["High"] = price_df["Close"] * (1 + abs_returns)
        atr_input["Low"] = (price_df["Close"] * (1 - abs_returns)).clip(lower=0.0)
    atr_series = calculate_atr(atr_input, period=14)
    atr_mean = calculate_mean_atr(atr_series)

    multi_returns_df = compute_log_returns(multi_prices_df)
    returns_stats = summarize_returns(log_returns)

    mu_sigma = estimate_mu_sigma(
        log_returns,
        periods_per_year=PERIODS_PER_YEAR,
        alpha=0.0,
        risk_free_rate_prior=rf_annual,
    )
    constant_sigma = mu_sigma["daily_sigma"]

    current_price = float(features["Close"].iloc[-1])

    regime_result = detect_regime(log_returns, method=regime_method)

    ewma_series = compute_ewma_vol(log_returns, lam=0.94)
    garch_series, garch_params, vol_warning = fit_garch_vol(log_returns)
    if vol_warning:
        st.warning(vol_warning)

    selected_sigma_daily = constant_sigma
    if mc_model_type == "Growth (Stochastic Vol)" and garch_series is not None:
        selected_sigma_daily = float(garch_series.iloc[-1])

    # Factor model artifacts (decoupled pipeline)
    signal_horizon = 60
    sector_name = SECTOR_LOOKUP.get(primary_asset)
    cv_selector = ArtifactSelector("cv_batch")
    bt_selector = ArtifactSelector("bulk_bt")
    risk_selectors = {
        label: ArtifactSelector(meta["namespace"])
        for label, meta in RISK_ENGINE_OPTIONS.items()
    }
    cv_runs = cv_selector.available_runs()
    cv_run_options = ["Latest"] + list(reversed(cv_runs)) if cv_runs else []
    if cv_run_options:
        selected_cv_label = st.sidebar.selectbox(
            "Factor Artifact Run",
            options=cv_run_options,
            index=0,
        )
    else:
        st.sidebar.warning(
            "No CV batch artifacts found. Use the Job Queue Controls to populate the signal model."
        )
        selected_cv_label = "No runs available"
    cv_run_override = (
        None
        if selected_cv_label in ("Latest", "No runs available")
        else selected_cv_label
    )
    active_cv_run = cv_run_override or cv_selector.latest()
    artifact_payload = (
        cv_selector.load(primary_asset, run_id=cv_run_override)
        if active_cv_run
        else None
    )
    cv_run_meta = (
        load_run_metadata("cv_batch", run_id=active_cv_run) if active_cv_run else None
    )
    factor_model_result = (
        artifact_to_factor_model(artifact_payload) if artifact_payload else None
    )

    # Multi-asset snapshot using cached artifacts
    snapshot_rows = []
    missing_symbols = []
    if active_cv_run and selected_tickers:
        max_snapshot = 25
        for sym in selected_tickers[:max_snapshot]:
            payload = cv_selector.load(sym, run_id=cv_run_override)
            if not payload:
                missing_symbols.append(sym)
                continue
            result = artifact_to_factor_model(payload)
            if result is None:
                missing_symbols.append(sym)
                continue
            snapshot_rows.append(
                {
                    "Symbol": sym,
                    "Probability": round((result.probability or 0.5) * 100, 2),
                    "Tier": getattr(result, "model_tier", "Unrated"),
                    "Trust %": round(getattr(result, "trust_score", 1.0) * 100, 1),
                    "CV Score %": round((result.accuracy_cv or 0.0) * 100, 2),
                    "CV Std %": round((result.accuracy_std or 0.0) * 100, 2),
                    "Prob Edge %": round(
                        (result.cv_metrics or {}).get("prob_edge", 0.0) * 100, 2
                    )
                    if result.cv_metrics
                    else None,
                    "Warnings": "; ".join(result.warnings) if result.warnings else "",
                }
            )
    if snapshot_rows:
        st.subheader("Multi-Asset Signal Snapshot")
        st.caption(
            "Pulled from cached CV artifacts (run %s)." % (active_cv_run or "Latest")
        )
        st.dataframe(pd.DataFrame(snapshot_rows))
        if len(selected_tickers) > len(snapshot_rows):
            st.caption(
                "Limited to %d symbols for brevity. Adjust the multiselect to focus on a smaller set."
                % len(snapshot_rows)
            )
    elif selected_tickers and active_cv_run:
        st.info(
            "No cached CV artifacts available for the selected symbols in run %s."
            % (active_cv_run or "Latest")
        )
    if missing_symbols:
        st.caption(
            "Artifacts missing for: %s. Re-run the CV batch if you need fresh signals."
            % ", ".join(missing_symbols[:10])
        )

    if (
        factor_model_result
        and factor_model_result.history is not None
        and not factor_model_result.history.empty
    ):
        perf_metrics = calculate_performance_metrics(
            history=factor_model_result.history,
            log_returns=log_returns,
            threshold=0.60,
        )
    else:
        perf_metrics = {
            "win_loss_ratio": 1.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "trade_count": 0,
        }
    win_loss_ratio = perf_metrics.get("win_loss_ratio", 1.0)
    
    with st.sidebar.expander("Kelly Inputs (Backtest)", expanded=False):
        st.write(f"**Win/Loss Ratio (b):** {win_loss_ratio:.2f}")
        st.write(f"**Hist. Win Rate:** {perf_metrics.get('win_rate', 0.5):.1%}")
        st.write(f"**Avg Win:** {perf_metrics.get('avg_win', 0.0) * 100:.2f}%")
        st.write(f"**Avg Loss:** {perf_metrics.get('avg_loss', 0.0) * 100:.2f}%")
        st.write(f"**Trades Analyzed:** {perf_metrics.get('trade_count', 0)}")

    downside_result = compute_downside_risk(price_df["Close"], log_returns)
    rho = compute_autocorrelation(log_returns)

    return_method = selected_risk_engine["return_method"]
    garch_payload = None

    er_hist_recent = mu_sigma.get("annual_mu_recent", mu_sigma["annual_mu"])
    market_er_annual = mu_sigma.get("annual_mu_long", er_hist_recent)
    er_capm = estimate_capm_er(
        beta=1.0, market_er_annual=market_er_annual, rf_annual=rf_annual
    )

    if shrinkage_mode:
        market_proxy = None
        if len(selected_tickers) > 1:
            try:
                market_proxy = compute_log_returns(
                    multi_prices_df[selected_tickers]
                ).mean(axis=1)
            except Exception:
                market_proxy = None

        js_out = james_stein_drift(
            log_returns.values if hasattr(log_returns, "values") else log_returns,
            market_returns=(
                market_proxy.values if isinstance(market_proxy, pd.Series) else market_proxy
            ),
        )

        if isinstance(js_out, dict):
            if "annual_mu" in js_out:
                robust_er_annual = float(js_out["annual_mu"])
            elif "mu_annual" in js_out:
                robust_er_annual = float(js_out["mu_annual"])
            elif "mu" in js_out:
                robust_er_annual = float(js_out["mu"])
            else:
                robust_er_annual = float(next(iter(js_out.values())))
        elif isinstance(js_out, (list, tuple, np.ndarray, pd.Series)):
            robust_er_annual = float(js_out[0])
        else:
            robust_er_annual = float(js_out)

        er_hist_recent = robust_er_annual

    hist_returns_payload = (
        _series_to_payload(log_returns) if return_method == "BlockBootstrap" else None
    )

    canonical_result = cached_canonical_engine(
        current_price=current_price,
        sigma_daily=selected_sigma_daily,
        er_recent_annual=er_hist_recent,
        er_capm_annual=er_capm,
        shrinkage_factor=shrinkage_factor,
        capm_weight=capm_weight,
        rf_annual=rf_annual,
        horizon_days=int(mc_horizon),
        sims=int(mc_sims),
        es_conf_level=es_conf_level,
        distribution=distribution_choice,
        student_df=student_df,
        return_generation_method=return_method,
        rho=rho,
        garch_payload=garch_payload,
        kappa=kappa,
        theta=theta_val,
        lambda_jump=lambda_jump,
        jump_mean=jump_mean,
        jump_std=jump_std,
        hist_returns_payload=hist_returns_payload,
    )

    display_paths = canonical_result["paths"]
    terminal_returns_display = canonical_result["terminal_returns"]
    final_prices_display = display_paths[-1, :] if display_paths.size else np.array([])
    expected_return = canonical_result["mc_expected_return"]
    es_horizon = canonical_result["mc_expected_shortfall"]
    es_annual = canonical_result["es_annual"]
    prob_gain_mc = canonical_result["prob_gain"]
    blended_er = canonical_result["er_blended"]
    sim_kurtosis = canonical_result.get("sim_kurtosis", 0.0)

    canonical_daily_mu = math.log1p(max(blended_er, -0.999)) / PERIODS_PER_YEAR
    annual_mu_mc = canonical_daily_mu * PERIODS_PER_YEAR
    annual_sigma_mc = selected_sigma_daily * math.sqrt(PERIODS_PER_YEAR)

    # =========================================================================
    # [REALISM UPGRADE] SMART SIGNAL FUSION (Calibrated)
    # =========================================================================
    final_prob_gain = prob_gain_mc
    direction_label = "Neutral"
    position_size = 0.0
    expected_edge_value = 0.0
    decision_reasons: List[str] = []
    raw_ai_prob_display = 0.5
    ai_prob_value = None

    if factor_model_result is not None:
        ai_prob_value = getattr(factor_model_result, "probability", None)
        if ai_prob_value is not None:
            raw_ai_prob_display = ai_prob_value

    ai_prob_active = None
    use_ai_brain = False
    if ai_prob_value is not None:
        st.sidebar.markdown("---")
        use_ai_brain = st.sidebar.checkbox(
            "🤖 Enable AI Brain?",
            value=False,
            help="Uncheck to ignore the calibrated AI probability and trade purely on Monte Carlo math.",
        )
        if use_ai_brain:
            ai_prob_active = ai_prob_value

    (
        direction_label,
        position_size,
        expected_edge_value,
        fused_prob_gain,
        decision_reasons,
    ) = decision_engine(
        er_blended=blended_er,
        expected_shortfall=es_annual,
        prob_gain=prob_gain_mc,
        terminal_returns=terminal_returns_display,
        sim_kurtosis=sim_kurtosis,
        risk_aversion_lambda=risk_aversion_lambda,
        no_trade_threshold=NO_TRADE_UTILITY_THRESHOLD,
        regime=regime_result.regime,
        regime_confidence=(
            (regime_result.confidence / 100.0)
            if hasattr(regime_result, "confidence")
            else None
        ),
        transaction_cost=transaction_cost,
        win_loss_ratio=win_loss_ratio,
        kelly_scale=0.5,
        trend_score=trend_score,
        ai_raw_prob=ai_prob_active,
    )
    final_prob_gain = fused_prob_gain
    raw_ai_prob = raw_ai_prob_display
    
    st.subheader(f"Canonical Decision Model ({mc_model_type})")
    st.caption("This is the primary signal engine. All decisions are based on this model.")

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    summary_col1.metric("Signal Direction", direction_label)
    summary_col2.metric("Position Recommendation", f"{position_size * 100:.1f}%")
    summary_col3.metric("Expected Edge", f"{expected_edge_value * 100:.2f}%")
    summary_col4.metric(
        "Tail Risk (Kurtosis)",
        f"{sim_kurtosis:.2f}",
        delta="High" if sim_kurtosis > 3.0 else "Normal",
        delta_color="inverse",
    )

    detail_cols = st.columns(4)
    detail_cols[0].metric("Blended Expected Return", f"{blended_er * 100:.2f}%")
    detail_cols[1].metric("Unified ES (ann.)", f"{es_annual * 100:.2f}%")
    detail_cols[2].metric("Probability of Gain", f"{final_prob_gain * 100:.2f}%", help="Fused probability (Monte Carlo + AI)")
    detail_cols[3].metric(
        "Regime", f"{regime_result.regime} ({regime_result.confidence:.1f}% conf)"
    )

    raw_prob = getattr(factor_model_result, "probability", None)
    if raw_prob is None:
        raw_prob = 0.5
    brain_col1, brain_col2 = st.columns(2)
    brain_col1.metric(
        "Raw Win Probability",
        f"{raw_prob:.1%}",
        delta=f"{raw_prob - 0.5:.1%}",
        help="Raw statistical chance (from the factor model) that the stock goes UP over the chosen prediction horizon.",
    )
    confidence_status = "High" if abs(raw_prob - 0.5) > 0.05 else "Low (Noise)"
    brain_col2.metric(
        "Statistical Confidence",
        confidence_status,
        help="How far the raw win probability is from a 50/50 coin flip.",
    )

    cleaned_reasons = [reason for reason in decision_reasons if isinstance(reason, str)]

    auto_reasons: List[str] = []

    kelly_sentence = (
        "Kelly sizing uses the fused probability and the payoff distribution from Monte Carlo to determine the allocation "
        "without ad-hoc overrides."
    )

    if direction_label.lower().startswith("bull"):
        auto_reasons.append(
            (
                "Stock looks attractive on a risk-adjusted basis: the fused probability of gain is "
                f"{final_prob_gain * 100:.1f}% with a blended expected return of {blended_er * 100:.1f}% "
                f"against an annualized expected shortfall of {es_annual * 100:.1f}%. "
                f"Half-Kelly sizing recommends allocating {position_size * 100:.1f}% because the expected edge is "
                f"{expected_edge_value * 100:.2f}% per $1 risked."
            )
        )
    elif direction_label.lower().startswith("avoid") or direction_label.lower().startswith("bear"):
        auto_reasons.append(
            (
                "Model recommends avoiding or reducing exposure: despite a blended expected return of "
                f"{blended_er * 100:.1f}% the fused probability of gain ({final_prob_gain * 100:.1f}%) "
                f"and expected edge ({expected_edge_value * 100:.2f}%) do not justify taking risk against "
                f"an annualized expected shortfall of {es_annual * 100:.1f}%."
            )
        )
    else:
        auto_reasons.append(
            (
                "No strong trade is recommended because the fused probability of gain "
                f"({final_prob_gain * 100:.1f}%) and expected edge ({expected_edge_value * 100:.2f}%) "
                "are too close to zero relative to the downside risk."
            )
        )

    if expected_edge_value < 0:
        auto_reasons.append(
            "Expected edge is negative under the current payoff distribution, so Kelly sizing collapses the position to zero."
        )
    else:
        auto_reasons.append(kelly_sentence)

    raw_prob = getattr(factor_model_result, "probability", None)
    if raw_prob is not None:
        auto_reasons.append(
            (
                f"Internally, the factor model's raw win probability is {raw_prob:.1%}. Values close to 50% indicate "
                "low statistical edge (noise), while values further away from 50% indicate a stronger directional "
                "signal. The decision engine then layers risk aversion and downside risk on top of this raw signal."
            )
        )

    st.write("**Rationale**")
    for line in cleaned_reasons + auto_reasons:
        st.write(f"- {line}")

    if "show_diag" not in st.session_state:
        st.session_state["show_diag"] = False
    diag_label = (
        "Hide Advanced Diagnostics"
        if st.session_state["show_diag"]
        else "Show Advanced Diagnostics"
    )
    if st.button(diag_label):
        st.session_state["show_diag"] = not st.session_state["show_diag"]

    if st.session_state["show_diag"]:
        st.markdown("### Advanced Diagnostics")
        st.caption("Secondary analytical views for research/debugging.")

        hist_paths = run_monte_carlo_paths(
            current_price=current_price,
            annual_mu=mu_sigma["annual_mu"],
            annual_sigma=mu_sigma["annual_sigma"],
            days=int(mc_horizon),
            sims=int(mc_sims),
            distribution=distribution_choice,
            df=student_df,
            return_generation_method="GBM",
            rho=rho,
            daily_mu=mu_sigma["daily_mu"],
            daily_sigma=mu_sigma["daily_sigma"],
        )

        st.plotly_chart(
            build_mc_paths_figure(hist_paths, "Comparison: Historical Drift (GBM)"),
            use_container_width=True,
        )

        st.markdown("**Drift & Volatility Diagnostics**")
        diag_vol_cols = st.columns(2)
        diag_vol_cols[0].plotly_chart(
            build_trend_figure(features, short_window, long_window),
            use_container_width=True,
            key="trend_diag",
        )
        diag_vol_cols[1].plotly_chart(
            build_volatility_figure(
                price_df.index,
                constant_sigma=constant_sigma,
                ewma_sigma=ewma_series,
                garch_sigma=garch_series,
            ),
            use_container_width=True,
            key="vol_chart_diag",
        )

    st.markdown("---")
    st.markdown("**Model Inputs & Diagnostics**")
    params_col1, params_col2, params_col3, params_col4 = st.columns(4)
    params_col1.metric("Current Price", f"Rs. {current_price:,.2f}")
    params_col2.metric("Annualized μ", f"{mu_sigma['annual_mu'] * 100:.2f}%")
    params_col3.metric("Annualized σ", f"{mu_sigma['annual_sigma'] * 100:.2f}%")
    params_col4.metric(
        "Daily σ (selected)", f"{selected_sigma_daily * 100:.3f}%"
    )

    st.subheader("Regime Analysis")
    st.write(
        f"**Current Regime:** {regime_result.regime} "
        f"(Confidence {regime_result.confidence:.1f}%)"
    )
    st.caption(regime_result.info)
    if regime_result.vol_series is not None:
        st.plotly_chart(
            build_regime_chart(
                regime_result.vol_series,
                lower=regime_result.vol_series.quantile(0.3),
                upper=regime_result.vol_series.quantile(0.7),
            ),
            use_container_width=True,
            key="regime_chart_main",
        )

    if not features.empty:
        latest_row = features.iloc[-1]
        trend_flag = bool(latest_row.get("Signal", 0))
        safe_flag = bool(latest_row.get("Safe_To_Enter", False))
    else:
        trend_flag = False
        safe_flag = False
    trend_text = "Bullish" if trend_flag else "Bearish"
    vol_text = "Calm" if safe_flag else "Choppy"

    (
        tab_decision,
        tab_trend,
        tab_signal,
        tab_risk,
        tab_pca,
        tab_portfolio,
        tab_beta,
        tab_downside,
    ) = st.tabs(
        [
            "🧭 Decision Summary",
            "📈 Single Asset Analysis",
            "🤖 Signal Model",
            "🎲 Monte Carlo & Risk",
            "🧮 PCA & Hidden Factors",
            "💼 Portfolio Optimizer (Markowitz)",
            "📊 Beta & Hedge Ratio",
            "📉 Downside Risk & Stress",
        ]
    )

    with tab_decision:
        st.header("Canonical Decision Summary")
        metric_block = st.columns(4)
        metric_block[0].metric("Signal", direction_label)
        metric_block[1].metric("Position", f"{position_size * 100:.1f}%")
        metric_block[2].metric("Expected Edge", f"{expected_edge_value * 100:.2f}%")
        metric_block[3].metric("Probability of Gain", f"{final_prob_gain * 100:.2f}%")

        stats_cols = st.columns(3)
        stats_cols[0].metric("Blended ER", f"{blended_er * 100:.2f}%")
        stats_cols[1].metric("MC Expected Return", f"{expected_return * 100:.2f}%")
        stats_cols[2].metric("Unified ES (ann.)", f"{es_annual * 100:.2f}%")

        st.markdown("**Reasons**")
        for line in decision_reasons:
            st.write(f"- {line}")

    with tab_trend:
        st.subheader("Single Asset Analysis")
        st.plotly_chart(
            build_trend_figure(features, short_window, long_window),
            use_container_width=True,
        )
        st.info(
            f"Trend overlay currently shows {trend_text.lower()} and volatility regime is {vol_text.lower()}. "
            "A positive trend with stable volatility supports taking exposure; the opposite suggests caution."
        )

    with tab_signal:
        st.subheader("Multi-Factor Probability Model")
        st.caption(f"Artifact run: {active_cv_run or 'N/A'}")
        if cv_run_meta:
            st.caption(
                _format_run_metadata(
                    "CV run config",
                    cv_run_meta,
                    keys=["run_id", "horizon", "cv_folds", "limit"],
                )
            )
        if factor_model_result is None:
            st.warning(
                f"No factor artifact is available for {primary_asset} "
                f"(run {active_cv_run or 'N/A'})."
            )
            st.info("Use the Job Queue Controls in the sidebar to request a new CV batch run.")
        else:
            if factor_model_result.warnings:
                for w in factor_model_result.warnings:
                    st.warning(w)
            if factor_model_result.message:
                st.write(factor_model_result.message)

            diag_cols = st.columns(5)
            diag_cols[0].metric(
                "CV Accuracy",
                f"{factor_model_result.accuracy_cv * 100:.2f}%"
                if factor_model_result.accuracy_cv is not None
                else "N/A",
            )
            diag_cols[1].metric(
                "CV Std",
                f"{factor_model_result.accuracy_std * 100:.2f}%"
                if factor_model_result.accuracy_std is not None
                else "N/A",
            )
            diag_cols[2].metric(
                "Position Multiplier",
                f"{factor_model_result.position_multiplier:.2f}x"
                if factor_model_result.position_multiplier is not None
                else "N/A",
            )
            diag_cols[3].metric(
                "Model Tier",
                getattr(factor_model_result, "model_tier", "Unrated") or "Unrated",
            )
            trust_val = getattr(factor_model_result, "trust_score", None)
            diag_cols[4].metric(
                "Trust Score",
                f"{trust_val * 100:.0f}%" if trust_val is not None else "N/A",
            )
            if factor_model_result.cv_metrics:
                st.markdown("**CV Diagnostics (averaged across purged folds)**")
                pretty_rows = []
                metric_schema = [
                    ("balanced_accuracy", "Balanced Accuracy", "pct"),
                    ("auc", "ROC AUC", "pct"),
                    ("brier", "Brier Score", "raw"),
                    ("log_loss", "Log Loss", "raw"),
                    ("prob_edge", "Probability Edge vs. 50/50", "pct"),
                ]
                for key, label, mode in metric_schema:
                    value = factor_model_result.cv_metrics.get(key)
                    if value is None:
                        continue
                    if mode == "pct":
                        display = f"{value * 100:.2f}%"
                    else:
                        display = f"{value:.4f}"
                    pretty_rows.append({"Metric": label, "Value": display})
                if pretty_rows:
                    st.table(pd.DataFrame(pretty_rows))
                prob_scaler = factor_model_result.cv_metrics.get("prob_scaler")
                governance_flags = factor_model_result.cv_metrics.get("governance_flags") or []
                if prob_scaler is not None:
                    st.caption(f"Probability scaler applied: {prob_scaler:.2f}")
                if governance_flags:
                    for flag in governance_flags:
                        st.info(flag)
                fold_scores = factor_model_result.cv_metrics.get("fold_scores")
                if fold_scores:
                    fold_df = pd.DataFrame(
                        {
                            "Fold": list(range(1, len(fold_scores) + 1)),
                            "Score": fold_scores,
                        }
                    )
                    fold_fig = px.line(
                        fold_df,
                        x="Fold",
                        y="Score",
                        markers=True,
                        title="Purged CV Fold Scores",
                    )
                    fold_fig.update_layout(yaxis_tickformat=".2%")
                    st.plotly_chart(fold_fig, use_container_width=True, key="cv_fold_curve")
                fold_details = factor_model_result.cv_metrics.get("fold_details")
                if fold_details:
                    details_df = pd.DataFrame(fold_details)
                    details_df = details_df[
                        [
                            col
                            for col in [
                                "fold",
                                "train_start",
                                "train_end",
                                "test_start",
                                "test_end",
                                "balanced_accuracy",
                                "auc",
                                "brier",
                                "log_loss",
                            ]
                            if col in details_df.columns
                        ]
                    ]
                    st.dataframe(
                        details_df.rename(
                            columns={
                                "fold": "Fold",
                                "train_start": "Train Start",
                                "train_end": "Train End",
                                "test_start": "Test Start",
                                "test_end": "Test End",
                                "balanced_accuracy": "Balanced Acc.",
                                "auc": "AUC",
                                "brier": "Brier",
                                "log_loss": "Log Loss",
                            }
                        ).style.format(
                            {
                                "Balanced Acc.": "{:.2%}",
                                "AUC": "{:.2%}",
                                "Brier": "{:.4f}",
                                "Log Loss": "{:.4f}",
                            }
                        ),
                        use_container_width=True,
                    )
            validation = factor_model_result.validation_metrics
            if validation:
                st.markdown("**Post-CV Validation**")
                val_rows = []
                val_schema = [
                    ("ic", "Information Coefficient", "raw"),
                    ("ic_tstat", "IC t-stat", "raw"),
                    ("trade_count", "Trades", "raw"),
                    ("win_rate", "Win Rate", "pct"),
                    ("cagr", "CAGR", "pct"),
                    ("max_drawdown", "Max Drawdown", "pct"),
                ]
                for key, label, mode in val_schema:
                    value = validation.get(key)
                    if value is None:
                        continue
                    if mode == "pct":
                        display = f"{value * 100:.2f}%"
                    else:
                        display = f"{value:.3f}" if isinstance(value, float) else str(value)
                    val_rows.append({"Metric": label, "Value": display})
                if val_rows:
                    st.table(pd.DataFrame(val_rows))
            if factor_model_result.disabled_reason:
                st.error(f"Signal disabled: {factor_model_result.disabled_reason}")

            st.subheader("Factor Contribution and Importance")
            if (
                factor_model_result.feature_importance_df is not None
                and not factor_model_result.feature_importance_df.empty
            ):
                try:
                    importance_df = factor_model_result.feature_importance_df.sort_values(
                        "Importance", ascending=False
                    )
                    top_features = importance_df.head(10)
                    fig_importance = px.bar(
                        top_features.sort_values("Importance"),
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title="Top 10 Factor Importance Scores",
                        color="Importance",
                        color_continuous_scale=px.colors.sequential.Plasma,
                    )
                    fig_importance.update_layout(
                        yaxis={"categoryorder": "total ascending"}
                    )
                    st.plotly_chart(
                        fig_importance,
                        use_container_width=True,
                        key="factor_importance_chart",
                    )
                except Exception as exc:
                    st.error(f"Could not display feature importance chart: {exc}")

            if factor_model_result.contributions is not None:
                contrib_col, stats_col = st.columns([2, 1])
                contrib_col.plotly_chart(
                    build_factor_contribution_chart(factor_model_result.contributions),
                    use_container_width=True,
                )
                coeff_df = pd.DataFrame(
                    {
                        "Coefficient": factor_model_result.coefficients,
                        "Latest z-score": factor_model_result.latest_factors,
                    }
                )
                stats_col.dataframe(coeff_df.style.format("{:.3f}"))

            if factor_model_result.history is not None:
                st.plotly_chart(
                    build_signal_history_chart(factor_model_result.history),
                    use_container_width=True,
                )
            latest_prob = factor_model_result.probability or 0.5
            interpret = (
                "supports"
                if latest_prob >= 0.55
                else "argues against"
                if latest_prob <= 0.45
                else "suggests neutral"
            )
            st.info(
                f"Latest factor probability is {latest_prob:.2f}, which {interpret} adding exposure relative to the base trend."
            )

        st.markdown("---")
        st.subheader("Production Backtest Snapshot")
        bt_runs = bt_selector.available_runs()
        if not bt_runs:
            st.warning("No bulk backtest artifacts found.")
            st.info("Use the Job Queue Controls in the sidebar to request a new bulk backtest run.")
        else:
            bt_options = ["Latest"] + list(reversed(bt_runs))
            selected_bt_label = st.selectbox(
                "Backtest Artifact Run",
                options=bt_options,
                index=0,
                key=f"bt_run_select_{primary_asset}",
            )
            bt_run_override = None if selected_bt_label == "Latest" else selected_bt_label
            active_bt_run = bt_run_override or bt_selector.latest()
            bt_payload = (
                bt_selector.load(primary_asset, run_id=bt_run_override)
                if active_bt_run
                else None
            )
            bt_run_meta = (
                load_run_metadata("bulk_bt", run_id=active_bt_run)
                if active_bt_run
                else None
            )
            if not bt_payload:
                st.warning(
                    f"No backtest artifact for {primary_asset} in run {active_bt_run}."
                )
                st.info("Use the Job Queue Controls in the sidebar to request a new bulk backtest run for this symbol.")
            else:
                st.caption(
                    f"Artifact run: {active_bt_run} · Beats Market: {bt_payload.get('Beats Market')}"
                )
                if bt_run_meta:
                    st.caption(
                        _format_run_metadata(
                            "Bulk backtest config",
                            bt_run_meta,
                            keys=["run_id", "limit", "horizon_days", "transaction_cost"],
                        )
                    )
                metrics_cols = st.columns(3)
                metrics_cols[0].metric(
                    "Probability",
                    _format_percent(bt_payload.get("Probability"), digits=1, already_percent=True),
                )
                metrics_cols[1].metric(
                    "Confidence",
                    _format_percent(bt_payload.get("Confidence"), digits=1, already_percent=True),
                )
                metrics_cols[2].metric(
                    "Win Rate",
                    _format_percent(bt_payload.get("Win Rate"), digits=1, already_percent=True),
                )
                stats_cols = st.columns(3)
                stats_cols[0].metric(
                    "Trades",
                    f"{bt_payload.get('Trades', 'N/A')}",
                )
                stats_cols[1].metric(
                    "AI Daily Return",
                    _format_percent(bt_payload.get("AI Daily Return"), digits=2, already_percent=False),
                )
                stats_cols[2].metric(
                    "Profit Factor",
                    f"{bt_payload.get('Profit Factor', 'N/A')}",
                )
                components_raw = bt_payload.get("Components")
                components_data = None
                if components_raw:
                    if isinstance(components_raw, str):
                        try:
                            components_data = json.loads(components_raw)
                        except Exception:
                            components_data = None
                    elif isinstance(components_raw, dict):
                        components_data = components_raw
                if components_data:
                    comp_df = (
                        pd.DataFrame(
                            [{"Model": k, "Probability": v} for k, v in components_data.items()]
                        )
                        .sort_values("Probability", ascending=False)
                    )
                    comp_df["Probability"] = comp_df["Probability"].map(
                        lambda x: f"{x:.2f}"
                    )
                    st.markdown("**Model Blend Contributions**")
                    st.table(comp_df)

    with tab_risk:
        st.subheader("Monte Carlo Simulation & Tail Risk")
        risk_view_labels = list(RISK_ENGINE_OPTIONS.keys())
        default_view_engine = st.session_state.get(
            "risk_engine_label", DEFAULT_RISK_ENGINE_LABEL
        )
        if default_view_engine not in risk_view_labels:
            default_view_engine = DEFAULT_RISK_ENGINE_LABEL
        risk_view_label = st.selectbox(
            "Risk Artifact Source",
            options=risk_view_labels,
            index=risk_view_labels.index(default_view_engine),
            key=f"risk_view_select_{primary_asset}",
            help="Choose which Monte Carlo engine's artifacts to inspect.",
        )
        risk_selector = risk_selectors[risk_view_label]
        risk_namespace = RISK_ENGINE_OPTIONS[risk_view_label]["namespace"]
        risk_runs = risk_selector.available_runs()
        if not risk_runs:
            st.warning("No risk artifacts found.")
            st.info("Use the Job Queue Controls in the sidebar to enqueue a new risk job.")
        else:
            risk_options = ["Latest"] + list(reversed(risk_runs))
            selected_risk_label = st.selectbox(
                "Risk Artifact Run",
                options=risk_options,
                index=0,
                key=f"risk_run_select_{primary_asset}",
            )
            risk_run_override = (
                None if selected_risk_label == "Latest" else selected_risk_label
            )
            active_risk_run = risk_run_override or risk_selector.latest()
            risk_payload = (
                risk_selector.load(primary_asset, run_id=risk_run_override)
                if active_risk_run
                else None
            )
            risk_run_meta = load_run_metadata(
                risk_namespace, run_id=active_risk_run
            ) if active_risk_run else None
            if not risk_payload or "distribution" not in risk_payload:
                st.warning(
                    f"No risk artifact for {primary_asset} in run {active_risk_run}."
                )
                st.info("Use the Job Queue Controls in the sidebar to enqueue a risk job for this symbol.")
            else:
                updated_at = risk_payload.get("updated_at", "N/A")
                st.caption(
                    f"Artifact run: {active_risk_run} · Engine: {risk_view_label} · Updated: {updated_at}"
                )
                if risk_run_meta:
                    st.caption(
                        _format_run_metadata(
                            "Risk job config",
                            risk_run_meta,
                            keys=["run_id", "horizon_days", "sims"],
                        )
                    )
                distribution_payload = risk_payload.get("distribution")
                tails = risk_payload.get("tails", {})
                tail_cols = st.columns(4)
                tail_cols[0].metric(
                    "VaR (95%)",
                    _format_percent(tails.get("var_95"), digits=2, already_percent=False),
                )
                tail_cols[1].metric(
                    "VaR (99%)",
                    _format_percent(tails.get("var_99"), digits=2, already_percent=False),
                )
                tail_cols[2].metric(
                    "ES (95%)",
                    _format_percent(tails.get("es_95"), digits=2, already_percent=False),
                )
                tail_cols[3].metric(
                    "ES (99%)",
                    _format_percent(tails.get("es_99"), digits=2, already_percent=False),
                )
                st.metric(
                    "MC Expected Return",
                    _format_percent(
                        risk_payload.get("expected_return"), digits=2, already_percent=False
                    ),
                )
                failure_prob = None
                failure_return = failure_threshold_pct / 100.0
                if (
                    distribution_payload
                    and distribution_payload.get("bins")
                    and distribution_payload.get("counts")
                ):
                    bins = distribution_payload["bins"]
                    counts = distribution_payload["counts"]
                    if len(bins) == len(counts) + 1:
                        total = sum(counts)
                        if total > 0:
                            failure_counts = sum(
                                count
                                for count, upper in zip(counts, bins[1:])
                                if upper <= failure_return
                            )
                            failure_prob = failure_counts / total
                st.metric(
                    f"P(Return ≤ {failure_threshold_pct:.1f}%)",
                    _format_percent(failure_prob, digits=2, already_percent=False)
                    if failure_prob is not None
                    else "N/A",
                )
                dist_fig = _build_distribution_chart(distribution_payload)
                fan_fig = _build_fan_chart(risk_payload.get("fan_chart"))
                chart_cols = st.columns(2)
                if dist_fig is not None:
                    chart_cols[0].plotly_chart(dist_fig, use_container_width=True)
                else:
                    chart_cols[0].info("Distribution data unavailable.")
                if fan_fig is not None:
                    chart_cols[1].plotly_chart(fan_fig, use_container_width=True)
                else:
                    chart_cols[1].info("Fan chart data unavailable.")

    with tab_pca:
        st.subheader("PCA & Hidden Factors")
        if len(selected_tickers) < 2:
            st.warning("Select at least two assets for PCA analysis.")
        else:
            ret_sub = multi_returns_df[selected_tickers].dropna()
            if ret_sub.shape[0] < 30:
                st.warning("Need at least 30 observations of returns for PCA.")
            else:
                try:
                    eigvals, eigvecs, explained_var = run_pca(ret_sub)
                except ValueError as exc:
                    st.warning(str(exc))
                else:
                    components = np.arange(1, len(eigvals) + 1)
                    explained_df = pd.DataFrame(
                        {
                            "Component": components,
                            "Explained Variance": explained_var,
                            "Cumulative": explained_var.cumsum(),
                        }
                    )
                    bar_count = min(5, len(components))
                    bar_fig = go.Figure()
                    bar_fig.add_trace(
                        go.Bar(
                            x=[f"PC{i}" for i in components[:bar_count]],
                            y=explained_var[:bar_count] * 100,
                            marker_color="#26c6da",
                        )
                    )
                    bar_fig.update_layout(
                        template="plotly_dark",
                        margin=dict(l=10, r=10, t=40, b=40),
                        xaxis_title="Principal Component",
                        yaxis_title="Explained Variance (%)",
                    )
                    st.plotly_chart(bar_fig, use_container_width=True)
                    st.dataframe(
                        explained_df.head(5).style.format(
                            {
                                "Explained Variance": "{:.2%}",
                                "Cumulative": "{:.2%}",
                            }
                        )
                    )

                    num_components = min(3, eigvecs.shape[1])
                    loadings = pd.DataFrame(
                        eigvecs[:, :num_components],
                        index=selected_tickers,
                        columns=[f"PC{i}" for i in range(1, num_components + 1)],
                    )
                    st.write("**Loadings (first 3 components)**")
                    st.dataframe(loadings.style.format("{:.3f}"))

                    eff_factors = (eigvals.sum() ** 2) / (np.square(eigvals).sum())
                    variance_two = (
                        explained_var[:2].sum() * 100
                        if len(explained_var) >= 2
                        else explained_var[0] * 100
                    )
                    st.info(
                        f"First 2 components explain approximately {variance_two:.1f}% of total variance. "
                        f"Effective number of factors ≈ {eff_factors:.2f}, suggesting the selected assets are driven "
                        "by a small number of hidden market influences."
                    )

    with tab_portfolio:
        st.subheader("Portfolio Optimizer (Markowitz & Black-Litterman)")
        st.caption(
            "Compare allocations using Classical Mean-Variance, Robust Hierarchical Risk Parity (HRP), and Bayesian Black-Litterman models."
        )
        
        opt_assets = st.multiselect(
            "Select assets for portfolio optimization",
            options=available_tickers,
            default=selected_tickers,
            key="portfolio_assets_v2",
        )
        
        if len(opt_assets) < 2:
            st.warning("Select at least two assets to run the optimizer.")
        else:
            ret_subset = multi_returns_df[opt_assets].dropna()
            
            if ret_subset.shape[0] < 60:
                st.warning(
                    "Need at least 60 return observations to estimate mean and covariance reliably."
                )
            else:
                mu_vec, cov_mat = estimate_mean_cov(ret_subset)
                n_assets = len(opt_assets)
                
                col_cap, col_rf = st.columns(2)
                with col_cap:
                    capital = st.number_input(
                        "Total capital (Rs.)",
                        min_value=0.0,
                        value=100000.0,
                        step=10000.0,
                        key="capital_v2"
                    )
                with col_rf:
                    rf_rate_input = st.number_input(
                        "Risk-free rate (annual, decimal)",
                        min_value=0.0,
                        max_value=0.2,
                        value=0.05,
                        step=0.01,
                        key="rf_rate_v2"
                    )

                portfolios = []

                # --- A. STANDARD PORTFOLIOS ---
                w_equal = np.ones(n_assets) / n_assets
                ew_return, ew_vol = compute_portfolio_stats(w_equal, mu_vec, cov_mat)
                portfolios.append({
                    "Name": "Equal Weight",
                    "Weights": w_equal,
                    "Return": ew_return,
                    "Vol": ew_vol,
                    "Capital": w_equal * capital,
                })

                w_min = min_variance_portfolio(cov_mat)
                if w_min is not None:
                    mv_return, mv_vol = compute_portfolio_stats(w_min, mu_vec, cov_mat)
                    portfolios.append({
                        "Name": "Min Variance",
                        "Weights": w_min,
                        "Return": mv_return,
                        "Vol": mv_vol,
                        "Capital": w_min * capital,
                    })
                else:
                    st.warning("Minimum-variance optimizer failed to converge.")

                if rf_rate_input >= 0:
                    w_tan = tangency_portfolio(mu_vec, cov_mat, rf_rate_input)
                    if w_tan is not None:
                        tan_return, tan_vol = compute_portfolio_stats(w_tan, mu_vec, cov_mat)
                        portfolios.append({
                            "Name": "Tangency (Classical)",
                            "Weights": w_tan,
                            "Return": tan_return,
                            "Vol": tan_vol,
                            "Capital": w_tan * capital,
                        })

                # --- B. BLACK-LITTERMAN PORTFOLIO ---
                st.markdown("---")
                st.markdown("### 🧠 Black-Litterman Model (AI-Augmented)")
                st.caption("Blends the stable 'Market Prior' with your specific View on the primary asset.")

                market_prior = mu_vec * 0.5 

                view_ticker = primary_asset if primary_asset in opt_assets else opt_assets[0]
                
                col_view1, col_view2 = st.columns(2)
                with col_view1:
                    view_return = st.slider(f"Expected Return for {view_ticker}", -0.5, 1.0, 0.20, 0.05, key="bl_view_return")
                with col_view2:
                    view_conf = st.slider(f"Confidence in View ({view_ticker})", 0.1, 0.9, 0.60, 0.1, key="bl_view_conf")

                views = {view_ticker: view_return}
                confidences = {view_ticker: view_conf}

                try:
                    bl_mu = black_litterman_posterior(market_prior, cov_mat, views, confidences)
                    w_bl = tangency_portfolio(bl_mu, cov_mat, rf_rate_input)
                    
                    if w_bl is not None:
                        bl_return, bl_vol = compute_portfolio_stats(w_bl, bl_mu, cov_mat)
                        portfolios.append({
                            "Name": "Black-Litterman (AI)",
                            "Weights": w_bl,
                            "Return": bl_return,
                            "Vol": bl_vol,
                            "Capital": w_bl * capital,
                        })
                    else:
                        st.warning("Black-Litterman optimizer did not converge.")
                except Exception as e:
                    st.error(f"Black-Litterman calculation error: {e}")

                # --- C. HRP OPTIMIZATION (Button) ---
                st.markdown("---")
                st.subheader("Hierarchical Risk Parity (HRP) [Robust Allocation]")
                if st.button("Run HRP Optimization", key="hrp_button"):
                    try:
                        hrp_weights = get_hrp_weights(cov_mat)
                        hrp_ret, hrp_vol = compute_portfolio_stats(hrp_weights.values, mu_vec, cov_mat)
                        
                        portfolios.append({
                            "Name": "HRP (Clustering)",
                            "Weights": hrp_weights.values,
                            "Return": hrp_ret,
                            "Vol": hrp_vol,
                            "Capital": hrp_weights.values * capital,
                        })
                        
                        w_df = pd.DataFrame({"Asset": hrp_weights.index, "Weight": hrp_weights.values})
                        fig_pie = px.pie(w_df, values="Weight", names="Asset", title="HRP Cluster Allocation")
                        st.plotly_chart(fig_pie, use_container_width=True)
                        st.info("HRP clusters correlated assets (e.g., Hydros vs Banks) to prevent concentration risk.")
                        
                    except Exception as exc:
                        st.error(f"HRP optimization failed: {exc}")

                # --- D. RESULTS DISPLAY ---
                if portfolios:
                    summary_df = pd.DataFrame({
                        "Portfolio": [p["Name"] for p in portfolios],
                        "Exp Return": [p["Return"] for p in portfolios],
                        "Volatility": [p["Vol"] for p in portfolios],
                        "Sharpe Ratio": [(p["Return"] - rf_rate_input) / p["Vol"] if p["Vol"] > 0 else 0 for p in portfolios]
                    })
                    st.write("### 📊 Performance Comparison")
                    st.dataframe(
                        summary_df.style.format({
                            "Exp Return": "{:.2%}",
                            "Volatility": "{:.2%}",
                            "Sharpe Ratio": "{:.2f}"
                        })
                    )

                    weights_df = pd.DataFrame(
                        {p["Name"]: p["Weights"] for p in portfolios}, index=opt_assets
                    ).T
                    st.write("### ⚖️ Asset Weights")
                    st.dataframe(weights_df.style.format("{:.1%}"))

                    capital_df = pd.DataFrame(
                        {p["Name"]: p["Capital"] for p in portfolios}, index=opt_assets
                    ).T
                    st.write("### 💰 Capital Allocation (Rs.)")
                    st.dataframe(capital_df.style.format("Rs. {:,.0f}"))

                    scatter = go.Figure()
                    
                    asset_vols = np.sqrt(np.diag(cov_mat.values))
                    scatter.add_trace(go.Scatter(
                        x=asset_vols, y=mu_vec.values, mode="markers", name="Individual Assets",
                        text=opt_assets, marker=dict(color="gray", size=8, opacity=0.5)
                    ))
                    
                    colors = ["#ffd54f", "#4fc3f7", "#81c784", "#ef5350", "#ba68c8"]
                    for i, p in enumerate(portfolios):
                        color = colors[i % len(colors)]
                        scatter.add_trace(go.Scatter(
                            x=[p["Vol"]], y=[p["Return"]], mode="markers+text",
                            text=[p["Name"]], textposition="top center", name=p["Name"],
                            marker=dict(size=15, color=color, line=dict(width=2, color="white"))
                        ))
                        
                    scatter.update_layout(
                        title="Risk-Return Landscape (Efficient Frontier)",
                        xaxis_title="Volatility (Risk)",
                        yaxis_title="Expected Return",
                        template="plotly_dark",
                        height=500,
                        xaxis=dict(tickformat=".1%"),
                        yaxis=dict(tickformat=".1%")
                    )
                    st.plotly_chart(scatter, use_container_width=True)
                    
                    best_port = max(portfolios, key=lambda p: (p["Return"] - rf_rate_input) / p["Vol"] if p["Vol"] > 0 else 0)
                    st.success(
                        f"🏆 **Winner:** {best_port['Name']} offers the best risk-adjusted return (Sharpe). "
                        "Consider allocating capital according to its weights."
                    )
                else:
                    st.warning("No portfolios generated.")

    with tab_beta:
        st.subheader("Beta & Hedge Ratio")
        if multi_returns_df.empty:
            st.warning(
                "No return data available. Please upload a valid multi-asset price CSV."
            )
        else:
            tickers = list(multi_returns_df.columns)
            if len(tickers) < 2:
                st.warning("Need at least two assets to compute beta.")
            else:
                stock_ticker = st.selectbox(
                    "Select stock (asset)", tickers, index=0, key="beta_stock"
                )
                default_index_idx = 1 if len(tickers) > 1 else 0
                index_ticker = st.selectbox(
                    "Select market index / benchmark",
                    tickers,
                    index=default_index_idx,
                    key="beta_index",
                )

                if stock_ticker == index_ticker:
                    st.warning("Stock and index must be different.")
                else:
                    stock_ret = multi_returns_df[stock_ticker]
                    index_ret = multi_returns_df[index_ticker]
                    alpha, beta, r2 = compute_beta(stock_ret, index_ret)
                    if beta is None:
                        st.warning(
                            "Not enough data to estimate beta (need at least ~30 observations)."
                        )
                    else:
                        st.subheader("Regression Results (Daily Returns)")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Beta", f"{beta:.2f}")
                        col2.metric("Alpha (daily)", f"{alpha:.4f}")
                        col3.metric(
                            "Alpha (annual)",
                            f"{alpha * PERIODS_PER_YEAR:.2f}",
                        )
                        col4.metric(
                            "R²",
                            f"{(r2 if r2 is not None else float('nan')):.2f}",
                        )
                        st.markdown(
                            "Beta > 1 means the stock tends to amplify market moves; beta < 1 means it moves less. "
                            "R² measures how much of the stock's movement is explained by the chosen benchmark."
                        )

                        hedged_ret = build_hedged_returns(
                            stock_ret.rename(stock_ticker),
                            index_ret.rename(index_ticker),
                            beta,
                        )
                        if not hedged_ret.empty:
                            st.subheader("Hedged vs Unhedged Performance")
                            cum_df = pd.concat(
                                [
                                    stock_ret.cumsum().rename(
                                        f"{stock_ticker} (cum. returns)"
                                    ),
                                    index_ret.cumsum().rename(
                                        f"{index_ticker} (cum. returns)"
                                    ),
                                    hedged_ret.cumsum().rename(
                                        f"{stock_ticker} hedged vs {index_ticker}"
                                    ),
                                ],
                                axis=1,
                            ).dropna()
                            st.line_chart(cum_df)

                            st.subheader("Hedge Ratio Example")
                            capital_stock = st.number_input(
                                "Notional investment in stock (Rs.)",
                                min_value=0.0,
                                value=100000.0,
                                step=10000.0,
                                key="hedge_capital",
                            )
                            latest_index_price = st.number_input(
                                f"Current level/price of {index_ticker} (approx.)",
                                min_value=0.0,
                                value=2000.0,
                                step=50.0,
                                key="hedge_index_price",
                            )
                            if capital_stock > 0 and latest_index_price > 0:
                                notional_index = beta * capital_stock
                                units_index = notional_index / latest_index_price
                                st.markdown(
                                    f"- Estimated **beta** of `{stock_ticker}` vs `{index_ticker}` is **{beta:.2f}**.\n"
                                    f"- To hedge **Rs. {capital_stock:,.0f}** of `{stock_ticker}`, short approx. "
                                    f"**Rs. {notional_index:,.0f}** of `{index_ticker}` "
                                    f"(≈ **{units_index:,.2f} units** at the current index level)."
                                )
                        else:
                            st.warning(
                                "Unable to build hedged series due to insufficient overlapping data."
                            )
                        beta_view = (
                            "amplifies market swings"
                            if beta > 1.2
                            else "tracks the market closely"
                            if beta >= 0.8
                            else "is relatively defensive"
                        )
                        st.info(
                            f"The estimated beta of {beta:.2f} suggests the asset {beta_view}; "
                            "consider hedging only if that profile conflicts with your risk target."
                        )

    with tab_downside:
        st.subheader("Downside Risk & Stress")
        if multi_returns_df.empty or multi_prices_df.empty:
            st.warning(
                "No multi-asset data available. Please upload a valid price CSV."
            )
        else:
            risk_asset = st.selectbox(
                "Select asset for downside analysis",
                options=available_tickers,
                index=available_tickers.index(primary_asset)
                if primary_asset in available_tickers
                else 0,
            )
            ret_series = multi_returns_df[risk_asset].dropna()
            price_series = multi_prices_df[risk_asset].dropna()
            if ret_series.empty or price_series.empty:
                st.warning(
                    "Insufficient historical data for the selected asset."
                )
            else:
                stats = compute_downside_stats(ret_series)
                (
                    col_d1,
                    col_d2,
                    col_d3,
                    col_d4,
                    col_d5,
                ) = st.columns(5)
                col_d1.metric(
                    "Downside Vol (ann.)",
                    f"{stats['semi_vol_annual'] * 100:.2f}%",
                )
                col_d2.metric(
                    "Max Drawdown", f"{stats['max_drawdown'] * 100:.2f}%"
                )
                col_d3.metric(
                    "Current Drawdown",
                    f"{stats['current_drawdown'] * 100:.2f}%",
                )
                col_d4.metric(
                    "Longest Drawdown", f"{stats['longest_dd_days']} days"
                )
                col_d5.metric("Sortino Ratio", f"{stats['sortino']:.2f}")

                dd_df = compute_drawdowns(price_series)
                if not dd_df.empty:
                    norm_price = (
                        dd_df["price"] / dd_df["price"].iloc[0] * 100
                    )
                    fig_dd = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                    )
                    fig_dd.add_trace(
                        go.Scatter(
                            x=norm_price.index,
                            y=norm_price.values,
                            mode="lines",
                            name="Normalized Price (100 = start)",
                            line=dict(color="#26c6da", width=2),
                        ),
                        row=1,
                        col=1,
                    )
                    fig_dd.add_trace(
                        go.Scatter(
                            x=dd_df.index,
                            y=dd_df["drawdown"].values * 100,
                            mode="lines",
                            name="Drawdown (%)",
                            line=dict(color="#ef5350", width=2),
                        ),
                        row=2,
                        col=1,
                    )
                    fig_dd.update_yaxes(
                        title_text="Index (100=Start)", row=1, col=1
                    )
                    fig_dd.update_yaxes(
                        title_text="Drawdown (%)", row=2, col=1
                    )
                    fig_dd.update_layout(
                        template="plotly_dark",
                        margin=dict(l=10, r=10, t=40, b=40),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_dd, use_container_width=True)

                st.subheader("Stress Testing")
                shock1 = st.number_input(
                    "Shock Scenario 1 (%)",
                    min_value=-0.90,
                    max_value=0.0,
                    value=-0.10,
                    step=0.01,
                )
                shock2 = st.number_input(
                    "Shock Scenario 2 (%)",
                    min_value=-0.95,
                    max_value=0.0,
                    value=-0.30,
                    step=0.01,
                )
                current_asset_price = price_series.iloc[-1]
                st.write(
                    f"- Current price: Rs. {current_asset_price:,.2f}\n"
                    f"- Shock 1 ({shock1 * 100:.0f}%): Rs. {current_asset_price * (1 + shock1):,.2f}\n"
                    f"- Shock 2 ({shock2 * 100:.0f}%): Rs. {current_asset_price * (1 + shock2):,.2f}"
                )

                mc_probabilities = {}
                if ret_series.std() > 0 and ret_series.shape[0] >= 60:
                    annual_mu_local = ret_series.mean() * PERIODS_PER_YEAR
                    annual_sigma_local = (
                        ret_series.std() * math.sqrt(PERIODS_PER_YEAR)
                    )

                    local_paths = run_monte_carlo_paths(
                        current_price=current_asset_price,
                        annual_mu=annual_mu_local,
                        annual_sigma=annual_sigma_local,
                        days=int(mc_horizon),
                        sims=int(mc_sims),
                        distribution=distribution_choice,
                        df=student_df,
                    )
                    final_prices_local = local_paths[-1, :]
                    target1 = current_asset_price * (1 + shock1)
                    target2 = current_asset_price * (1 + shock2)
                    mc_probabilities[shock1] = float(
                        np.mean(final_prices_local <= target1)
                    )
                    mc_probabilities[shock2] = float(
                        np.mean(final_prices_local <= target2)
                    )
                    st.write(
                        f"- Chance terminal price ≤ Shock 1 ({shock1 * 100:.0f}%): {mc_probabilities[shock1] * 100:.2f}%\n"
                        f"- Chance terminal price ≤ Shock 2 ({shock2 * 100:.0f}%): {mc_probabilities[shock2] * 100:.2f}%"
                    )
                else:
                    st.info(
                        "Not enough data to run Monte Carlo probabilities for this asset."
                    )
                st.info(
                    f"Historical max drawdown of {stats['max_drawdown'] * 100:.1f}% and Sortino {stats['sortino']:.2f} "
                    f"indicate this profile {'is acceptable' if stats['sortino'] > 0.5 else 'remains risky'} "
                    "for capital deployment."
                )

    with st.expander("🔬 Universal Sensitivity Lab (Stress Testing)", expanded=False):
        st.markdown("### 🌪️ Parameter Stability Analysis")
        st.caption("Test the robustness of your strategy by shocking key inputs. A 'Stable' strategy shows smooth changes; a 'Fragile' one jumps wildly.")

        param_to_sweep = st.selectbox(
            "Select Parameter to Shock-Test:",
            [
                "Risk Aversion (λ)",
                "AI Confidence (Raw Prob)",
                "Monte Carlo Probability (History)",
                "Transaction Cost (%)",
                "Trend Score (Signal Strength)"
            ],
            index=0
        )

        if param_to_sweep == "Risk Aversion (λ)":
            start, end, steps = 0.1, 5.0, 50
            current_val = risk_aversion_lambda
            x_label = "Risk Aversion (λ)"
        
        elif param_to_sweep == "AI Confidence (Raw Prob)":
            start, end, steps = 0.0, 1.0, 50
            current_val = raw_ai_prob
            x_label = "AI Estimated Probability"
            
        elif param_to_sweep == "Monte Carlo Probability (History)":
            start, end, steps = 0.30, 0.80, 50
            current_val = final_prob_gain 
            x_label = "Historical Win Probability (MC)"
            
        elif param_to_sweep == "Transaction Cost (%)":
            start, end, steps = 0.0, 0.02, 40
            current_val = transaction_cost
            x_label = "Transaction Cost (Decimal)"
            
        elif param_to_sweep == "Trend Score (Signal Strength)":
            start, end, steps = -1.0, 1.0, 3
            current_val = trend_score
            x_label = "Trend Score (-1=Bear, 1=Bull)"

        if st.button(f"Run Stress Test on {param_to_sweep}"):
            scan_results = []
            
            if steps > 3:
                value_range = np.linspace(start, end, steps)
            else:
                value_range = np.array([-1.0, 0.0, 1.0])

            progress_bar = st.progress(0)
            
            for i, val in enumerate(value_range):
                sim_lambda = risk_aversion_lambda
                sim_ai_prob = ai_prob_active
                sim_mc_prob = final_prob_gain
                sim_cost = transaction_cost
                sim_trend = trend_score
                
                if param_to_sweep == "Risk Aversion (λ)": sim_lambda = val
                elif param_to_sweep == "AI Confidence (Raw Prob)": sim_ai_prob = val
                elif param_to_sweep == "Monte Carlo Probability (History)": sim_mc_prob = val
                elif param_to_sweep == "Transaction Cost (%)": sim_cost = val
                elif param_to_sweep == "Trend Score (Signal Strength)": sim_trend = val

                _, d_size, _, _ = decision_engine(
                    er_blended=blended_er,
                    expected_shortfall=es_annual,
                    prob_gain=sim_mc_prob,         
                    terminal_returns=terminal_returns_display,
                    sim_kurtosis=sim_kurtosis,
                    risk_aversion_lambda=sim_lambda,
                    no_trade_threshold=NO_TRADE_UTILITY_THRESHOLD,
                    regime=regime_result.regime,
                    regime_confidence=regime_result.confidence/100.0,
                    transaction_cost=sim_cost,       
                    win_loss_ratio=win_loss_ratio,
                    kelly_scale=0.5,
                    trend_score=sim_trend,           
                    ai_raw_prob=sim_ai_prob          
                )
                scan_results.append({"Value": val, "Allocation": d_size})
                progress_bar.progress((i + 1) / len(value_range))

            scan_df = pd.DataFrame(scan_results)
            
            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(
                x=scan_df["Value"], 
                y=scan_df["Allocation"],
                mode='lines+markers',
                name='Allocation',
                line=dict(color='#00e676', width=3),
                fill='tozeroy'
            ))

            fig_sens.add_vline(x=current_val, line_dash="dash", line_color="white", annotation_text="Current")

            if "Probability" in param_to_sweep:
                fig_sens.add_vrect(x0=start, x1=0.45, fillcolor="red", opacity=0.1, annotation_text="Guardrail (<45%)")

            fig_sens.update_layout(
                title=f"Robustness Check: {param_to_sweep}",
                xaxis_title=x_label,
                yaxis_title="Recommended Position Size",
                yaxis_tickformat='.0%',
                template="plotly_dark",
                height=450
            )
            st.plotly_chart(fig_sens, use_container_width=True)

            max_alloc = scan_df["Allocation"].max()
            current_alloc_sim = scan_df.iloc[(scan_df['Value'] - current_val).abs().argsort()[:1]]['Allocation'].values[0]

            if max_alloc < 0.01:
                st.error("💀 Dead Zone: This parameter has NO impact because the trade is vetoed by other factors (e.g., Regime or Low Probability).")
            elif "Probability" in param_to_sweep:
                zero_crossings = scan_df[scan_df['Allocation'] > 0.01]['Value'].min()
                if pd.notna(zero_crossings):
                    st.info(f"ℹ️ Break-Even Point: The strategy starts verifying a trade when {x_label} > {zero_crossings:.1%}")
            
            if "Transaction Cost" in param_to_sweep:
                limit_cost = scan_df[scan_df['Allocation'] < 0.01]['Value'].min()
                if pd.notna(limit_cost):
                    st.warning(f"⚠️ Cost Criticality: Strategy becomes unprofitable if transaction costs exceed {limit_cost:.2%}")


if __name__ == "__main__":
    main()
