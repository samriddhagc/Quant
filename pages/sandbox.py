import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from nepse_quant_pro.data_io import get_dynamic_data
from nepse_quant_pro.returns import compute_log_returns
from nepse_quant_pro.factors import (
    build_factor_dataframe,
    fit_signal_model,
    build_factor_contribution_chart,
    build_signal_history_chart,
)
from nepse_quant_pro.sector_utils import fetch_sector_benchmark_series
from nepse_quant_pro.sectors import SECTOR_LOOKUP
from nepse_quant_pro.macro_io import (
    get_cached_forex_series,
    build_world_bank_macro_panel,
    build_imf_macro_panel,
)
from nepse_quant_pro.visuals import build_trend_figure, build_mc_paths_figure
from nepse_quant_pro.regime import detect_regime
from nepse_quant_pro.risk_engine import run_monte_carlo_paths


st.title("Research Sandbox (Interactive)")
st.warning(
    "This page runs all computations inline. Use it for experimentation only—production dashboards rely on the "
    "pre-computed artifacts."
)

symbol = st.text_input("Symbol", value="NABIL")
horizon = st.slider("Prediction Horizon (days)", 20, 120, 60, step=5)
cv_folds = st.slider("CV Folds", 3, 10, 5, step=1)

run_button = st.button("Run Factor Fit")

if run_button:
    try:
        data = get_dynamic_data(symbol.upper().strip())
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        data = None

    if data is None or data.empty:
        st.error("No price data available for this symbol.")
    else:
        st.success(f"Loaded {len(data)} rows. Building factor stack…")
        data = data.sort_index()
        log_returns = compute_log_returns(data["Close"])
        sector_series = fetch_sector_benchmark_series(symbol.upper().strip())
        sector_name = SECTOR_LOOKUP.get(symbol.upper().strip())
        macro_parts = []
        fx_series = get_cached_forex_series("USD")
        if fx_series is not None:
            macro_parts.append(
                pd.DataFrame(
                    {"Macro_FX_USD": fx_series.reindex(data.index).ffill()},
                    index=data.index,
                )
            )
        wb_panel = build_world_bank_macro_panel(data.index)
        if wb_panel is not None:
            macro_parts.append(wb_panel)
        imf_panel = build_imf_macro_panel(data.index)
        if imf_panel is not None:
            macro_parts.append(imf_panel)
        macro_df = (
            pd.concat(macro_parts, axis=1).sort_index() if macro_parts else None
        )

        factors = build_factor_dataframe(
            data,
            log_returns,
            long_window=80,
            horizon=horizon,
            macro_series=macro_df,
            sector_series=sector_series,
            sector_name=sector_name,
        )
        if factors.empty:
            st.error("Factor dataframe ended up empty. Try adjusting the horizon.")
        else:
            model = fit_signal_model(
                factors,
                horizon=horizon,
                cv_folds=cv_folds,
                sector_name=sector_name,
            )
            st.subheader("Model Diagnostics")
            diag_cols = st.columns(3)
            diag_cols[0].metric(
                "Probability", f"{(model.probability or 0.5) * 100:.2f}%"
            )
            diag_cols[1].metric(
                "CV Accuracy",
                f"{(model.accuracy_cv or 0.0) * 100:.2f}%",
            )
            diag_cols[2].metric(
                "CV Std",
                f"{(model.accuracy_std or 0.0) * 100:.2f}%",
            )
            if model.warnings:
                for warning in model.warnings:
                    st.warning(warning)
            if model.message:
                st.info(model.message)

            st.plotly_chart(
                build_trend_figure(
                    pd.DataFrame({"Close": data["Close"]}),
                    short_window=30,
                    long_window=80,
                ),
                use_container_width=True,
            )

            if model.feature_importance_df is not None and not model.feature_importance_df.empty:
                st.markdown("**Feature Importance**")
                st.dataframe(
                    model.feature_importance_df.sort_values("Importance", ascending=False).head(10),
                    use_container_width=True,
                )
            if model.contributions is not None:
                st.plotly_chart(
                    build_factor_contribution_chart(model.contributions),
                    use_container_width=True,
                )
            if model.history is not None and not model.history.empty:
                st.plotly_chart(
                    build_signal_history_chart(model.history),
                    use_container_width=True,
                )

            st.subheader("Ad-hoc Monte Carlo")
            mc_cols = st.columns(3)
            mc_horizon = mc_cols[0].number_input(
                "MC Horizon (days)", min_value=10, max_value=250, value=60, step=5
            )
            mc_sims = mc_cols[1].number_input(
                "Simulations", min_value=500, max_value=10000, value=2000, step=500
            )
            mc_vol_scale = mc_cols[2].slider(
                "Volatility Multiplier", min_value=0.5, max_value=3.0, value=1.0, step=0.1
            )
            if st.button("Run Monte Carlo", key="sandbox_mc"):
                log_ret = compute_log_returns(data["Close"])
                mu = float(log_ret.mean()) * 252
                sigma = float(log_ret.std()) * np.sqrt(252) * mc_vol_scale
                paths = run_monte_carlo_paths(
                    current_price=float(data["Close"].iloc[-1]),
                    annual_mu=mu,
                    annual_sigma=sigma,
                    days=int(mc_horizon),
                    sims=int(mc_sims),
                )
                st.plotly_chart(
                    build_mc_paths_figure(paths, "Sandbox Monte Carlo Paths"),
                    use_container_width=True,
                )
                terminal_returns = paths[-1, :] / paths[0, 0] - 1.0
                st.write(
                    f"Mean terminal return: {terminal_returns.mean() * 100:.2f}% "
                    f"| 5th percentile: {np.percentile(terminal_returns, 5) * 100:.2f}%"
                )
