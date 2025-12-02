import pandas as pd

from nepse_quant_pro.data_io import get_dynamic_data
from nepse_quant_pro.returns import compute_log_returns
from nepse_quant_pro.factors import build_factor_dataframe, fit_signal_model
from nepse_quant_pro.macro_io import (
    get_cached_forex_series,
    build_world_bank_macro_panel,
    load_world_bank_macro,
    build_imf_macro_panel,
    load_imf_macro,
)
from nepse_quant_pro.sector_utils import fetch_sector_benchmark_series
from nepse_quant_pro.sectors import SECTOR_LOOKUP

TOP_EDGE_SYMBOLS = ["CORBL", "KRBL", "BNT", "CHL", "EDBL", "BFC", "CFCL", "JOSHI"]


def _build_macro_bundle(index: pd.DatetimeIndex, world_bank_macro) -> pd.DataFrame:
    components = []
    fx_series = get_cached_forex_series("USD")
    if fx_series is not None:
        components.append(
            pd.DataFrame({"Macro_FX_USD": fx_series.reindex(index).ffill()}, index=index)
        )
    wb_panel = build_world_bank_macro_panel(index, macro_df=world_bank_macro)
    if wb_panel is not None:
        components.append(wb_panel)
    if not components:
        return pd.DataFrame(index=index)
    return pd.concat(components, axis=1).sort_index()


def main():
    world_bank_macro = load_world_bank_macro()
    imf_macro = load_imf_macro()
    for symbol in TOP_EDGE_SYMBOLS:
        try:
            df = get_dynamic_data(symbol)
        except Exception as exc:
            print(f"[{symbol}] Failed to load data: {exc}")
            continue
        if df is None or df.empty or "Close" not in df.columns:
            print(f"[{symbol}] No data available.")
            continue
        df = df.sort_index()
        log_returns = compute_log_returns(df["Close"])
        sector_series = fetch_sector_benchmark_series(symbol)
        macro_bundle = _build_macro_bundle(df.index, world_bank_macro)
        imf_panel = build_imf_macro_panel(df.index, macro_df=imf_macro)
        if imf_panel is not None and not imf_panel.empty:
            macro_bundle = (
                pd.concat([macro_bundle, imf_panel], axis=1).sort_index()
                if macro_bundle is not None and not macro_bundle.empty
                else imf_panel
            )
        sector_name = SECTOR_LOOKUP.get(symbol)
        factors = build_factor_dataframe(
            price_df=df,
            log_returns=log_returns,
            long_window=200,
            horizon=30,
            macro_series=macro_bundle if not macro_bundle.empty else None,
            sector_series=sector_series,
            sector_name=sector_name,
        )
        model = fit_signal_model(
            factors,
            horizon=30,
            cv_folds=5,
            sector_name=sector_name,
        )
        print(f"\n=== {symbol} (Sector: {sector_name or 'Unknown'}) ===")
        print(f"Probability: {model.probability:.3f} | CV Score: {model.accuracy_cv:.3f}")
        print(f"Probability Edge: {model.cv_metrics.get('prob_edge') if model.cv_metrics else None}")
        if model.feature_importance_df is not None and not model.feature_importance_df.empty:
            top_factors = (
                model.feature_importance_df.sort_values("Importance", ascending=False)
                .head(8)
            )
            print(top_factors.to_string(index=False))
        else:
            print("No feature importance available.")


if __name__ == "__main__":
    main()
