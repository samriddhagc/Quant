import numpy as np
import pandas as pd

from nepse_quant_pro.factors import build_factor_dataframe, fit_signal_model
from nepse_quant_pro.returns import compute_log_returns
from market_scanner import analyze_task


def _make_price_fixture(rows: int = 400) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    trend = np.linspace(0, 5, rows)
    seasonal = 2 * np.sin(np.linspace(0, 20, rows))
    noise = np.sin(np.linspace(0, 50, rows)) * 0.5
    close = 100 + trend + seasonal + noise
    base_open = close * (1 + 0.001)
    high = close * 1.01
    low = close * 0.99
    volume = 50000 + np.linspace(0, 1000, rows)
    data = pd.DataFrame(
        {
            "Open": base_open,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    return data


def test_fit_signal_model_exposes_expected_fields():
    df = _make_price_fixture()
    log_returns = compute_log_returns(df["Close"])
    factors = build_factor_dataframe(df, log_returns, long_window=180, horizon=30, macro_series=None)
    result = fit_signal_model(factors, horizon=30, cv_folds=3)
    assert result.probability is not None
    assert result.accuracy_cv is not None
    assert result.feature_importance_df is not None


def test_analyze_task_runs_on_fixture_without_attribute_errors():
    df = _make_price_fixture()
    outcome = analyze_task("TEST", df)
    assert outcome is None or isinstance(outcome, dict)
