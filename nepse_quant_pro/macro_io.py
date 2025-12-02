import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import requests

NRB_BASE_URL = "https://www.nrb.org.np/api/forex/v1/rates"
CACHE_DIR = Path(".cache/forex")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
WORLD_BANK_BASE = "https://api.worldbank.org/v2"
WORLD_BANK_CACHE = CACHE_DIR / "world_bank_nepal.csv"
IMF_BASE = "https://www.imf.org/external/datamapper/api/v1"
IMF_CACHE = CACHE_DIR / "imf_nepal_core_macro.csv"


def _fetch_nrb_rates(
    symbol: str,
    start_date: str,
    end_date: str,
    per_page: int = 100,
) -> pd.Series:
    page = 1
    records = []
    while True:
        params = {
            "from": start_date,
            "to": end_date,
            "page": page,
            "per_page": per_page,
        }
        resp = requests.get(NRB_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", {}).get("payload") or []
        for entry in data:
            date_str = entry.get("date")
            rates = entry.get("rates") or []
            match = next(
                (
                    r
                    for r in rates
                    if (r.get("currency") or {}).get("iso3", "").upper()
                    == symbol.upper()
                ),
                None,
            )
            if match:
                buy = match.get("buy")
                sell = match.get("sell")
                try:
                    buy_val = float(buy)
                    sell_val = float(sell)
                    mid = (buy_val + sell_val) / 2.0
                except (TypeError, ValueError):
                    continue
                records.append((pd.to_datetime(date_str), mid))
        pagination = payload.get("pagination") or {}
        links = pagination.get("links") or {}
        if not links.get("next"):
            break
        page += 1

    if not records:
        return pd.Series(dtype=float)
    series = pd.Series(
        data=[r[1] for r in records],
        index=[r[0] for r in records],
        name=f"{symbol.upper()}_NPR",
    )
    return series.sort_index()


def get_cached_forex_series(
    symbol: str = "USD",
    start_date: Optional[str] = "2015-01-01",
    force_refresh: bool = False,
) -> Optional[pd.Series]:
    cache_file = CACHE_DIR / f"{symbol.upper()}_forex.csv"
    cached_series = None
    if cache_file.exists():
        try:
            cached_series = (
                pd.read_csv(cache_file, parse_dates=["date"], index_col="date")
                .squeeze("columns")
                .astype(float)
            )
        except Exception:
            cached_series = None

    need_refresh = force_refresh or cached_series is None
    if cached_series is not None:
        latest = cached_series.index.max()
        if latest is not None:
            if latest >= pd.Timestamp(datetime.utcnow().date() - timedelta(days=3)):
                need_refresh = False

    auto_fetch = os.environ.get("NRB_FOREX_AUTO_FETCH", "1") != "0"
    if "PYTEST_CURRENT_TEST" in os.environ:
        auto_fetch = False
    if not need_refresh or not auto_fetch:
        return cached_series

    try:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        fetched = _fetch_nrb_rates(symbol, start_date or "2015-01-01", end_date)
        if fetched.empty:
            return cached_series
        if cached_series is not None:
            combined = pd.concat([cached_series, fetched])
            combined = combined[~combined.index.duplicated(keep="last")]
            fetched = combined.sort_index()
        fetched.to_frame("rate").to_csv(
            cache_file,
            index_label="date",
        )
        return fetched
    except Exception as exc:
        logging.warning("Failed to refresh NRB FX data: %s", exc)
        return cached_series


def _fetch_world_bank_indicator(
    country: str,
    indicator: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    url = f"{WORLD_BANK_BASE}/country/{country}/indicator/{indicator}"
    params = {
        "format": "json",
        "per_page": 20000,
        "date": f"{start_year}:{end_year}",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError(f"Unexpected World Bank response for {indicator}: {data}")
    rows = []
    for entry in data[1]:
        year = entry.get("date")
        value = entry.get("value")
        if year is None:
            continue
        try:
            year_int = int(year)
        except ValueError:
            continue
        rows.append({"year": year_int, "value": value})
    return pd.DataFrame(rows)


def _load_world_bank_macro(force_refresh: bool = False) -> pd.DataFrame:
    if WORLD_BANK_CACHE.exists() and not force_refresh:
        try:
            return pd.read_csv(WORLD_BANK_CACHE)
        except Exception:
            pass
    auto_fetch = os.environ.get("WORLD_BANK_AUTO_FETCH", "1") != "0"
    if "PYTEST_CURRENT_TEST" in os.environ:
        auto_fetch = False
    if not auto_fetch:
        return pd.read_csv(WORLD_BANK_CACHE) if WORLD_BANK_CACHE.exists() else pd.DataFrame()
    country = "NPL"
    start_year = 2015
    end_year = datetime.now().year
    indicators = {
        "gdp_current_usd": "NY.GDP.MKTP.CD",
        "gdp_per_capita_usd": "NY.GDP.PCAP.CD",
        "cpi_2010_100": "FP.CPI.TOTL",
        "inflation_cpi_pct": "FP.CPI.TOTL.ZG",
    }
    dfs = []
    for col_name, code in indicators.items():
        df_indicator = _fetch_world_bank_indicator(country, code, start_year, end_year)
        df_indicator = df_indicator.rename(columns={"value": col_name})
        dfs.append(df_indicator)
    if not dfs:
        return pd.DataFrame()
    macro = dfs[0]
    for df_indicator in dfs[1:]:
        macro = macro.merge(df_indicator, on="year", how="outer")
    macro = macro.sort_values("year").reset_index(drop=True)
    macro.to_csv(WORLD_BANK_CACHE, index=False)
    return macro


def load_world_bank_macro(force_refresh: bool = False) -> pd.DataFrame:
    return _load_world_bank_macro(force_refresh=force_refresh).copy()


def build_world_bank_macro_panel(
    index: pd.DatetimeIndex,
    macro_df: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    if macro_df is None:
        macro_df = _load_world_bank_macro()
    if macro_df is None or macro_df.empty:
        return None
    panel = pd.DataFrame(index=index)
    date_index = pd.to_datetime(macro_df["year"].astype(str) + "-12-31")
    for column in macro_df.columns:
        if column == "year":
            continue
        series = pd.Series(macro_df[column].values, index=date_index).sort_index()
        series = series.ffill()
        aligned = (
            series.reindex(index, method="ffill")
            if not series.empty
            else pd.Series(index=index, dtype=float)
        )
        panel[f"Macro_{column}"] = aligned
    return panel.ffill().bfill()


def _fetch_imf_macro(indicators: Dict[str, str], start_year: int, end_year: int) -> pd.DataFrame:
    period_list = [str(year) for year in range(start_year, end_year + 1)]
    periods_param = ",".join(period_list)
    rows = []
    for indicator, label in indicators.items():
        url = f"{IMF_BASE}/{indicator}/NPL"
        resp = requests.get(url, params={"periods": periods_param}, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        values_root = payload.get("values", {}).get(indicator, {})
        country_values = values_root.get("NPL") or {}
        for year_str, val in country_values.items():
            if not year_str.isdigit():
                continue
            year = int(year_str)
            if val is None:
                continue
            rows.append({"year": year, label: float(val)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.groupby("year").first().reset_index()
    return df.sort_values("year")


def _load_imf_macro(force_refresh: bool = False) -> pd.DataFrame:
    if IMF_CACHE.exists() and not force_refresh:
        try:
            return pd.read_csv(IMF_CACHE)
        except Exception:
            pass
    auto_fetch = os.environ.get("IMF_AUTO_FETCH", "1") != "0"
    if "PYTEST_CURRENT_TEST" in os.environ:
        auto_fetch = False
    if not auto_fetch:
        return pd.read_csv(IMF_CACHE) if IMF_CACHE.exists() else pd.DataFrame()
    current_year = datetime.now().year
    start_year = max(2000, current_year - 10)
    indicators = {
        "NGDPD": "gdp_current_usd_imf",
        "NGDP_RPCH": "real_gdp_growth_imf",
        "PCPIPCH": "inflation_cpi_imf",
        "LUR": "unemployment_rate_imf",
        "BCA_NGDPD": "current_account_pct_gdp_imf",
        "GGXWDG_NGDP": "govt_debt_pct_gdp_imf",
    }
    df = _fetch_imf_macro(indicators, start_year, current_year)
    df.to_csv(IMF_CACHE, index=False)
    return df


def load_imf_macro(force_refresh: bool = False) -> pd.DataFrame:
    return _load_imf_macro(force_refresh=force_refresh).copy()


def build_imf_macro_panel(
    index: pd.DatetimeIndex,
    macro_df: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    if macro_df is None:
        macro_df = _load_imf_macro()
    if macro_df is None or macro_df.empty:
        return None
    panel = pd.DataFrame(index=index)
    date_index = pd.to_datetime(macro_df["year"].astype(str) + "-12-31")
    for column in macro_df.columns:
        if column == "year":
            continue
        series = pd.Series(macro_df[column].values, index=date_index).sort_index()
        series = series.ffill()
        aligned = (
            series.reindex(index, method="ffill")
            if not series.empty
            else pd.Series(index=index, dtype=float)
        )
        panel[f"Macro_{column}"] = aligned
    return panel.ffill().bfill()
