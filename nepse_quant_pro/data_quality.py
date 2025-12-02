from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import pandas as pd

NEPAL_TRADING_DAY = pd.offsets.CustomBusinessDay(weekmask="Sun Mon Tue Wed Thu")


@dataclass
class SymbolHealth:
    symbol: str
    trading_days: int
    missing_close_pct: float
    flat_close_pct: float
    median_volume: float
    median_liquidity: float
    zero_volume_pct: float
    first_trade: Optional[pd.Timestamp]
    healthy: bool
    reasons: List[str]
    tier: str = "standard"

    def as_dict(self) -> dict:
        out = asdict(self)
        if isinstance(out.get("first_trade"), (pd.Timestamp, pd.DatetimeIndex)):
            out["first_trade"] = out["first_trade"].isoformat() if out["first_trade"] is not None else None
        return out


def _scrub_flat_runs(series: pd.Series, max_run: int = 3) -> pd.Series:
    if series is None or series.empty:
        return series
    diffs = series.diff().fillna(0.0).abs() < 1e-8
    groups = diffs.groupby((~diffs).cumsum()).transform("sum")
    mask = groups >= max_run
    scrubbed = series.copy()
    scrubbed[mask] = np.nan
    return scrubbed.ffill().bfill()


def align_to_trading_calendar(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return price_df
    df = price_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    start, end = df.index.min(), df.index.max()
    if pd.isna(start) or pd.isna(end):
        return df
    calendar_index = pd.date_range(start=start, end=end, freq=NEPAL_TRADING_DAY)
    aligned = df.reindex(calendar_index)

    if "Close" in aligned.columns:
        aligned["Close"] = _scrub_flat_runs(aligned["Close"])

    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in aligned.columns]
    if price_cols:
        aligned[price_cols] = aligned[price_cols].ffill().bfill()

    if "Volume" in aligned.columns:
        aligned["Volume"] = aligned["Volume"].fillna(0.0)

    other_cols = [c for c in aligned.columns if c not in price_cols and c != "Volume"]
    if other_cols:
        aligned[other_cols] = aligned[other_cols].ffill().bfill()

    return aligned


def resample_to_weekly(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return price_df
    agg = price_df.resample("W-FRI").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    agg = agg.dropna(subset=["Close"])
    if "Volume" in agg.columns:
        agg["Volume"] = agg["Volume"].fillna(0.0)
    return agg


def resample_event_bars(
    price_df: pd.DataFrame,
    price_sigma: float = 1.0,
    volume_sigma: float = 1.0,
    lookback: int = 30,
    min_bars: int = 120,
) -> pd.DataFrame:
    if price_df is None or price_df.empty or "Close" not in price_df.columns:
        return pd.DataFrame()
    df = price_df.copy()
    closes = df["Close"].astype(float)
    volumes = df.get("Volume", pd.Series(0.0, index=df.index)).astype(float)
    log_returns = np.log(closes).diff().abs()
    ret_std = log_returns.rolling(lookback).std().fillna(log_returns.std()).replace(0.0, log_returns.std())
    price_trigger = log_returns >= (price_sigma * ret_std).fillna(0.0)
    vol_mean = volumes.rolling(lookback).mean()
    vol_std = volumes.rolling(lookback).std().replace(0.0, np.nan)
    volume_trigger = volumes >= (vol_mean + volume_sigma * vol_std).fillna(np.inf)
    trigger = price_trigger | volume_trigger
    if not trigger.any():
        return resample_to_weekly(price_df)
    event_df = df.loc[trigger]
    if len(event_df) < min_bars:
        return resample_to_weekly(price_df)
    event_df = event_df.copy()
    event_df["Volume"] = event_df.get("Volume", 0.0).fillna(0.0)
    return event_df


def compute_symbol_health(
    symbol: str,
    price_df: pd.DataFrame,
    min_trading_days: int = 350,
    max_flat_pct: float = 0.65,
    max_missing_pct: float = 0.35,
    emerging_min_days: int = 200,
    emerging_flat_pct: float = 0.70,
    emerging_missing_pct: float = 0.40,
    min_median_volume: float = 200.0,
    min_median_liquidity: float = 50_000.0,
    max_zero_volume_pct: float = 0.45,
) -> SymbolHealth:
    reasons: List[str] = []
    if price_df is None or price_df.empty:
        reasons.append("No price history available.")
        return SymbolHealth(
            symbol=symbol,
            trading_days=0,
            missing_close_pct=1.0,
            flat_close_pct=1.0,
            median_volume=0.0,
            median_liquidity=0.0,
            zero_volume_pct=1.0,
            first_trade=None,
            healthy=False,
            reasons=reasons,
            tier="none",
        )

    df = price_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if "Close" not in df.columns:
        reasons.append("Close column missing.")
        vol_series = df.get("Volume", pd.Series(dtype=float))
        median_volume = float(vol_series.median()) if not vol_series.empty else 0.0
        if np.isnan(median_volume):
            median_volume = 0.0
        return SymbolHealth(
            symbol=symbol,
            trading_days=len(df),
            missing_close_pct=1.0,
            flat_close_pct=1.0,
            median_volume=median_volume,
            median_liquidity=0.0,
            zero_volume_pct=1.0,
            first_trade=df.index.min() if len(df.index) else None,
            healthy=False,
            reasons=reasons,
            tier="none",
        )

    closes = df["Close"].astype(float)
    volumes_raw = df.get("Volume", pd.Series(0.0, index=df.index)).astype(float).fillna(0.0)
    valid_closes = closes.dropna()
    trading_days = int(valid_closes.shape[0])

    returns = closes.pct_change().abs().fillna(0.0)
    active_mask = volumes_raw > 0.0
    if active_mask.any():
        flat_close_pct = float(((returns < 1e-8) & active_mask).sum() / active_mask.sum())
    else:
        flat_close_pct = 0.0
    aligned = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq=NEPAL_TRADING_DAY))
    missing_close_pct = float(aligned["Close"].isna().mean()) if "Close" in aligned else 1.0
    vol_series = volumes_raw
    median_volume = float(vol_series.median()) if not vol_series.empty else 0.0
    if np.isnan(median_volume):
        median_volume = 0.0
    zero_volume_pct = float((vol_series <= 0.0).mean()) if not vol_series.empty else 1.0
    liquidity_series = (df["Close"].astype(float) * vol_series).replace([np.inf, -np.inf], np.nan)
    median_liquidity = float(liquidity_series.median()) if not liquidity_series.empty else 0.0
    if np.isnan(median_liquidity):
        median_liquidity = 0.0
    first_trade = df.index.min()

    tier = "standard"
    effective_min_days = min_trading_days
    effective_flat_pct = max_flat_pct
    effective_missing_pct = max_missing_pct
    if trading_days < min_trading_days:
        if trading_days >= emerging_min_days:
            tier = "emerging"
            effective_min_days = emerging_min_days
            effective_flat_pct = max(max_flat_pct, emerging_flat_pct)
            effective_missing_pct = max(max_missing_pct, emerging_missing_pct)
        else:
            tier = "insufficient"

    if trading_days < effective_min_days:
        reasons.append(f"Insufficient trading history ({trading_days} < {effective_min_days}).")
    if flat_close_pct > effective_flat_pct:
        reasons.append(f"Flat sessions exceed limit ({flat_close_pct:.1%} > {effective_flat_pct:.1%}).")
    if missing_close_pct > effective_missing_pct:
        reasons.append(f"Missing close percentage too high ({missing_close_pct:.1%} > {effective_missing_pct:.1%}).")

    safe_zero_vol_limit = max_zero_volume_pct
    if zero_volume_pct > (safe_zero_vol_limit + 1e-6):
        reasons.append(f"Too many zero-volume days ({zero_volume_pct:.1%} > {safe_zero_vol_limit:.1%}).")

    if median_volume < min_median_volume:
        reasons.append(f"Volume effectively zero (Median {median_volume:,.0f} < {min_median_volume:,.0f}).")
    if median_liquidity < min_median_liquidity:
        reasons.append(
            f"Liquidity too low for sizing ({median_liquidity:,.0f} < {min_median_liquidity:,.0f})."
        )

    healthy = len(reasons) == 0
    return SymbolHealth(
        symbol=symbol,
        trading_days=trading_days,
        missing_close_pct=missing_close_pct,
        flat_close_pct=flat_close_pct,
        median_volume=median_volume,
        median_liquidity=median_liquidity,
        zero_volume_pct=zero_volume_pct,
        first_trade=first_trade,
        healthy=healthy,
        reasons=reasons,
        tier=tier,
    )


__all__ = [
    "SymbolHealth",
    "align_to_trading_calendar",
    "compute_symbol_health",
    "resample_to_weekly",
    "NEPAL_TRADING_DAY",
]
