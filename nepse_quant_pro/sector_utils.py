from typing import Optional

import pandas as pd

from .data_io import get_dynamic_data
from .sectors import SECTOR_GROUPS, SECTOR_LOOKUP


def fetch_sector_benchmark_series(symbol: str, max_peers: int = 8) -> Optional[pd.Series]:
    """
    Build an equal-weighted sector benchmark for the given symbol by averaging
    peer closing prices. Returns None if peers cannot be gathered quickly.
    """
    if not symbol:
        return None
    sector = SECTOR_LOOKUP.get(symbol.upper())
    if not sector:
        return None
    peers = [s for s in SECTOR_GROUPS[sector] if s != symbol.upper()]
    closes = []
    for peer in peers:
        try:
            df = get_dynamic_data(peer)
        except Exception:
            continue
        if df is None or df.empty or "Close" not in df.columns:
            continue
        closes.append(df["Close"].astype(float).rename(peer))
        if len(closes) >= max_peers:
            break
    if not closes:
        return None
    joined = pd.concat(closes, axis=1).sort_index()
    if joined.empty:
        return None
    return joined.mean(axis=1, skipna=True)
