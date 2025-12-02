from typing import Optional

from .config import TRIPLE_BARRIER_HORIZON
from .sectors import SECTOR_LOOKUP

SECTOR_HORIZON_OVERRIDES = {
    "Commercial Banks": 40,
    "Development Banks": 35,
    "Finance": 35,
    "Hydropower": 20,
    "Manufacturing": 45,
    "Insurance": 50,
    "Microfinance": 30,
    "Investment": 45,
}


def resolve_symbol_horizon(symbol: Optional[str], default: Optional[int] = None) -> int:
    base = default if default is not None else TRIPLE_BARRIER_HORIZON
    if not symbol:
        return base
    sector = SECTOR_LOOKUP.get(symbol.upper())
    if sector is None:
        return base
    return SECTOR_HORIZON_OVERRIDES.get(sector, base)


__all__ = ["resolve_symbol_horizon", "SECTOR_HORIZON_OVERRIDES"]
