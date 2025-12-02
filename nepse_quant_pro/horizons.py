from typing import Optional

from .config import TRIPLE_BARRIER_HORIZON
from .sectors import SECTOR_LOOKUP

SECTOR_HORIZON_OVERRIDES = {
    "Commercial Banks": 45,   # Still medium-term, but more samples
    "Development Banks": 40,
    "Finance": 25,            # High beta, reacts quickly
    "Hydropower": 30,      # Very noisy; shorter horizon helps
    "Manufacturing": 60,
    "Insurance": 60,       # Slow, but don’t starve yourself of folds
    "Microfinance": 40,
    "Investment": 75,         # Long slow movers can stay longer
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
