from typing import Optional

from .config import TRIPLE_BARRIER_HORIZON
from .sectors import SECTOR_LOOKUP

SECTOR_HORIZON_OVERRIDES = {
    "Commercial Banks": 60,   # Macro & credit-cycle driven, slow drift
    "Development Banks": 45,  # Mid-cap, some policy/theme momentum
    "Finance": 30,            # High beta, reacts fast to sentiment
    "Hydropower": 40,         # Volatile, story/narrative cycles need a bit more than 30
    "Manufacturing": 60,      # Fundamentals change slowly
    "Insurance": 75,          # Very slow repricing of expectations
    "Microfinance": 45,       # Noisy but with medium-term themes
    "Investment": 90,         # Holding-company / investment vehicles, very slow
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
