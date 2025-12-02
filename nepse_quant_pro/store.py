import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def _default_run_id() -> str:
    """
    Generate a human-friendly run identifier.

    Format: YYYY-MM-DDTHH-MM (e.g., 2025-11-30T17-29)
    """
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M")


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (pd.Series, list, tuple)):
        return list(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value


@dataclass
class JobStore:
    namespace: str
    root: Path = field(default_factory=lambda: Path(os.environ.get("ARTIFACTS_DIR", "artifacts")).expanduser().resolve())
    run_id: str = field(default_factory=_default_run_id)

    def __post_init__(self):
        self.path = self.root / self.namespace / self.run_id
        self.symbol_dir = self.path / "symbols"
        self.meta_dir = self.path / "meta"
        self.path.mkdir(parents=True, exist_ok=True)
        self.symbol_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def write_metadata(self, payload: Dict, name: str = "params.json") -> Path:
        data = {k: _json_safe(v) for k, v in payload.items()}
        path = self.meta_dir / name
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        return path

    def write_dataframe(self, df: pd.DataFrame, name: str = "summary.parquet", fmt: str = "parquet") -> Path:
        path = self.path / name
        if fmt == "parquet":
            df.to_parquet(path, index=False)
        elif fmt == "csv":
            df.to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported fmt {fmt}")
        return path

    def write_symbol_record(self, symbol: str, record: Dict) -> Path:
        clean = {k: _json_safe(v) for k, v in record.items()}
        file_path = self.symbol_dir / f"{symbol.upper()}.json"
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(clean, handle, indent=2, sort_keys=True)
        return file_path

    def read_summary(self, name: str = "summary.parquet") -> Optional[pd.DataFrame]:
        path = self.path / name
        if not path.exists():
            return None
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        return None

    def list_symbol_records(self):
        return sorted(self.symbol_dir.glob("*.json"))


__all__ = ["JobStore"]
