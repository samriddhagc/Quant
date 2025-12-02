from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import pandas as pd


def _namespace_root(namespace: str, artifacts_dir: Path) -> Path:
    return artifacts_dir / namespace


def _parse_run_id(name: str) -> Optional[datetime]:
    """
    Attempts to parse a run identifier into a datetime.

    Supports both legacy names (e.g. 20251130T172443Z) and the new
    ISO-like names (e.g. 2025-11-30T21-42).
    """
    cleaned = name.replace("-", "")
    cleaned = cleaned.rstrip("Z")
    patterns = ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M")
    for pattern in patterns:
        try:
            return datetime.strptime(cleaned, pattern)
        except ValueError:
            continue
    return None


def list_runs(namespace: str, artifacts_dir: str = "artifacts") -> List[str]:
    root = _namespace_root(namespace, Path(artifacts_dir).expanduser().resolve())
    if not root.exists():
        return []
    runs = [d.name for d in root.iterdir() if d.is_dir()]
    runs.sort(key=lambda name: (_parse_run_id(name) or datetime.min, name))
    return runs


def latest_run(namespace: str, artifacts_dir: str = "artifacts") -> Optional[str]:
    runs = list_runs(namespace, artifacts_dir=artifacts_dir)
    return runs[-1] if runs else None


def _symbol_path(namespace: str, symbol: str, run_id: str, artifacts_dir: str) -> Path:
    root = _namespace_root(namespace, Path(artifacts_dir).expanduser().resolve())
    return root / run_id / "symbols" / f"{symbol.upper()}.json"


def load_symbol_artifact(
    namespace: str,
    symbol: str,
    artifacts_dir: str = "artifacts",
    run_id: Optional[str] = None,
) -> Optional[Dict]:
    """
    Loads the stored artifact for a symbol from the specified namespace.

    If run_id is None, the most recent run is used. Returns None if the artifact
    does not exist in the requested run.
    """
    selected_run = run_id or latest_run(namespace, artifacts_dir=artifacts_dir)
    if not selected_run:
        return None
    symbol_path = _symbol_path(namespace, symbol, selected_run, artifacts_dir)
    if not symbol_path.exists():
        return None
    with open(symbol_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run_metadata(
    namespace: str,
    run_id: Optional[str] = None,
    artifacts_dir: str = "artifacts",
    name: str = "params.json",
) -> Optional[Dict]:
    """
    Loads a job-level metadata JSON (e.g., params.json) for a namespace/run.
    """
    selected = run_id or latest_run(namespace, artifacts_dir=artifacts_dir)
    if not selected:
        return None
    meta_path = (
        _namespace_root(namespace, Path(artifacts_dir).expanduser().resolve())
        / selected
        / "meta"
        / name
    )
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class ArtifactSelector:
    namespace: str
    artifacts_dir: Path = Path("artifacts")

    def available_runs(self) -> List[str]:
        return list_runs(self.namespace, artifacts_dir=str(self.artifacts_dir))

    def latest(self) -> Optional[str]:
        return latest_run(self.namespace, artifacts_dir=str(self.artifacts_dir))

    def load(self, symbol: str, run_id: Optional[str] = None) -> Optional[Dict]:
        return load_symbol_artifact(
            self.namespace, symbol, artifacts_dir=str(self.artifacts_dir), run_id=run_id
        )


def _series_from_payload(payload) -> Optional[pd.Series]:
    if not payload:
        return None
    data = {}
    for entry in payload:
        name = entry.get("name")
        value = entry.get("value")
        if name is not None:
            data[name] = value
    if not data:
        return None
    return pd.Series(data)


def _history_from_payload(payload) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    df = pd.DataFrame(payload)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.set_index("date")
    rename_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower == "probability":
            rename_map[col] = "Probability"
        elif lower.replace("_", " ") == "realized positive":
            rename_map[col] = "Realized Positive"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _feature_importance_from_payload(payload) -> Optional[pd.DataFrame]:
    if not payload:
        return None
    df = pd.DataFrame(payload)
    if df.empty:
        return None
    return df


def artifact_to_factor_model(artifact: Dict) -> Optional[SimpleNamespace]:
    if not artifact:
        return None
    meta = artifact.get("meta", {})
    history_df = _history_from_payload(artifact.get("signal_history"))
    factor_obj = SimpleNamespace(
        probability=meta.get("probability"),
        accuracy_cv=meta.get("cv_score"),
        accuracy_std=meta.get("cv_std"),
        warnings=meta.get("warnings", []),
        message=meta.get("message"),
        cv_metrics=artifact.get("cv_metrics") or {},
        coefficients=_series_from_payload(artifact.get("coefficients")),
        latest_factors=_series_from_payload(artifact.get("latest_factors")),
        feature_importance_df=_feature_importance_from_payload(
            artifact.get("feature_importance")
        ),
        contributions=_series_from_payload(artifact.get("contributions")),
        history=history_df,
        position_multiplier=artifact.get("position_multiplier"),
        sector_rank_snapshot=_series_from_payload(artifact.get("sector_rank_snapshot")),
        validation_metrics=artifact.get("validation_metrics"),
        disabled_reason=artifact.get("disabled_reason"),
        model_tier=artifact.get("model_tier", "Unrated"),
        trust_score=artifact.get("trust_score", 1.0),
    )
    return factor_obj


__all__ = [
    "list_runs",
    "latest_run",
    "load_symbol_artifact",
    "ArtifactSelector",
    "artifact_to_factor_model",
    "load_run_metadata",
]
