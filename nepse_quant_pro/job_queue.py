from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class JobQueue:
    namespace: str
    root: Path = Path("jobs")

    def __post_init__(self):
        self.root = self.root.expanduser().resolve()
        self.namespace_root = self.root / self.namespace
        self.pending_dir = self.namespace_root / "pending"
        self.active_dir = self.namespace_root / "active"
        self.completed_dir = self.namespace_root / "completed"
        self.failed_dir = self.namespace_root / "failed"
        self.lock_path = self.namespace_root / "job.lock"
        for folder in (
            self.pending_dir,
            self.active_dir,
            self.completed_dir,
            self.failed_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)

    def submit_request(self, params: Dict) -> Path:
        run_id = params.get("run_id") or _utc_timestamp()
        payload = dict(params)
        payload["run_id"] = run_id
        filename = f"job_{run_id}.json"
        path = self.pending_dir / filename
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return path

    def _write_status(self, folder: Path, run_id: str, payload: Optional[Dict]):
        folder.mkdir(parents=True, exist_ok=True)
        data = dict(payload or {})
        data.setdefault("run_id", run_id)
        data.setdefault("timestamp", _utc_timestamp())
        path = folder / f"job_{run_id}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)

    def record_completion(self, run_id: str, payload: Optional[Dict] = None):
        self._write_status(self.completed_dir, run_id, payload)

    def record_failure(self, run_id: str, payload: Optional[Dict] = None):
        self._write_status(self.failed_dir, run_id, payload)

    def _acquire_lock(self) -> bool:
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as handle:
                handle.write(_utc_timestamp())
            return True
        except FileExistsError:
            return False

    def _release_lock(self):
        if self.lock_path.exists():
            self.lock_path.unlink()

    @contextmanager
    def job_lock(self):
        acquired = self._acquire_lock()
        if not acquired:
            raise RuntimeError(
                f"Job '{self.namespace}' is already running (lock file present at {self.lock_path})."
            )
        try:
            yield
        finally:
            self._release_lock()


__all__ = ["JobQueue"]
