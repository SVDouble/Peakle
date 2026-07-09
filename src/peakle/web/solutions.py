"""Persistent pose-solution storage for web-created solves."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from peakle.scene.scene import Solve, View


class SolutionStore:
    """Persists solver poses so GT-backed view solutions survive server restarts."""

    def __init__(self, store_dir: Path) -> None:
        self._store_dir = store_dir
        self._path = store_dir / "solutions.json"
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {"views": {}}
        with self._lock:
            self._load_locked()

    def save(self, view: View, solve: Solve) -> None:
        """Persist or replace one solve for a view."""

        key = solution_key(view)
        with self._lock:
            bucket = self._data.setdefault("views", {}).setdefault(key, {"solves": []})
            rows = [row for row in bucket.get("solves", []) if row.get("id") != solve.id]
            rows.append(solve.model_dump(mode="json"))
            bucket["solves"] = rows
            bucket["label"] = view.label
            bucket["source"] = view.source
            bucket["gt_name"] = view.gt_name
            self._persist_locked()

    def load(self, view: View) -> list[Solve]:
        """Load persisted solves for a view."""

        key = solution_key(view)
        with self._lock:
            rows = list(self._data.get("views", {}).get(key, {}).get("solves", []))
        solves: list[Solve] = []
        for row in rows:
            try:
                solves.append(Solve.model_validate(row))
            except ValueError:
                continue
        return solves

    def remove(self, view: View, solve_id: str) -> None:
        """Remove a persisted solve from a view."""

        key = solution_key(view)
        with self._lock:
            bucket = self._data.get("views", {}).get(key)
            if not bucket:
                return
            bucket["solves"] = [row for row in bucket.get("solves", []) if row.get("id") != solve_id]
            self._persist_locked()

    def _load_locked(self) -> None:
        if not self._path.exists():
            return
        try:
            loaded = json.loads(self._path.read_text())
        except OSError, json.JSONDecodeError:
            return
        if isinstance(loaded, dict) and isinstance(loaded.get("views"), dict):
            self._data = loaded

    def _persist_locked(self) -> None:
        self._store_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, indent=1))
        tmp.replace(self._path)


def solution_key(view: View) -> str:
    """Stable solution key for a view."""

    if view.gt_name:
        return f"gt:{view.gt_name}"
    return f"view:{view.id}"
