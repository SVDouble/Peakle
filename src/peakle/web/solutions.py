"""Persistent pose-solution storage for web-created solves."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from peakle.scene.scene import Solve, View

SOLUTION_SCHEMA_VERSION = 2


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
        fingerprint = solution_fingerprint(view)
        with self._lock:
            bucket = self._data.setdefault("views", {}).setdefault(key, {"solves": []})
            compatible = (
                bucket.get("schema_version") == SOLUTION_SCHEMA_VERSION
                and bucket.get("view_fingerprint") == fingerprint
            )
            existing = bucket.get("solves", []) if compatible else []
            rows = [row for row in existing if row.get("id") != solve.id]
            rows.append(solve.model_dump(mode="json"))
            bucket["solves"] = rows
            bucket["label"] = view.label
            bucket["source"] = view.source
            bucket["gt_name"] = view.gt_name
            bucket["schema_version"] = SOLUTION_SCHEMA_VERSION
            bucket["view_fingerprint"] = fingerprint
            self._persist_locked()

    def load(self, view: View) -> list[Solve]:
        """Load persisted solves for a view."""

        key = solution_key(view)
        with self._lock:
            bucket = self._data.get("views", {}).get(key, {})
            if bucket.get("schema_version") != SOLUTION_SCHEMA_VERSION:
                return []
            if bucket.get("view_fingerprint") != solution_fingerprint(view):
                return []
            rows = list(bucket.get("solves", []))
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


def solution_fingerprint(view: View) -> str:
    """Fingerprint every input whose change makes a persisted pose incomparable."""

    payload = {
        "schema_version": SOLUTION_SCHEMA_VERSION,
        "source": view.source,
        "gt_name": view.gt_name,
        "intrinsics": view.intrinsics.model_dump(mode="json"),
        "image_camera": view.image_camera.model_dump(mode="json"),
        "reference": view.true_extrinsics.model_dump(mode="json") if view.true_extrinsics else None,
        "prior": view.prior.model_dump(mode="json") if view.prior else None,
        "contour": view.contour.model_dump(mode="json"),
        "evidence_contours": {
            key: contour.model_dump(mode="json") for key, contour in sorted(view.evidence_contours.items())
        },
        "evidence_metadata": view.evidence_metadata,
        "default_evidence_source": view.default_evidence_source,
        "pitch_comparable": view.pitch_comparable,
        "terrain_fingerprint": view.terrain_fingerprint,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
