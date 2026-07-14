"""Persistent background job queue for web-triggered batch work."""

from __future__ import annotations

import builtins
import copy
import json
import threading
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
RUNNABLE_STATUSES = {"queued", "running"}

JobWorker = Callable[[dict[str, Any]], dict[str, Any] | None]


class JobQueue:
    """Small persisted queue for jobs made of independent tasks.

    The queue persists job history and task results to JSON. Live worker callables are process-local,
    so jobs left queued/running across a server restart are marked failed on startup instead of
    pretending they are still active.
    """

    def __init__(self, store_dir: Path, *, max_workers: int = 4) -> None:
        self._store_dir = store_dir
        self._path = store_dir / "jobs.json"
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="peakle-job")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._workers: dict[str, JobWorker] = {}
        self._active: dict[str, int] = {}
        with self._lock:
            self._load_locked()
            if self._mark_interrupted_locked():
                self._persist_locked()

    def submit(
        self,
        kind: str,
        tasks: Iterable[dict[str, Any]],
        worker: JobWorker,
        *,
        params: dict[str, Any] | None = None,
        max_workers: int = 2,
    ) -> dict[str, Any]:
        """Create a job and start scheduling its queued tasks."""

        task_rows = [
            {
                "id": f"task-{index:03d}",
                "label": str(task.get("label") or f"Task {index}"),
                "payload": copy.deepcopy(task.get("payload") or {}),
                "status": "queued",
                "created_at": _now(),
                "started_at": None,
                "finished_at": None,
                "result": None,
                "error": None,
            }
            for index, task in enumerate(tasks, start=1)
        ]
        if not task_rows:
            msg = "cannot submit a job without tasks"
            raise ValueError(msg)

        with self._lock:
            job_id = f"job-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
            job = {
                "id": job_id,
                "kind": kind,
                "status": "queued",
                "created_at": _now(),
                "updated_at": _now(),
                "finished_at": None,
                "params": copy.deepcopy(params or {}),
                "max_workers": max(1, int(max_workers)),
                "tasks": task_rows,
            }
            self._refresh_counts_locked(job)
            self._jobs[job_id] = job
            self._workers[job_id] = worker
            self._active[job_id] = 0
            self._persist_locked()
            snapshot = _snapshot(job)

        self._schedule(job_id)
        return snapshot

    def list(self, *, kind: str | None = None) -> builtins.list[dict[str, Any]]:
        """Return newest jobs first."""

        with self._lock:
            jobs = [job for job in self._jobs.values() if kind is None or job.get("kind") == kind]
            return [_snapshot(job) for job in sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)]

    def get(self, job_id: str) -> dict[str, Any]:
        """Return one job snapshot."""

        with self._lock:
            return _snapshot(self._jobs[job_id])

    def shutdown(self) -> None:
        """Stop accepting/running background tasks during app shutdown."""

        self._executor.shutdown(wait=False, cancel_futures=True)

    def _schedule(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            worker = self._workers.get(job_id)
            if job is None or worker is None or job.get("status") not in RUNNABLE_STATUSES:
                return

            launched = False
            active = self._active.get(job_id, 0)
            queued = [task for task in job["tasks"] if task["status"] == "queued"]
            while queued and active < int(job["max_workers"]):
                task = queued.pop(0)
                task["status"] = "running"
                task["started_at"] = _now()
                task["error"] = None
                job["status"] = "running"
                job["updated_at"] = _now()
                active += 1
                self._active[job_id] = active
                self._executor.submit(self._run_task, job_id, task["id"], copy.deepcopy(task["payload"]), worker)
                launched = True

            if launched:
                self._refresh_counts_locked(job)
                self._persist_locked()

    def _run_task(self, job_id: str, task_id: str, payload: dict[str, Any], worker: JobWorker) -> None:
        try:
            result = worker(payload) or {}
        except Exception as exc:  # noqa: BLE001 - one failed task must not kill the whole job
            self._finish_task(job_id, task_id, status="failed", result=None, error=str(exc)[:500])
        else:
            self._finish_task(job_id, task_id, status="completed", result=result, error=None)

    def _finish_task(
        self,
        job_id: str,
        task_id: str,
        *,
        status: str,
        result: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        should_schedule = False
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            task = next((row for row in job["tasks"] if row["id"] == task_id), None)
            if task is None:
                return
            task["status"] = status
            task["finished_at"] = _now()
            task["result"] = result
            task["error"] = error
            self._active[job_id] = max(0, self._active.get(job_id, 1) - 1)

            queued = any(row["status"] == "queued" for row in job["tasks"])
            running = any(row["status"] == "running" for row in job["tasks"])
            if queued or running:
                job["status"] = "running" if running else "queued"
                should_schedule = queued
            else:
                failed = any(row["status"] == "failed" for row in job["tasks"])
                job["status"] = "failed" if failed else "completed"
                job["finished_at"] = _now()
                self._workers.pop(job_id, None)
                self._active.pop(job_id, None)

            job["updated_at"] = _now()
            self._refresh_counts_locked(job)
            self._persist_locked()

        if should_schedule:
            self._schedule(job_id)

    def _load_locked(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
        except OSError, json.JSONDecodeError:
            return
        rows = raw.get("jobs", raw) if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            return
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("id"), str):
                self._refresh_counts_locked(row)
                self._jobs[row["id"]] = row

    def _mark_interrupted_locked(self) -> bool:
        changed = False
        for job in self._jobs.values():
            if job.get("status") not in RUNNABLE_STATUSES:
                continue
            for task in job.get("tasks", []):
                if task.get("status") in RUNNABLE_STATUSES:
                    task["status"] = "failed"
                    task["finished_at"] = _now()
                    task["error"] = "server restarted before this task finished"
                    changed = True
            job["status"] = "failed"
            job["finished_at"] = _now()
            job["updated_at"] = _now()
            self._refresh_counts_locked(job)
            changed = True
        return changed

    def _refresh_counts_locked(self, job: dict[str, Any]) -> None:
        tasks = job.get("tasks", [])
        job["total"] = len(tasks)
        job["done"] = sum(1 for task in tasks if task.get("status") == "completed")
        job["failed"] = sum(1 for task in tasks if task.get("status") == "failed")
        job["running_tasks"] = [
            {"id": task.get("id"), "label": task.get("label")} for task in tasks if task.get("status") == "running"
        ]

    def _persist_locked(self) -> None:
        self._store_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        rows = sorted(self._jobs.values(), key=lambda item: item.get("created_at", ""))
        tmp.write_text(json.dumps({"jobs": rows}, indent=1))
        tmp.replace(self._path)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _snapshot(job: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(job)
