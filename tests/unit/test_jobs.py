"""Persistent background job queue tests."""

from __future__ import annotations

import json
import threading
import time

from peakle.web.jobs import JobQueue


def _wait_job(queue: JobQueue, job_id: str, timeout_s: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = queue.get(job_id)
        if job["status"] in {"completed", "failed", "cancelled"}:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish")


def test_job_queue_runs_tasks_in_parallel_and_persists_results(tmp_path) -> None:
    queue = JobQueue(tmp_path, max_workers=2)
    active = 0
    peak_active = 0
    lock = threading.Lock()

    def worker(payload: dict) -> dict:
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return {"value": payload["value"] * 2}

    try:
        job = queue.submit(
            "test",
            [{"label": str(i), "payload": {"value": i}} for i in range(4)],
            worker,
            max_workers=2,
        )
        final = _wait_job(queue, job["id"])
    finally:
        queue.shutdown()

    assert final["status"] == "completed"
    assert final["done"] == 4
    assert peak_active == 2

    loaded = JobQueue(tmp_path, max_workers=1)
    try:
        persisted = loaded.get(job["id"])
    finally:
        loaded.shutdown()

    assert persisted["status"] == "completed"
    assert [task["result"]["value"] for task in persisted["tasks"]] == [0, 2, 4, 6]


def test_job_queue_marks_running_jobs_failed_after_restart(tmp_path) -> None:
    (tmp_path / "jobs.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "job-1",
                        "kind": "test",
                        "status": "running",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "updated_at": "2026-01-01T00:00:00+00:00",
                        "finished_at": None,
                        "params": {},
                        "max_workers": 2,
                        "tasks": [
                            {"id": "task-001", "label": "a", "status": "running"},
                            {"id": "task-002", "label": "b", "status": "queued"},
                        ],
                    }
                ]
            }
        )
    )

    queue = JobQueue(tmp_path, max_workers=1)
    try:
        job = queue.get("job-1")
    finally:
        queue.shutdown()

    assert job["status"] == "failed"
    assert job["failed"] == 2
    assert {task["status"] for task in job["tasks"]} == {"failed"}
    assert all(task["error"] == "server restarted before this task finished" for task in job["tasks"])
