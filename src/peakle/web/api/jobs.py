"""Background job endpoints for batch view solving."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from peakle.scene.scene import Scene
from peakle.web.jobs import JobQueue
from peakle.web.payloads import solve_summary
from peakle.web.schemas import JobCreateRequest

router = APIRouter(tags=["jobs"])


def _queue(request: Request) -> JobQueue:
    return request.app.state.job_queue


def _scene(request: Request) -> Scene:
    return request.app.state.scene


@router.get("/jobs")
async def list_jobs(request: Request, kind: str | None = Query(default=None)) -> list[dict[str, Any]]:
    """Return persisted job history, newest first."""

    return _queue(request).list(kind=kind)


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> dict[str, Any]:
    """Return one background job with task-level status."""

    try:
        return _queue(request).get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}") from None


@router.post("/jobs", status_code=202)
async def create_job(body: JobCreateRequest, request: Request) -> dict[str, Any]:
    """Enqueue solving for one or more views.

    Targets must be loaded scene views. Catalogue-wide pose evaluation belongs to the immutable
    benchmark runner; it must not mutate/rebuild GT records as a side effect of a solve request.
    """

    view_ids = _unique_ids(body.view_ids)
    if not view_ids:
        raise HTTPException(status_code=400, detail="view_ids is empty")

    scene = _scene(request)
    async with request.app.state.scene_lock:
        loaded_view_ids = set(scene.views)
    unloaded = [view_id for view_id in view_ids if view_id not in loaded_view_ids]
    if unloaded:
        raise HTTPException(
            status_code=400,
            detail=f"catalogue targets must be opened as views or run through the benchmark: {unloaded[:5]}",
        )
    if body.strategy is None:
        raise HTTPException(status_code=400, detail="strategy is required")

    strategy = body.strategy
    params = dict(body.params)
    solution_store = request.app.state.solution_store

    def solve(payload: dict[str, Any]) -> dict[str, Any]:
        view_id = str(payload["view_id"])
        if strategy is None:  # guarded above; keeps the closure type-safe
            msg = "strategy is required"
            raise ValueError(msg)
        result = scene.run_solve(view_id, strategy, params)
        solution_store.save(scene.views[view_id], result)
        return {"view_id": view_id, "pose": solve_summary(result)}

    tasks = [{"label": view_id, "payload": {"view_id": view_id}} for view_id in view_ids]
    return _queue(request).submit(
        "solve_views",
        tasks,
        solve,
        params={"view_ids": view_ids, "strategy": strategy, "params": params},
        max_workers=body.max_workers,
    )


def _unique_ids(view_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(view_id.strip() for view_id in view_ids if view_id and view_id.strip()))
