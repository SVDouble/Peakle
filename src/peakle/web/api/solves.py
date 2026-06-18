"""Per-view solve endpoints (single view, many solves)."""

from __future__ import annotations

from typing import Any

from anyio import to_thread
from fastapi import APIRouter, HTTPException, Request, Response

from peakle.scene.scene import Scene, View
from peakle.web.payloads import solve_payload, solve_summary
from peakle.web.schemas import SolveRequest

router = APIRouter(tags=["solves"])


def _scene(request: Request) -> Scene:
    return request.app.state.scene


def _require_view(scene: Scene, view_id: str) -> View:
    view = scene.views.get(view_id)
    if view is None:
        raise HTTPException(status_code=404, detail=f"unknown view {view_id!r}")
    return view


@router.post("/views/{view_id}/solves", status_code=201)
async def create_solve(view_id: str, body: SolveRequest, request: Request) -> dict[str, Any]:
    """Runs a solver against the view and returns the full trace."""

    scene = _scene(request)
    _require_view(scene, view_id)
    async with request.app.state.scene_lock:
        solve = await to_thread.run_sync(scene.run_solve, view_id, body.strategy, body.params)
    return solve_payload(solve)


@router.get("/views/{view_id}/solves")
async def list_solves(view_id: str, request: Request) -> list[dict[str, Any]]:
    """Returns compact summaries of a view's solves."""

    view = _require_view(_scene(request), view_id)
    return [solve_summary(solve) for solve in view.solves.values()]


@router.get("/views/{view_id}/solves/{solve_id}")
async def get_solve(view_id: str, solve_id: str, request: Request) -> dict[str, Any]:
    """Returns one solve with its full convergence trace."""

    view = _require_view(_scene(request), view_id)
    solve = view.solves.get(solve_id)
    if solve is None:
        raise HTTPException(status_code=404, detail=f"unknown solve {solve_id!r}")
    return solve_payload(solve)


@router.delete("/views/{view_id}/solves/{solve_id}", status_code=204)
async def delete_solve(view_id: str, solve_id: str, request: Request) -> Response:
    """Removes a solve from a view."""

    scene = _scene(request)
    view = _require_view(scene, view_id)
    async with request.app.state.scene_lock:
        view.solves.pop(solve_id, None)
    return Response(status_code=204)
