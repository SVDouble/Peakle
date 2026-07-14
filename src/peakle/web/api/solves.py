"""Per-view solve endpoints (single view, many solves)."""

from __future__ import annotations

from typing import Any

from anyio import to_thread
from fastapi import APIRouter, HTTPException, Request, Response

from peakle.scene.scene import Scene, View
from peakle.web.payloads import solve_payload, solve_summary
from peakle.web.pose_layers import POSE_LAYER_NAMES, render_pose_layer
from peakle.web.schemas import SolveRequest

router = APIRouter(tags=["solves"])


def _scene(request: Request) -> Scene:
    return request.app.state.scene


def _require_view(scene: Scene, view_id: str) -> View:
    view = scene.views.get(view_id)
    if view is None:
        raise HTTPException(status_code=404, detail=f"unknown view {view_id!r}")
    return view


def _pose_extrinsics(view: View, pose_key: str):
    if pose_key == "truth":
        if view.true_extrinsics is None:
            raise HTTPException(status_code=404, detail=f"view {view.id!r} has no baseline pose")
        return view.true_extrinsics
    if pose_key.startswith("solve:"):
        solve_id = pose_key.removeprefix("solve:")
        solve = view.solves.get(solve_id)
        if solve is None:
            raise HTTPException(status_code=404, detail=f"unknown solve {solve_id!r}")
        return solve.estimate.extrinsics
    raise HTTPException(status_code=404, detail=f"unknown pose {pose_key!r}")


@router.post("/views/{view_id}/solves", status_code=201)
async def create_solve(view_id: str, body: SolveRequest, request: Request) -> dict[str, Any]:
    """Runs a solver against the view and returns the full trace."""

    scene = _scene(request)
    _require_view(scene, view_id)
    async with request.app.state.scene_lock:
        try:
            solve = await to_thread.run_sync(scene.run_solve, view_id, body.strategy, body.params)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        request.app.state.solution_store.save(scene.views[view_id], solve)
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


@router.get("/views/{view_id}/poses/{pose_key:path}/layers/{layer}.png")
async def get_pose_layer(view_id: str, pose_key: str, layer: str, request: Request) -> Response:
    """Returns a derived debug layer for any concrete pose of a view."""

    if layer not in POSE_LAYER_NAMES:
        raise HTTPException(status_code=404, detail=f"unknown pose layer {layer!r}")
    scene = _scene(request)
    view = _require_view(scene, view_id)
    extrinsics = _pose_extrinsics(view, pose_key)
    try:
        png = await to_thread.run_sync(render_pose_layer, scene, view, extrinsics, layer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.delete("/views/{view_id}/solves/{solve_id}", status_code=204)
async def delete_solve(view_id: str, solve_id: str, request: Request) -> Response:
    """Removes a solve from a view."""

    scene = _scene(request)
    view = _require_view(scene, view_id)
    async with request.app.state.scene_lock:
        view.solves.pop(solve_id, None)
        request.app.state.solution_store.remove(view, solve_id)
    return Response(status_code=204)
