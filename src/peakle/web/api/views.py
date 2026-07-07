"""View CRUD and rendered-image endpoints."""

from __future__ import annotations

import io
from typing import Any

from anyio import to_thread
from fastapi import APIRouter, HTTPException, Request, Response

from peakle.scene.scene import Scene, View
from peakle.web.payloads import view_payload
from peakle.web.schemas import ViewCreateRequest, ViewPatchRequest

router = APIRouter(tags=["views"])


def _scene(request: Request) -> Scene:
    return request.app.state.scene


def _require_view(scene: Scene, view_id: str) -> View:
    view = scene.views.get(view_id)
    if view is None:
        raise HTTPException(status_code=404, detail=f"unknown view {view_id!r}")
    return view


@router.post("/views", status_code=201)
async def create_view(body: ViewCreateRequest, request: Request) -> dict[str, Any]:
    """Places a camera, renders its image, and detects its contour."""

    scene = _scene(request)
    async with request.app.state.scene_lock:
        view = await to_thread.run_sync(
            scene.create_view,
            body.east_m,
            body.north_m,
            body.yaw_deg,
            body.pitch_deg,
            body.eye_height_m,
            body.label,
        )
    return view_payload(view)


@router.get("/views")
async def list_views(request: Request) -> list[dict[str, Any]]:
    """Returns all views with solve summaries."""

    return [view_payload(view) for view in _scene(request).views.values()]


@router.get("/views/{view_id}")
async def get_view(view_id: str, request: Request) -> dict[str, Any]:
    """Returns one view with solve summaries."""

    return view_payload(_require_view(_scene(request), view_id))


@router.patch("/views/{view_id}")
async def patch_view(view_id: str, body: ViewPatchRequest, request: Request) -> dict[str, Any]:
    """Edits a view's pose/label; re-renders and clears stale solves."""

    scene = _scene(request)
    _require_view(scene, view_id)
    async with request.app.state.scene_lock:
        view = await to_thread.run_sync(_apply_patch, scene, view_id, body)
    return view_payload(view)


@router.delete("/views/{view_id}", status_code=204)
async def delete_view(view_id: str, request: Request) -> Response:
    """Removes a view."""

    scene = _scene(request)
    async with request.app.state.scene_lock:
        scene.delete_view(view_id)
    return Response(status_code=204)


@router.get("/views/{view_id}/image")
async def get_view_image(view_id: str, request: Request) -> Response:
    """Returns the rendered view image as PNG."""

    view = _require_view(_scene(request), view_id)
    buffer = io.BytesIO()
    view.render_arrays.image.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


@router.get("/views/{view_id}/photo")
async def get_view_photo(view_id: str, request: Request) -> Response:
    """Returns a GT-derived view's reference photograph as JPEG (404 for a synthetic view)."""

    view = _require_view(_scene(request), view_id)
    if view.reference_photo is None:
        raise HTTPException(status_code=404, detail=f"view {view_id!r} has no reference photo")
    buffer = io.BytesIO()
    view.reference_photo.convert("RGB").save(buffer, format="JPEG", quality=88)
    return Response(content=buffer.getvalue(), media_type="image/jpeg")


def _apply_patch(scene: Scene, view_id: str, body: ViewPatchRequest) -> View:
    return scene.update_view(view_id, **body.model_dump(exclude_unset=True))
