"""View CRUD and rendered-image endpoints."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from anyio import to_thread
from fastapi import APIRouter, HTTPException, Query, Request, Response
from PIL import Image, ImageOps

from peakle.domain.camera import CameraExtrinsics
from peakle.domain.contours import ImagePoint, SkylineContour
from peakle.domain.coordinates import LocalPoint
from peakle.localize.extract import extract_candidates
from peakle.scene.scene import Scene, View
from peakle.scene.state import build_intrinsics
from peakle.web.payloads import view_payload
from peakle.web.schemas import ViewCreateRequest, ViewPatchRequest

router = APIRouter(tags=["views"])
PHOTO_MAX_UPLOAD_BYTES = 12 * 1024 * 1024
PHOTO_MAX_WIDTH_PX = 1280
MIN_SKYLINE_COVERAGE = 0.10


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


@router.post("/views/from-photo", status_code=201)
async def create_view_from_photo(
    request: Request,
    lat_deg: float = Query(ge=-90.0, le=90.0),
    lon_deg: float = Query(ge=-180.0, le=180.0),
    horizontal_fov_deg: float = Query(gt=1.0, lt=179.0),
    eye_height_m: float = Query(default=2.0, ge=0.0, le=5000.0),
    label: str | None = Query(default=None, max_length=120),
) -> dict[str, Any]:
    """Uploads a geotagged photo and materializes it as a solvable, photo-backed view."""

    image_bytes = await _read_photo_body(request)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty photo upload")
    scene = _scene(request)
    async with request.app.state.scene_lock:
        view = await to_thread.run_sync(
            _create_photo_view,
            scene,
            image_bytes,
            lat_deg,
            lon_deg,
            horizontal_fov_deg,
            eye_height_m,
            label,
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


@router.post("/views/{view_id}/duplicate", status_code=201)
async def duplicate_view(view_id: str, request: Request, body: dict[str, str] | None = None) -> dict[str, Any]:
    """Duplicate a view (any kind) under a new id + optional custom label, without its solves."""

    scene = _scene(request)
    _require_view(scene, view_id)
    async with request.app.state.scene_lock:
        dup = await to_thread.run_sync(scene.duplicate_view, view_id, (body or {}).get("label"))
    return view_payload(dup)


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


def _reject_oversized_photo(request: Request) -> None:
    content_length = request.headers.get("content-length")
    if content_length is None:
        return
    try:
        upload_bytes = int(content_length)
    except ValueError:
        return
    if upload_bytes > PHOTO_MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="photo upload exceeds 12 MB limit")


async def _read_photo_body(request: Request) -> bytes:
    _reject_oversized_photo(request)
    chunks = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > PHOTO_MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="photo upload exceeds 12 MB limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _create_photo_view(
    scene: Scene,
    image_bytes: bytes,
    lat_deg: float,
    lon_deg: float,
    horizontal_fov_deg: float,
    eye_height_m: float,
    label: str | None,
) -> View:
    photo = _load_photo(image_bytes)
    contour = _extract_photo_contour(photo)
    scene.focus_geo(lat_deg, lon_deg)
    intrinsics = build_intrinsics(photo.width, photo.height, horizontal_fov_deg)
    position = LocalPoint(
        east_m=0.0,
        north_m=0.0,
        up_m=scene.terrain.elevation_at(0.0, 0.0) + eye_height_m,
    )
    extrinsics = CameraExtrinsics(position=position, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0)
    return scene.add_backed_view(
        intrinsics,
        extrinsics,
        contour,
        photo,
        source="photo",
        label=label or "Uploaded photo",
    )


def _load_photo(image_bytes: bytes) -> Image.Image:
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="unsupported image upload") from exc
    if image.width > PHOTO_MAX_WIDTH_PX:
        height = max(1, round(image.height * (PHOTO_MAX_WIDTH_PX / image.width)))
        image = image.resize((PHOTO_MAX_WIDTH_PX, height), Image.Resampling.LANCZOS)
    return image


def _extract_photo_contour(photo: Image.Image) -> SkylineContour:
    rgb = np.asarray(photo, dtype=np.uint8)
    candidates = sorted(extract_candidates(rgb).values(), key=lambda c: c.coverage, reverse=True)
    if not candidates or candidates[0].coverage < MIN_SKYLINE_COVERAGE:
        raise HTTPException(status_code=422, detail="no usable skyline found in photo")
    rows = candidates[0].rows
    points = [
        ImagePoint(x_px=float(col), y_px=float(rows[col])) for col in range(photo.width) if np.isfinite(rows[col])
    ]
    return SkylineContour(image_width_px=photo.width, image_height_px=photo.height, points=points, source="photo")
