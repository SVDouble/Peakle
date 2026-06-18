"""Scene configuration and read-only scene data endpoints."""

from __future__ import annotations

from typing import Any

from anyio import to_thread
from fastapi import APIRouter, Request

from peakle.optimization.solve import AVAILABLE_STRATEGIES
from peakle.scene.scene import Scene
from peakle.web.payloads import peaks_payload, scene_payload, terrain_payload
from peakle.web.schemas import SceneConfigRequest

router = APIRouter(tags=["scene"])


def _scene(request: Request) -> Scene:
    return request.app.state.scene


@router.get("/scene")
async def get_scene(request: Request) -> dict[str, Any]:
    """Returns scene config, intrinsics, providers, and available strategies."""

    return scene_payload(_scene(request))


@router.put("/scene/config")
async def put_scene_config(body: SceneConfigRequest, request: Request) -> dict[str, Any]:
    """Rebuilds the map and intrinsics from new config, clearing views."""

    scene = _scene(request)
    async with request.app.state.scene_lock:
        await to_thread.run_sync(
            scene.set_config,
            body.provider,
            body.seed,
            body.image_width,
            body.image_height,
            body.horizontal_fov_deg,
            body.default_strategy,
        )
    return scene_payload(scene)


@router.get("/terrain")
async def get_terrain(request: Request) -> dict[str, Any]:
    """Returns the terrain grid for client rendering."""

    return terrain_payload(_scene(request).terrain)


@router.get("/peaks")
async def get_peaks(request: Request) -> list[dict[str, Any]]:
    """Returns the detected peaks."""

    return peaks_payload(_scene(request).peaks)


@router.get("/algorithms")
async def get_algorithms() -> dict[str, Any]:
    """Returns available solver strategies."""

    return {"strategies": AVAILABLE_STRATEGIES}
