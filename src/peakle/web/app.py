"""FastAPI application factory for the Peakle workbench."""

from __future__ import annotations

import asyncio
from importlib import resources
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
from starlette.responses import Response

from peakle.scene.scene import Scene
from peakle.web.api import scene as scene_api
from peakle.web.api import solves as solves_api
from peakle.web.api import views as views_api


class _NoCacheStaticFiles(StaticFiles):
    """Serves the viewer with caching disabled.

    The workbench is a live dev tool whose HTML/JS/CSS change often; browser
    caching otherwise serves stale assets until a hard refresh. We always send a
    full, non-cacheable response so a normal reload picks up edits.
    """

    def is_not_modified(self, response_headers: Headers, request_headers: Headers) -> bool:
        return False

    async def get_response(self, path: str, scope: Any) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response


def create_app(scene: Scene) -> FastAPI:
    """Builds the workbench application around a mutable in-memory scene.

    Args:
        scene: Pre-built scene state holding terrain, peaks, and views.

    Returns:
        A FastAPI app serving the static viewer and the JSON API.
    """

    app = FastAPI(title="Peakle", version="0.1.0")
    app.state.scene = scene
    app.state.scene_lock = asyncio.Lock()

    app.include_router(scene_api.router, prefix="/api")
    app.include_router(views_api.router, prefix="/api")
    app.include_router(solves_api.router, prefix="/api")

    static_dir = resources.files("peakle.web") / "static"
    app.mount("/", _NoCacheStaticFiles(directory=str(static_dir), html=True), name="static")
    return app
