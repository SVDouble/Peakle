"""FastAPI application factory for the Peakle workbench."""

from __future__ import annotations

import asyncio
from importlib import resources

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from peakle.scene.scene import Scene
from peakle.web.api import scene as scene_api
from peakle.web.api import solves as solves_api
from peakle.web.api import views as views_api


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
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    return app
