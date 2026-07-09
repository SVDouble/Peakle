"""FastAPI application factory for the Peakle workbench."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
from starlette.responses import RedirectResponse, Response

from peakle.scene.scene import Scene
from peakle.web.api import gtlab as gtlab_api
from peakle.web.api import jobs as jobs_api
from peakle.web.api import scene as scene_api
from peakle.web.api import solves as solves_api
from peakle.web.api import views as views_api
from peakle.web.jobs import JobQueue
from peakle.web.solutions import SolutionStore

BASE = Path(__file__).resolve().parents[3]


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


def create_app(scene: Scene, job_store_dir: Path | None = None, solution_store_dir: Path | None = None) -> FastAPI:
    """Builds the workbench application around a mutable in-memory scene.

    Args:
        scene: Pre-built scene state holding terrain, peaks, and views.

    Returns:
        A FastAPI app serving the static viewer and the JSON API.
    """

    job_queue = JobQueue(job_store_dir or BASE / "local/derived/web_jobs")
    solution_store = SolutionStore(solution_store_dir or BASE / "local/derived/web_solutions")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            job_queue.shutdown()

    app = FastAPI(title="Peakle", version="0.1.0", lifespan=lifespan)
    app.state.scene = scene
    app.state.scene_lock = asyncio.Lock()
    app.state.job_queue = job_queue
    app.state.solution_store = solution_store

    app.include_router(scene_api.router, prefix="/api")
    app.include_router(gtlab_api.router, prefix="/api")
    app.include_router(jobs_api.router, prefix="/api")
    app.include_router(views_api.router, prefix="/api")
    app.include_router(solves_api.router, prefix="/api")

    @app.get("/gt", include_in_schema=False)
    async def gt_lab() -> RedirectResponse:  # the GT-dataset debugger page
        return RedirectResponse("/gtlab.html")

    static_dir = resources.files("peakle.web") / "static"
    app.mount("/", _NoCacheStaticFiles(directory=str(static_dir), html=True), name="static")
    return app
