"""Live viewer server.

Builds the in-memory scene from settings and serves the FastAPI viewer with
uvicorn. Views and alignment solves are computed on demand; nothing is read
from precomputed artifacts.
"""

from __future__ import annotations

import uvicorn

from peakle.config import AppSettings
from peakle.scene.scene import Scene
from peakle.web.app import create_app


def serve(settings: AppSettings, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serves the workbench.

    Args:
        settings: Application settings used to build the initial scene.
        host: Interface to bind.
        port: TCP port to bind.
    """

    scene = Scene.from_settings(settings)
    app = create_app(scene)
    url = f"http://{host}:{port}/"
    print(f"Serving Peakle workbench at {url}")
    print(f"Scene: {len(scene.peaks)} peaks, seed {scene.config.seed} (place cameras to create views)")
    uvicorn.run(app, host=host, port=port, log_level=settings.log_level.lower())
