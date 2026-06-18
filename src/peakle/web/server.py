"""Static artifact web server."""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def serve_artifacts(artifact_dir: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serves generated artifacts with Python's standard library server.

    Args:
        artifact_dir: Directory containing `index.html` and generated artifacts.
        host: Interface to bind.
        port: TCP port to bind.
    """

    directory = artifact_dir.resolve()
    if not (directory / "index.html").exists():
        msg = f"{directory} does not contain index.html; run `peakle demo run` first"
        raise FileNotFoundError(msg)

    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"Serving Peakle viewer at {url}")
    print(f"Artifact directory: {directory}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server")
    finally:
        server.server_close()
