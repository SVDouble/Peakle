"""Shared test fixtures."""

from __future__ import annotations

import pytest

from peakle.config import AppSettings, load_settings
from peakle.scene.scene import Scene


@pytest.fixture
def small_settings() -> AppSettings:
    """Returns settings with a small, fast scene for tests."""

    settings = load_settings()
    return settings.model_copy(
        update={
            # Pin a small, distinctive terrain independent of the startup default (which is a large
            # 40 km window) — the synthetic solver needs a compact, feature-rich scene to recover.
            "terrain": settings.terrain.model_copy(
                update={"width_m": 14000.0, "height_m": 10000.0, "grid_width": 96, "grid_height": 72, "seed": 42}
            ),
            "render": settings.render.model_copy(update={"image_width": 320, "image_height": 180}),
        }
    )


@pytest.fixture
def scene(small_settings: AppSettings) -> Scene:
    """Returns a freshly built small scene."""

    return Scene.from_settings(small_settings, provider="demo")  # hermetic: no DEM/OSM
