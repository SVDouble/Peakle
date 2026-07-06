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
            "terrain": settings.terrain.model_copy(update={"grid_width": 96, "grid_height": 72, "seed": 42}),
            "render": settings.render.model_copy(update={"image_width": 320, "image_height": 180}),
        }
    )


@pytest.fixture
def scene(small_settings: AppSettings) -> Scene:
    """Returns a freshly built small scene."""

    return Scene.from_settings(small_settings, provider="demo")  # hermetic: no DEM/OSM
