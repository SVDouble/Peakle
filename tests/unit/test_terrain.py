"""Terrain generation tests."""

import numpy as np

from peakle.config import load_settings
from peakle.terrain.generator import TerrainGenerator


def test_terrain_generation_is_deterministic_for_seed() -> None:
    spec = load_settings().terrain.model_copy(
        update={
            "seed": 7,
            "grid_width": 48,
            "grid_height": 40,
        }
    )

    first = TerrainGenerator(spec).generate()
    second = TerrainGenerator(spec).generate()

    np.testing.assert_allclose(first.elevation_m, second.elevation_m)
    assert first.elevation_m.shape == (40, 48)
    assert np.isclose(first.elevation_m.min(), spec.min_elevation_m)
    assert np.isclose(first.elevation_m.max(), spec.max_elevation_m)
