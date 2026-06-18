"""Synthetic terrain generation."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter

from peakle.domain.coordinates import EARTH_RADIUS_M
from peakle.domain.terrain import TerrainMap, TerrainSpec


class TerrainGenerator:
    """Generates deterministic synthetic terrain.

    Args:
        spec: Validated terrain generation parameters.
    """

    def __init__(self, spec: TerrainSpec) -> None:
        self.spec = spec

    def generate(self) -> TerrainMap:
        """Builds a terrain map with Earth-like coordinates.

        Returns:
            Generated terrain map containing local axes, geodetic grids, and
            elevation data.
        """

        rng = np.random.default_rng(self.spec.seed)
        x_m = np.linspace(-self.spec.width_m / 2.0, self.spec.width_m / 2.0, self.spec.grid_width)
        y_m = np.linspace(
            -self.spec.height_m / 2.0,
            self.spec.height_m / 2.0,
            self.spec.grid_height,
        )
        x_grid, y_grid = np.meshgrid(x_m, y_m)

        relief = self._broad_relief(rng)
        ridges = self._ridge_fields(rng, x_grid, y_grid)
        peaks = self._peak_fields(rng, x_grid, y_grid)
        elevation = relief + ridges + peaks
        elevation = gaussian_filter(elevation, sigma=1.15)
        elevation = self._normalize(elevation)

        lat_grid, lon_grid = self._geodetic_grids(x_grid, y_grid)
        return TerrainMap(
            spec=self.spec,
            x_m=x_m.astype(np.float64),
            y_m=y_m.astype(np.float64),
            elevation_m=elevation.astype(np.float64),
            latitude_deg=lat_grid.astype(np.float64),
            longitude_deg=lon_grid.astype(np.float64),
        )

    def _broad_relief(self, rng: np.random.Generator) -> np.ndarray:
        noise = rng.normal(size=(self.spec.grid_height, self.spec.grid_width))
        relief = gaussian_filter(noise, sigma=(self.spec.grid_height / 8.0, self.spec.grid_width / 8.0))
        return relief * 0.65

    def _ridge_fields(
        self,
        rng: np.random.Generator,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
    ) -> np.ndarray:
        ridges = np.zeros_like(x_grid, dtype=np.float64)
        for _ in range(8):
            center_x = rng.uniform(-self.spec.width_m * 0.35, self.spec.width_m * 0.35)
            center_y = rng.uniform(-self.spec.height_m * 0.35, self.spec.height_m * 0.35)
            theta = rng.uniform(0.0, math.pi)
            length = rng.uniform(self.spec.width_m * 0.16, self.spec.width_m * 0.42)
            width = rng.uniform(self.spec.height_m * 0.035, self.spec.height_m * 0.11)
            amplitude = rng.uniform(0.35, 1.15)

            dx = x_grid - center_x
            dy = y_grid - center_y
            along = dx * math.cos(theta) + dy * math.sin(theta)
            across = -dx * math.sin(theta) + dy * math.cos(theta)
            ridge = np.exp(-0.5 * ((along / length) ** 2 + (across / width) ** 2))
            ridges += amplitude * ridge
        return ridges

    def _peak_fields(
        self,
        rng: np.random.Generator,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
    ) -> np.ndarray:
        peaks = np.zeros_like(x_grid, dtype=np.float64)
        for _ in range(14):
            center_x = rng.uniform(-self.spec.width_m * 0.42, self.spec.width_m * 0.42)
            center_y = rng.uniform(-self.spec.height_m * 0.42, self.spec.height_m * 0.42)
            radius_x = rng.uniform(self.spec.width_m * 0.025, self.spec.width_m * 0.07)
            radius_y = rng.uniform(self.spec.height_m * 0.025, self.spec.height_m * 0.075)
            amplitude = rng.uniform(0.65, 1.9)
            bump = np.exp(-0.5 * (((x_grid - center_x) / radius_x) ** 2 + ((y_grid - center_y) / radius_y) ** 2))
            peaks += amplitude * bump
        return peaks

    def _normalize(self, elevation: np.ndarray) -> np.ndarray:
        elevation = elevation - float(np.min(elevation))
        elevation = elevation / max(float(np.max(elevation)), 1e-9)
        span = self.spec.max_elevation_m - self.spec.min_elevation_m
        return self.spec.min_elevation_m + elevation * span

    def _geodetic_grids(self, x_grid: np.ndarray, y_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        origin = self.spec.origin
        lat_grid = origin.latitude_deg + np.degrees(y_grid / EARTH_RADIUS_M)
        lon_grid = origin.longitude_deg + np.degrees(
            x_grid / (EARTH_RADIUS_M * math.cos(math.radians(origin.latitude_deg)))
        )
        return lat_grid, lon_grid
