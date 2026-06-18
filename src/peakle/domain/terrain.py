"""Terrain domain models."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from peakle.domain.coordinates import GeoPoint, LocalFrame, LocalPoint


class TerrainSpec(BaseModel):
    """Configuration for deterministic synthetic terrain generation.

    Attributes:
        origin: Geodetic origin for the local terrain frame.
        width_m: East-west terrain width in meters.
        height_m: North-south terrain height in meters.
        grid_width: Number of east-west samples.
        grid_height: Number of north-south samples.
        min_elevation_m: Minimum normalized generated elevation.
        max_elevation_m: Maximum normalized generated elevation.
        seed: Random seed.
    """

    origin: GeoPoint
    width_m: float = Field(gt=100.0)
    height_m: float = Field(gt=100.0)
    grid_width: int = Field(ge=32)
    grid_height: int = Field(ge=32)
    min_elevation_m: float
    max_elevation_m: float
    seed: int

    @model_validator(mode="after")
    def _validate_elevation_range(self) -> TerrainSpec:
        if self.max_elevation_m <= self.min_elevation_m:
            msg = "max_elevation_m must be greater than min_elevation_m"
            raise ValueError(msg)
        return self


class TerrainMap(BaseModel):
    """Generated terrain grid and coordinate metadata.

    Attributes:
        spec: Source terrain generation specification.
        x_m: Easting samples in meters.
        y_m: Northing samples in meters.
        elevation_m: Elevation grid with shape `(grid_height, grid_width)`.
        latitude_deg: Latitude grid with the same shape as elevation.
        longitude_deg: Longitude grid with the same shape as elevation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: TerrainSpec
    x_m: NDArray[np.float64]
    y_m: NDArray[np.float64]
    elevation_m: NDArray[np.float64]
    latitude_deg: NDArray[np.float64]
    longitude_deg: NDArray[np.float64]

    @model_validator(mode="after")
    def _validate_arrays(self) -> TerrainMap:
        expected_shape = (self.spec.grid_height, self.spec.grid_width)
        if self.elevation_m.shape != expected_shape:
            msg = f"elevation_m shape must be {expected_shape}"
            raise ValueError(msg)
        if self.latitude_deg.shape != expected_shape or self.longitude_deg.shape != expected_shape:
            msg = "latitude_deg and longitude_deg must match elevation_m shape"
            raise ValueError(msg)
        if self.x_m.shape != (self.spec.grid_width,) or self.y_m.shape != (self.spec.grid_height,):
            msg = "x_m and y_m must be one-dimensional coordinate axes"
            raise ValueError(msg)
        for name, array in self._array_items().items():
            if not np.all(np.isfinite(array)):
                msg = f"{name} contains non-finite values"
                raise ValueError(msg)
        return self

    def _array_items(self) -> dict[str, NDArray[np.float64]]:
        return {
            "x_m": self.x_m,
            "y_m": self.y_m,
            "elevation_m": self.elevation_m,
            "latitude_deg": self.latitude_deg,
            "longitude_deg": self.longitude_deg,
        }

    @property
    def frame(self) -> LocalFrame:
        """Returns the local coordinate frame."""

        return LocalFrame(origin=self.spec.origin)

    def point_at_index(self, row: int, col: int) -> LocalPoint:
        """Builds a local point for a terrain grid index.

        Args:
            row: Northing grid index.
            col: Easting grid index.

        Returns:
            Local terrain point at the requested grid cell.
        """

        return LocalPoint(
            east_m=float(self.x_m[col]),
            north_m=float(self.y_m[row]),
            up_m=float(self.elevation_m[row, col]),
        )

    def geo_at_index(self, row: int, col: int) -> GeoPoint:
        """Builds a geodetic point for a terrain grid index."""

        return GeoPoint(
            latitude_deg=float(self.latitude_deg[row, col]),
            longitude_deg=float(self.longitude_deg[row, col]),
            elevation_m=float(self.elevation_m[row, col]),
        )

    def elevation_at(self, east_m: float, north_m: float) -> float:
        """Interpolates terrain elevation at local coordinates.

        Args:
            east_m: Easting in meters.
            north_m: Northing in meters.

        Returns:
            Bilinearly interpolated elevation in meters.
        """

        col = float(np.interp(east_m, self.x_m, np.arange(self.x_m.size)))
        row = float(np.interp(north_m, self.y_m, np.arange(self.y_m.size)))
        row0 = int(np.clip(np.floor(row), 0, self.y_m.size - 1))
        col0 = int(np.clip(np.floor(col), 0, self.x_m.size - 1))
        row1 = min(row0 + 1, self.y_m.size - 1)
        col1 = min(col0 + 1, self.x_m.size - 1)
        dy = row - row0
        dx = col - col0
        z00 = self.elevation_m[row0, col0]
        z01 = self.elevation_m[row0, col1]
        z10 = self.elevation_m[row1, col0]
        z11 = self.elevation_m[row1, col1]
        north0 = (1.0 - dx) * z00 + dx * z01
        north1 = (1.0 - dx) * z10 + dx * z11
        return float((1.0 - dy) * north0 + dy * north1)

    def flattened_points(self, stride: int = 1) -> NDArray[np.float64]:
        """Returns terrain grid points as an `(N, 3)` array.

        Args:
            stride: Positive subsampling stride for both grid axes.

        Returns:
            Array of local `(east, north, up)` points.
        """

        if stride < 1:
            msg = "stride must be positive"
            raise ValueError(msg)
        x_grid, y_grid = np.meshgrid(self.x_m[::stride], self.y_m[::stride])
        z_grid = self.elevation_m[::stride, ::stride]
        return np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel())).astype(np.float64)

    def metadata(self) -> dict[str, Any]:
        """Returns JSON-serializable terrain metadata."""

        return {
            "spec": self.spec.model_dump(mode="json"),
            "elevation_min_m": float(np.min(self.elevation_m)),
            "elevation_max_m": float(np.max(self.elevation_m)),
        }
