"""Copernicus GLO-30 window as a workbench `TerrainMap`.

The `.hgt` provider covers only the tiles someone downloaded by hand; the Copernicus mosaic in
``local/data/copernicus`` covers the whole GT corpus (and auto-downloads missing 1° tiles from
the public S3 bucket). This adapter lets the web app recenter its 3D map on any GT sample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from peakle.domain.coordinates import EARTH_RADIUS_M, GeoPoint
from peakle.domain.terrain import TerrainMap, TerrainSpec
from peakle.localize.copdem import load_cop_around

DEFAULT_COP_DIR = Path(__file__).resolve().parents[3] / "local/data/copernicus"


def load_copernicus_terrain(
    center_lat_deg: float,
    center_lon_deg: float,
    extent_m: float = 40000.0,
    grid: int = 720,  # ~56 m/cell over a 40 km window (sampled at ~28 m ≈ native 30 m, then pooled)
    tile_dir: Path = DEFAULT_COP_DIR,
) -> TerrainMap:
    """Square `extent_m` window centred on a lat/lon, as a `TerrainMap`.

    The terrain origin is the requested coordinate (a camera there sits at local 0,0).
    """

    # Sample at 2x the target grid and mean-pool: point-sampling a 30 m surface model at
    # ~75 m spacing aliases badly — the mesh reads as spiky/rugged (user-reported).
    cop = load_cop_around(tile_dir, center_lat_deg, center_lon_deg, extent_m=extent_m, grid=grid * 2)
    elevation = cop.elevation_m.astype(np.float64).reshape(grid, 2, grid, 2).mean(axis=(1, 3))
    x_m = cop.x_m.reshape(grid, 2).mean(axis=1)
    y_m = cop.y_m.reshape(grid, 2).mean(axis=1)
    elev_min = float(elevation.min())
    elev_max = max(float(elevation.max()), elev_min + 1.0)
    spec = TerrainSpec(
        origin=GeoPoint(latitude_deg=center_lat_deg, longitude_deg=center_lon_deg, elevation_m=elev_min),
        width_m=extent_m,
        height_m=extent_m,
        grid_width=grid,
        grid_height=grid,
        min_elevation_m=elev_min,
        max_elevation_m=elev_max,
        seed=0,
    )
    x_grid, y_grid = np.meshgrid(x_m, y_m)
    lat_grid = center_lat_deg + np.degrees(y_grid / EARTH_RADIUS_M)
    lon_grid = center_lon_deg + np.degrees(x_grid / (EARTH_RADIUS_M * np.cos(np.radians(center_lat_deg))))
    return TerrainMap(
        spec=spec,
        x_m=x_m.astype(np.float64),
        y_m=y_m.astype(np.float64),
        elevation_m=elevation,
        latitude_deg=lat_grid.astype(np.float64),
        longitude_deg=lon_grid.astype(np.float64),
    )
