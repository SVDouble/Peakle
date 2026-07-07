"""Real-world elevation maps from `.hgt` DEM tiles.

Loads SRTM / Viewfinder-Panoramas `.hgt` tiles (a raw big-endian int16 grid over
a 1°x1° cell) and crops a `TerrainSpec`-sized window — centred on the tile's
highest point so the scene frames real summits — into the same `TerrainMap`
shape the synthetic generator produces. This is the data family PeakFinder uses
(SRTM / NASADEM / Viewfinder).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from peakle.domain.coordinates import EARTH_RADIUS_M, GeoPoint
from peakle.domain.terrain import TerrainMap, TerrainSpec

VOID_VALUE = -32768
METERS_PER_DEGREE_LAT = 111_320.0
DEFAULT_DEM_DIR = Path(os.environ.get("PEAKLE_DEM_DIR", "data/dem_samples"))

_TILE_RE = re.compile(r"([NS])(\d{2})([EW])(\d{3})", re.IGNORECASE)


def find_hgt_tile(directory: Path) -> Path:
    """Returns the first `.hgt` tile in `directory`."""

    tiles = sorted(directory.glob("*.hgt"))
    if not tiles:
        msg = f"no .hgt DEM tiles found in {directory!r} (set PEAKLE_DEM_DIR or download a tile)"
        raise FileNotFoundError(msg)
    return tiles[0]


def tile_name_for(lat_deg: float, lon_deg: float) -> str:
    """Returns the `.hgt` tile filename whose 1deg cell contains the coordinate."""

    ns = "N" if lat_deg >= 0 else "S"
    ew = "E" if lon_deg >= 0 else "W"
    return f"{ns}{abs(int(np.floor(lat_deg))):02d}{ew}{abs(int(np.floor(lon_deg))):03d}.hgt"


def load_dem_around(
    center_lat_deg: float,
    center_lon_deg: float,
    extent_m: float,
    grid: int = 384,
    dem_dir: Path = DEFAULT_DEM_DIR,
) -> TerrainMap:
    """Crops a square `extent_m` window centred on a given lat/lon into a `TerrainMap`.

    Mosaics across `.hgt` tiles so a window straddling a 1deg boundary works (common
    for a viewpoint near a tile edge). The terrain origin is the requested
    coordinate, so a camera there sits at local (0, 0).
    """

    half_lat = (extent_m / 2.0) / METERS_PER_DEGREE_LAT
    half_lon = (extent_m / 2.0) / (METERS_PER_DEGREE_LAT * np.cos(np.radians(center_lat_deg)))
    lats = np.linspace(center_lat_deg - half_lat, center_lat_deg + half_lat, grid)
    lons = np.linspace(center_lon_deg - half_lon, center_lon_deg + half_lon, grid)
    lon_grid, lat_grid = np.meshgrid(lons, lats)  # lats increase north == +y
    elevation = np.full((grid, grid), np.nan, dtype=np.float64)

    for tile_lat in range(int(np.floor(lats[0])), int(np.floor(lats[-1])) + 1):
        for tile_lon in range(int(np.floor(lons[0])), int(np.floor(lons[-1])) + 1):
            tile = Path(dem_dir) / tile_name_for(tile_lat + 0.5, tile_lon + 0.5)
            if not tile.exists():
                msg = f"missing DEM tile {tile.name} for window around ({center_lat_deg:.4f}, {center_lon_deg:.4f})"
                raise FileNotFoundError(msg)
            dem, south_lat_deg, west_lon_deg = _load_hgt(tile)
            side = dem.shape[0]
            inside = (
                (lat_grid >= south_lat_deg)
                & (lat_grid <= south_lat_deg + 1.0)
                & (lon_grid >= west_lon_deg)
                & (lon_grid <= west_lon_deg + 1.0)
            )
            if not inside.any():
                continue
            row_px = (lat_grid[inside] - south_lat_deg) * (side - 1)
            col_px = (lon_grid[inside] - west_lon_deg) * (side - 1)
            elevation[inside] = map_coordinates(dem, [row_px, col_px], order=1, mode="nearest")

    if not np.all(np.isfinite(elevation)):
        elevation = np.nan_to_num(elevation, nan=float(np.nanmin(elevation)))
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
    x_m = np.linspace(-extent_m / 2.0, extent_m / 2.0, grid)
    y_m = np.linspace(-extent_m / 2.0, extent_m / 2.0, grid)
    x_grid, y_grid = np.meshgrid(x_m, y_m)
    lat_grid = center_lat_deg + np.degrees(y_grid / EARTH_RADIUS_M)
    lon_grid = center_lon_deg + np.degrees(x_grid / (EARTH_RADIUS_M * np.cos(np.radians(center_lat_deg))))
    return TerrainMap(
        spec=spec,
        x_m=x_m.astype(np.float64),
        y_m=y_m.astype(np.float64),
        elevation_m=elevation.astype(np.float64),
        latitude_deg=lat_grid.astype(np.float64),
        longitude_deg=lon_grid.astype(np.float64),
    )


def load_dem_terrain(spec: TerrainSpec, hgt_path: Path) -> TerrainMap:
    """Crops a spec-sized window from a `.hgt` tile into a `TerrainMap`.

    Args:
        spec: Terrain spec providing the crop extent and grid resolution.
        hgt_path: Path to a square `.hgt` tile (SRTM/Viewfinder).

    Returns:
        A terrain map of real elevations centred on the tile's highest point.
    """

    dem, south_lat_deg, west_lon_deg = _load_hgt(hgt_path)
    side = dem.shape[0]
    deg_per_px = 1.0 / (side - 1)

    # Centre on the tile's tallest point so the scene frames real summits.
    peak_row, peak_col = np.unravel_index(int(np.argmax(dem)), dem.shape)
    center_lat = south_lat_deg + peak_row * deg_per_px
    center_lon = west_lon_deg + peak_col * deg_per_px

    meters_per_deg_lon = METERS_PER_DEGREE_LAT * np.cos(np.radians(center_lat))
    half_rows = (spec.height_m / 2.0) / (METERS_PER_DEGREE_LAT * deg_per_px)
    half_cols = (spec.width_m / 2.0) / (meters_per_deg_lon * deg_per_px)
    # Keep the crop window inside the tile.
    row_c = float(np.clip(peak_row, half_rows, side - 1 - half_rows))
    col_c = float(np.clip(peak_col, half_cols, side - 1 - half_cols))

    rows = np.linspace(row_c - half_rows, row_c + half_rows, spec.grid_height)
    cols = np.linspace(col_c - half_cols, col_c + half_cols, spec.grid_width)
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
    elevation = map_coordinates(dem, [row_grid.ravel(), col_grid.ravel()], order=1, mode="nearest").reshape(
        spec.grid_height, spec.grid_width
    )

    elev_min = float(elevation.min())
    elev_max = float(elevation.max())
    if elev_max <= elev_min:
        elev_max = elev_min + 1.0

    real_spec = spec.model_copy(
        update={
            "origin": GeoPoint(latitude_deg=center_lat, longitude_deg=center_lon, elevation_m=elev_min),
            "min_elevation_m": elev_min,
            "max_elevation_m": elev_max,
        }
    )

    x_m = np.linspace(-spec.width_m / 2.0, spec.width_m / 2.0, spec.grid_width)
    y_m = np.linspace(-spec.height_m / 2.0, spec.height_m / 2.0, spec.grid_height)
    x_grid, y_grid = np.meshgrid(x_m, y_m)
    lat_grid = center_lat + np.degrees(y_grid / EARTH_RADIUS_M)
    lon_grid = center_lon + np.degrees(x_grid / (EARTH_RADIUS_M * np.cos(np.radians(center_lat))))

    return TerrainMap(
        spec=real_spec,
        x_m=x_m.astype(np.float64),
        y_m=y_m.astype(np.float64),
        elevation_m=elevation.astype(np.float64),
        latitude_deg=lat_grid.astype(np.float64),
        longitude_deg=lon_grid.astype(np.float64),
    )


def _load_hgt(path: Path) -> tuple[np.ndarray, float, float]:
    """Reads a `.hgt` tile, flips it south-up, fills voids, and returns its SW corner."""

    raw = np.fromfile(path, dtype=">i2")
    side = int(round(raw.size**0.5))
    if side * side != raw.size:
        msg = f"{path.name}: not a square .hgt tile ({raw.size} samples)"
        raise ValueError(msg)
    # `.hgt` rows run north -> south; flip so row 0 is the southern edge to match
    # the terrain frame's north-positive y axis.
    dem = raw.reshape(side, side)[::-1, :].astype(np.float64)
    voids = dem == VOID_VALUE
    if voids.any():
        valid = dem[~voids]
        dem[voids] = float(np.median(valid)) if valid.size else 0.0
    south_lat_deg, west_lon_deg = _parse_tile_origin(path.name)
    return dem, south_lat_deg, west_lon_deg


def _parse_tile_origin(name: str) -> tuple[float, float]:
    """Parses the SW-corner lat/lon from a tile name like `N45E007.hgt`."""

    match = _TILE_RE.search(name)
    if not match:
        msg = f"cannot parse tile coordinates from {name!r}"
        raise ValueError(msg)
    lat = int(match.group(2)) * (1 if match.group(1).upper() == "N" else -1)
    lon = int(match.group(4)) * (1 if match.group(3).upper() == "E" else -1)
    return float(lat), float(lon)
