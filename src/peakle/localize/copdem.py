"""Copernicus GLO-30 DEM: tile download + local-ENU terrain around a viewpoint.

Copernicus is the default DEM for pose work — SRTM voids steep Alpine summits (the Matterhorn
cell reads ~900 m low), Copernicus does not.  Tiles are free, no-auth, 1°x1° GeoTIFF (EPSG:4326,
3600x3600) on S3.  Only the tiles intersecting the requested extent are opened, and open tiles
are cached per-process (a naive glob-everything loader costs >100 MB per tile per call).

North/East hemisphere only (the Alps); extend the tile-name scheme before using elsewhere.
"""

from __future__ import annotations

import math
import re
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

Image.MAX_IMAGE_PIXELS = None

_S3 = "https://copernicus-dem-30m.s3.amazonaws.com"
_TILE_CACHE: dict[Path, np.ndarray] = {}
_TILE_CACHE_MAX = 12  # ~50 MB per open tile; benchmark runs sweep many positions


class CopTerrain:
    """Local-ENU elevation grid: ``x_m`` east, ``y_m`` north, ``elevation_m[north, east]``."""

    def __init__(self, x_m: np.ndarray, y_m: np.ndarray, elevation_m: np.ndarray):
        self.x_m = x_m
        self.y_m = y_m
        self.elevation_m = elevation_m

    def elevation_at(self, east_m: float, north_m: float) -> float:
        j = (east_m - self.x_m[0]) / (self.x_m[1] - self.x_m[0])
        i = (north_m - self.y_m[0]) / (self.y_m[1] - self.y_m[0])
        return float(map_coordinates(self.elevation_m, [[i], [j]], order=1, mode="nearest")[0])


def tile_path(tile_dir: str | Path, lat0: int, lon0: int) -> Path:
    return Path(tile_dir) / f"cop_N{lat0:02d}E{lon0:03d}.tif"


def _existing_tile(tile_dir: Path, lat0: int, lon0: int) -> Path | None:
    for f in tile_dir.glob("*.tif"):
        m = re.search(r"N(\d{2}).*?E(\d{3})", f.name)
        if m and int(m.group(1)) == lat0 and int(m.group(2)) == lon0:
            return f
    return None


def ensure_tile(tile_dir: str | Path, lat0: int, lon0: int) -> Path:
    """Returns the local path of a 1° tile, downloading it from S3 when missing."""

    tile_dir = Path(tile_dir)
    tile_dir.mkdir(parents=True, exist_ok=True)
    found = _existing_tile(tile_dir, lat0, lon0)
    if found is not None:
        return found
    stem = f"Copernicus_DSM_COG_10_N{lat0:02d}_00_E{lon0:03d}_00_DEM"
    url = f"{_S3}/{stem}/{stem}.tif"
    dest = tile_path(tile_dir, lat0, lon0)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def _load_tile(path: Path) -> np.ndarray:
    if path not in _TILE_CACHE:
        while len(_TILE_CACHE) >= _TILE_CACHE_MAX:
            _TILE_CACHE.pop(next(iter(_TILE_CACHE)))
        _TILE_CACHE[path] = np.asarray(Image.open(path), dtype=np.float32)
    else:
        _TILE_CACHE[path] = _TILE_CACHE.pop(path)  # move to the back = most recently used
    return _TILE_CACHE[path]


def load_cop_around(
    tile_dir: str | Path,
    cam_lat: float,
    cam_lon: float,
    extent_m: float = 40000.0,
    grid: int = 1400,
    download: bool = True,
) -> CopTerrain:
    """Local-ENU terrain of ``extent_m`` centred on the viewpoint (camera at x=y=0)."""

    half_lat = (extent_m / 2) / 111320.0
    half_lon = (extent_m / 2) / (111320.0 * math.cos(math.radians(cam_lat)))
    lats = np.linspace(cam_lat - half_lat, cam_lat + half_lat, grid)
    lons = np.linspace(cam_lon - half_lon, cam_lon + half_lon, grid)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    elev = np.full((grid, grid), np.nan)
    for lat0 in range(int(math.floor(lats[0])), int(math.floor(lats[-1])) + 1):
        for lon0 in range(int(math.floor(lons[0])), int(math.floor(lons[-1])) + 1):
            path = ensure_tile(tile_dir, lat0, lon0) if download else _existing_tile(Path(tile_dir), lat0, lon0)
            if path is None:
                continue
            arr = _load_tile(path)
            hh, ww = arr.shape
            inside = (lat_grid >= lat0) & (lat_grid < lat0 + 1) & (lon_grid >= lon0) & (lon_grid < lon0 + 1)
            if not inside.any():
                continue
            rr = (lat0 + 1 - lat_grid[inside]) * (hh - 1)
            cc = (lon_grid[inside] - lon0) * (ww - 1)
            elev[inside] = map_coordinates(arr, [rr, cc], order=1, mode="nearest")
    if not np.all(np.isfinite(elev)):
        elev = np.nan_to_num(elev, nan=float(np.nanmin(elev)))
    x_m = (lons - cam_lon) * 111320.0 * math.cos(math.radians(cam_lat))
    y_m = (lats - cam_lat) * 111320.0
    return CopTerrain(x_m, y_m, elev)
