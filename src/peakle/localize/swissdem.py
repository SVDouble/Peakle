"""swissALTI3D 2 m patches: the fine map for near-field sharp terrain.

Copernicus GLO-30 cannot represent sub-30 m features — a jagged limestone tower renders as a
smooth hump, so the DEM skyline misses exactly the teeth the photo shows.  Where a sample sits in
Switzerland, a 2 m swissALTI3D patch around the camera fixes the near field; the coarse base map
still supplies the far horizon.  Strategy (matching the pose-search design): ROUGH SEARCH on the
coarse map first, the FINE patch only for final alignment, metrics and rendering.

Tiles come from the free swisstopo STAC API (no auth), cached under local/data/swissalti; the
patch is resampled into the camera's local-ENU frame (the ray-caster's frame).  Requires pyproj.
"""

from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

Image.MAX_IMAGE_PIXELS = None

STAC_ITEMS = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d/items"
# rough Switzerland bbox — gates whether a patch is even worth attempting
CH_BBOX = (45.80, 5.95, 47.85, 10.55)


class Patch:
    """High-res elevation patch in local ENU; NaN outside coverage."""

    def __init__(self, x_m: np.ndarray, y_m: np.ndarray, elevation_m: np.ndarray):
        self.x_m = x_m
        self.y_m = y_m
        self.elevation_m = elevation_m


def in_switzerland(lat: float, lon: float) -> bool:
    return CH_BBOX[0] <= lat <= CH_BBOX[2] and CH_BBOX[1] <= lon <= CH_BBOX[3]


def _lv95(lat: float, lon: float):
    from pyproj import Transformer

    return Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True).transform(lon, lat)


def ensure_swiss_tiles(tile_dir: str | Path, lat: float, lon: float, radius_m: float = 4000.0) -> int:
    """Downloads the 2 m tiles covering ``radius_m`` around a position; returns tiles present."""

    tile_dir = Path(tile_dir)
    tile_dir.mkdir(parents=True, exist_ok=True)
    dlat = radius_m / 111320.0
    dlon = radius_m / (111320.0 * math.cos(math.radians(lat)))
    url = f"{STAC_ITEMS}?bbox={lon-dlon},{lat-dlat},{lon+dlon},{lat+dlat}&limit=100"
    have = 0
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            items = json.loads(r.read()).get("features", [])
    except Exception:
        return len(list(tile_dir.glob("*.tif")))
    for item in items:
        for asset in item.get("assets", {}).values():
            href = asset.get("href", "")
            if href.endswith(".tif") and "_2_" in href.rsplit("/", 1)[-1]:  # the 2 m variant
                dest = tile_dir / href.rsplit("/", 1)[-1]
                if not dest.exists():
                    try:
                        urllib.request.urlretrieve(href, dest)
                    except Exception:
                        continue
                have += 1
    return have


def load_swiss_patch(
    tile_dir: str | Path, cam_lat: float, cam_lon: float, res: float = 4.0, radius_m: float = 4500.0
) -> Patch | None:
    """Mosaics cached tiles within ``radius_m`` into a local-ENU patch (None if no coverage)."""

    cE, cN = _lv95(cam_lat, cam_lon)
    tiles = []
    for f in Path(tile_dir).glob("*.tif"):
        try:
            ek, nk = f.name.split("_")[2].split("-")
            e0, n0 = int(ek) * 1000, int(nk) * 1000
        except Exception:
            continue
        if abs(e0 + 500 - cE) > radius_m + 1000 or abs(n0 + 500 - cN) > radius_m + 1000:
            continue
        a = np.array(Image.open(f), dtype=np.float32)
        a[a < -1000] = np.nan
        tiles.append((e0, n0, a))
    if not tiles:
        return None
    px = 2.0
    emin = min(t[0] for t in tiles)
    emax = max(t[0] for t in tiles) + 1000
    nmin = min(t[1] for t in tiles)
    nmax = max(t[1] for t in tiles) + 1000
    mos = np.full((int((nmax - nmin) / px), int((emax - emin) / px)), np.nan, np.float32)
    for e0, n0, a in tiles:
        r0 = int((nmax - (n0 + 1000)) / px)
        c0 = int((e0 - emin) / px)
        mos[r0 : r0 + a.shape[0], c0 : c0 + a.shape[1]] = a
    mlon = 111320.0 * math.cos(math.radians(cam_lat))
    xs = np.arange(math.floor(emin - cE), math.ceil(emax - cE), res)
    ys = np.arange(math.floor(nmin - cN), math.ceil(nmax - cN), res)
    xx, yy = np.meshgrid(xs, ys)
    lat = cam_lat + yy / 111320.0
    lon = cam_lon + xx / mlon
    from pyproj import Transformer

    ee, nn = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True).transform(lon.ravel(), lat.ravel())
    z = map_coordinates(
        np.where(np.isfinite(mos), mos, -9999.0),
        [(nmax - nn) / px, (ee - emin) / px],
        order=1,
        mode="constant",
        cval=-9999.0,
    ).reshape(xx.shape)
    z[z < -1000] = np.nan
    return Patch(xs.astype(float), ys.astype(float), z)
