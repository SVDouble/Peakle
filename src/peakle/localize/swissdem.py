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
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlencode, urljoin, urlsplit

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

Image.MAX_IMAGE_PIXELS = None

STAC_ITEMS = "https://data.geo.admin.ch/api/stac/v1/collections/ch.swisstopo.swissalti3d/items"
# rough Switzerland bbox — gates whether a patch is even worth attempting
CH_BBOX = (45.80, 5.95, 47.85, 10.55)
STAC_PAGE_SIZE = 100

type _TileKey = tuple[int, int]


@dataclass(frozen=True)
class _TileAsset:
    key: _TileKey
    edition: int
    name: str
    href: str
    item_timestamp: str

    @property
    def preference(self) -> tuple[int, str, str]:
        return self.edition, self.item_timestamp, self.href


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


def _tile_identity(name: str) -> tuple[_TileKey, int] | None:
    """Parse a cached swissALTI3D 2 m LV95 tile name."""

    if Path(name).suffix.lower() != ".tif":
        return None
    parts = Path(name).stem.split("_")
    if len(parts) < 5 or parts[0].lower() != "swissalti3d" or parts[3] != "2" or parts[4] != "2056":
        return None
    try:
        edition = int(parts[1])
        east_km, north_km = (int(value) for value in parts[2].split("-", maxsplit=1))
    except TypeError, ValueError:
        return None
    return (east_km, north_km), edition


def _preferred_cached_tiles(tile_dir: Path) -> dict[_TileKey, Path]:
    """Return one deterministic, newest cached edition for every LV95 tile."""

    preferred: dict[_TileKey, tuple[int, Path]] = {}
    for path in sorted(tile_dir.glob("*.tif"), key=lambda candidate: candidate.name):
        identity = _tile_identity(path.name)
        if identity is None:
            continue
        key, edition = identity
        candidate = (edition, path)
        current = preferred.get(key)
        if current is None or (candidate[0], candidate[1].name) > (current[0], current[1].name):
            preferred[key] = candidate
    return {key: edition_and_path[1] for key, edition_and_path in preferred.items()}


def _item_timestamp(item: dict[str, Any]) -> str:
    properties = item.get("properties")
    if not isinstance(properties, dict):
        return ""
    # ``updated`` distinguishes republished assets within the same named edition.
    for field in ("updated", "datetime", "created"):
        value = properties.get(field)
        if isinstance(value, str):
            return value
    return ""


def _preferred_assets(items: list[dict[str, Any]]) -> dict[_TileKey, _TileAsset]:
    preferred: dict[_TileKey, _TileAsset] = {}
    for item in items:
        assets = item.get("assets")
        if not isinstance(assets, dict):
            continue
        timestamp = _item_timestamp(item)
        for raw_asset in assets.values():
            if not isinstance(raw_asset, dict):
                continue
            href = raw_asset.get("href")
            if not isinstance(href, str):
                continue
            name = Path(unquote(urlsplit(href).path)).name
            identity = _tile_identity(name)
            if identity is None:
                continue
            key, edition = identity
            candidate = _TileAsset(key, edition, name, href, timestamp)
            current = preferred.get(key)
            if current is None or candidate.preference > current.preference:
                preferred[key] = candidate
    return preferred


def _next_request(page: dict[str, Any], current_url: str) -> str | urllib.request.Request | None:
    links = page.get("links")
    if not isinstance(links, list):
        return None
    link = next(
        (
            candidate
            for candidate in links
            if isinstance(candidate, dict) and str(candidate.get("rel", "")).lower() == "next"
        ),
        None,
    )
    if link is None or not isinstance(link.get("href"), str):
        return None
    href = urljoin(current_url, link["href"])
    method = str(link.get("method", "GET")).upper()
    raw_headers = link.get("headers", {})
    headers = {str(key): str(value) for key, value in raw_headers.items()} if isinstance(raw_headers, dict) else {}
    if method == "GET":
        return urllib.request.Request(href, headers=headers, method=method) if headers else href
    if method == "POST":
        body = link.get("body", {})
        if not isinstance(body, dict):
            raise ValueError("STAC next-link POST body must be an object")
        headers.setdefault("Content-Type", "application/json")
        return urllib.request.Request(
            href,
            data=json.dumps(body, sort_keys=True, separators=(",", ":")).encode(),
            headers=headers,
            method=method,
        )
    raise ValueError(f"unsupported STAC next-link method: {method}")


def _stac_items(initial_url: str) -> list[dict[str, Any]]:
    """Read every STAC page, following the server-provided next link."""

    request: str | urllib.request.Request | None = initial_url
    seen: set[tuple[str, str, str]] = set()
    items: list[dict[str, Any]] = []
    while request is not None:
        if isinstance(request, str):
            current_url = request
            identity = ("GET", request, "")
        else:
            current_url = request.full_url
            identity = (request.get_method(), current_url, repr(request.data))
        if identity in seen:
            raise ValueError(f"cyclic STAC pagination link: {current_url}")
        seen.add(identity)
        with urllib.request.urlopen(request, timeout=30) as response:
            page = json.loads(response.read())
        if not isinstance(page, dict):
            raise ValueError("STAC response must be a JSON object")
        features = page.get("features", [])
        if not isinstance(features, list):
            raise ValueError("STAC response features must be an array")
        items.extend(feature for feature in features if isinstance(feature, dict))
        request = _next_request(page, current_url)
    return items


def ensure_swiss_tiles(tile_dir: str | Path, lat: float, lon: float, radius_m: float = 4000.0) -> int:
    """Download missing 2 m tiles around a position; return unique LV95 tiles present.

    The STAC collection exposes multiple editions for many coordinates.  An existing cached
    edition remains valid; otherwise only the newest asset returned across *all* pages is fetched.
    """

    tile_dir = Path(tile_dir)
    tile_dir.mkdir(parents=True, exist_ok=True)
    cached = _preferred_cached_tiles(tile_dir)
    dlat = radius_m / 111320.0
    dlon = radius_m / (111320.0 * math.cos(math.radians(lat)))
    bbox = f"{lon - dlon},{lat - dlat},{lon + dlon},{lat + dlat}"
    url = f"{STAC_ITEMS}?{urlencode({'bbox': bbox, 'limit': STAC_PAGE_SIZE}, safe=',')}"
    try:
        assets = _preferred_assets(_stac_items(url))
    except Exception:
        return len(cached)
    for key in sorted(assets):
        if key in cached:
            continue
        asset = assets[key]
        dest = tile_dir / asset.name
        try:
            urllib.request.urlretrieve(asset.href, dest)
        except Exception:
            continue
        if dest.exists():
            cached[key] = dest
    return len(cached)


def load_swiss_patch(
    tile_dir: str | Path, cam_lat: float, cam_lon: float, res: float = 4.0, radius_m: float = 4500.0
) -> Patch | None:
    """Mosaics cached tiles within ``radius_m`` into a local-ENU patch (None if no coverage)."""

    cE, cN = _lv95(cam_lat, cam_lon)
    tiles = []
    preferred = _preferred_cached_tiles(Path(tile_dir))
    for (east_km, north_km), f in sorted(preferred.items()):
        e0, n0 = east_km * 1000, north_km * 1000
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
    z = _resample_with_full_finite_support(
        mos,
        [(nmax - nn) / px, (ee - emin) / px],
        xx.shape,
    )
    return Patch(xs.astype(float), ys.astype(float), z)


def _resample_with_full_finite_support(
    mosaic: np.ndarray,
    coordinates: list[np.ndarray],
    output_shape: tuple[int, ...],
) -> np.ndarray:
    """Bilinearly resample only where every contributing source cell is finite.

    Interpolating a numeric nodata sentinel directly can turn a mostly-valid footprint into a
    plausible-looking but impossible negative elevation.  Interpolate a validity plane alongside
    zero-filled elevations and retain only samples with complete bilinear support instead.
    """

    finite = np.isfinite(mosaic)
    values = map_coordinates(
        np.where(finite, mosaic, 0.0),
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
    ).reshape(output_shape)
    support = map_coordinates(
        finite.astype(np.float32),
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
    ).reshape(output_shape)
    values[support < 1.0 - 1e-6] = np.nan
    return values
