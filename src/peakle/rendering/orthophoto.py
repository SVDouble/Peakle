"""Georeferenced raster appearance for terrain renders.

The pose solver must never confuse the browser's procedural relief colours with
real image evidence.  This module therefore keeps orthophoto acquisition,
sampling, and provenance explicit.  Network access is opt-in; benchmark runs
normally consume an already populated, content-addressed tile cache.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import map_coordinates

from peakle.domain.coordinates import EARTH_RADIUS_M, LocalFrame

DEFAULT_SWISSIMAGE_LAYER = "ch.swisstopo.swissimage"
DEFAULT_SWISSIMAGE_URL = "https://wmts.geo.admin.ch/1.0.0/{layer}/default/{time}/3857/{zoom}/{x}/{y}.jpeg"
WEB_MERCATOR_MAX_LAT_DEG = 85.05112878
XYZ_TILE_SIZE_PX = 256


class AppearanceRaster(Protocol):
    """A georeferenced RGB raster that can be sampled in Peakle local ENU."""

    def sample_local(
        self,
        east_m: NDArray[np.float64],
        north_m: NDArray[np.float64],
    ) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
        """Return RGB and a validity mask for local coordinates."""

    def provenance(self) -> dict[str, Any]:
        """Return JSON-serializable source and content identity."""


class MissingOrthophotoTiles(RuntimeError):
    """Raised when an offline mosaic request is not fully cached."""

    def __init__(self, missing: list[Path], *, requested: int) -> None:
        preview = ", ".join(str(path) for path in missing[:3])
        suffix = "" if len(missing) <= 3 else f" (+{len(missing) - 3} more)"
        super().__init__(
            f"{len(missing)}/{requested} orthophoto tiles are absent from the offline cache: {preview}{suffix}"
        )
        self.missing = tuple(missing)
        self.requested = requested


@dataclass(frozen=True)
class LocalRasterTexture:
    """A regular RGB texture already expressed in terrain-local coordinates."""

    x_m: NDArray[np.float64]
    y_m: NDArray[np.float64]
    rgb: NDArray[np.uint8]
    valid_mask: NDArray[np.bool_] | None = None
    source: str = "local_raster"
    content_sha256: str | None = None
    _raster_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        x = np.asarray(self.x_m, dtype=np.float64)
        y = np.asarray(self.y_m, dtype=np.float64)
        rgb = np.asarray(self.rgb, dtype=np.uint8)
        if x.ndim != 1 or y.ndim != 1 or x.size < 2 or y.size < 2:
            raise ValueError("local texture axes must be one-dimensional with at least two samples")
        if not np.all(np.diff(x) > 0.0) or not np.all(np.diff(y) > 0.0):
            raise ValueError("local texture axes must be strictly increasing")
        if rgb.shape != (y.size, x.size, 3):
            raise ValueError(f"texture RGB shape must be {(y.size, x.size, 3)}, got {rgb.shape}")
        if self.valid_mask is not None and np.asarray(self.valid_mask).shape != rgb.shape[:2]:
            raise ValueError("texture validity mask must match its RGB height and width")
        object.__setattr__(self, "x_m", x)
        object.__setattr__(self, "y_m", y)
        object.__setattr__(self, "rgb", rgb)
        if self.valid_mask is not None:
            object.__setattr__(self, "valid_mask", np.asarray(self.valid_mask, dtype=np.bool_))
        if self.content_sha256 is not None and re.fullmatch(r"[0-9a-fA-F]{64}", self.content_sha256) is None:
            raise ValueError("source content SHA-256 must contain exactly 64 hexadecimal characters")
        digest = hashlib.sha256()
        digest.update(b"peakle-local-raster-v1\0")
        _update_array_digest(digest, x)
        _update_array_digest(digest, y)
        _update_array_digest(digest, rgb)
        if self.valid_mask is None:
            digest.update(b"no-validity-mask\0")
        else:
            _update_array_digest(digest, self.valid_mask)
        object.__setattr__(self, "_raster_sha256", digest.hexdigest())

    def sample_local(
        self,
        east_m: NDArray[np.float64],
        north_m: NDArray[np.float64],
    ) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
        east, north = _matching_coordinates(east_m, north_m)
        finite = np.isfinite(east.ravel()) & np.isfinite(north.ravel())
        # scipy.ndimage has platform-dependent behaviour for NaN coordinates.
        # Substitute a harmless in-bounds coordinate before sampling, then mask
        # the result below.
        safe_east = np.where(finite, east.ravel(), self.x_m[0])
        safe_north = np.where(finite, north.ravel(), self.y_m[0])
        col = np.interp(safe_east, self.x_m, np.arange(self.x_m.size, dtype=float))
        row = np.interp(safe_north, self.y_m, np.arange(self.y_m.size, dtype=float))
        inside = (
            finite
            & (east.ravel() >= self.x_m[0])
            & (east.ravel() <= self.x_m[-1])
            & (north.ravel() >= self.y_m[0])
            & (north.ravel() <= self.y_m[-1])
        )
        sampled = np.stack(
            [map_coordinates(self.rgb[..., channel], [row, col], order=1, mode="nearest") for channel in range(3)],
            axis=-1,
        )
        if self.valid_mask is not None:
            # RGB is bilinearly interpolated, so a nearest-neighbour validity
            # lookup could accept a colour contaminated by a nodata neighbour.
            # Require the complete interpolation support to be valid instead.
            support = map_coordinates(
                self.valid_mask.astype(np.float64),
                [row, col],
                order=1,
                mode="constant",
                cval=0.0,
            )
            inside &= support >= 1.0 - 1e-9
        sampled = np.clip(np.rint(sampled), 0.0, 255.0).astype(np.uint8)
        sampled[~inside] = 0
        return sampled.reshape((*east.shape, 3)), inside.reshape(east.shape)

    def provenance(self) -> dict[str, Any]:
        result = {
            "kind": "local_raster",
            "source": self.source,
            # This identifies the exact sampled raster, not just its RGB bytes:
            # coordinate axes and nodata support change sampling semantics too.
            "content_sha256": self._raster_sha256,
            "shape": list(self.rgb.shape),
            "bounds_local_m": {
                "east": [float(self.x_m[0]), float(self.x_m[-1])],
                "north": [float(self.y_m[0]), float(self.y_m[-1])],
            },
        }
        if self.content_sha256 is not None:
            # Keep an upstream file/artifact digest separately; it must not
            # override the identity of the decoded, georeferenced raster.
            result["source_content_sha256"] = self.content_sha256.lower()
        return result


@dataclass(frozen=True)
class OrthophotoMosaic:
    """A cached XYZ/WMTS tile mosaic with local-coordinate sampling."""

    frame: LocalFrame
    rgb: NDArray[np.uint8]
    zoom: int
    origin_global_x_px: int
    origin_global_y_px: int
    layer: str
    time: str
    tile_records: tuple[dict[str, Any], ...]
    service_url_template: str
    network_used_during_build: bool = False
    _mosaic_rgb_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("orthophoto mosaic must be HxWx3 RGB")
        object.__setattr__(self, "rgb", rgb)
        digest = hashlib.sha256()
        digest.update(b"peakle-orthophoto-mosaic-v1\0")
        _update_array_digest(digest, rgb)
        object.__setattr__(self, "_mosaic_rgb_sha256", digest.hexdigest())

    def sample_local(
        self,
        east_m: NDArray[np.float64],
        north_m: NDArray[np.float64],
    ) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
        east, north = _matching_coordinates(east_m, north_m)
        latitude = self.frame.origin.latitude_deg + np.degrees(north / EARTH_RADIUS_M)
        longitude = self.frame.origin.longitude_deg + np.degrees(
            east / (EARTH_RADIUS_M * math.cos(math.radians(self.frame.origin.latitude_deg)))
        )
        global_x, global_y = _lon_lat_to_global_pixels(longitude, latitude, self.zoom)
        # XYZ global-pixel coordinates describe pixel *edges*: the centre of
        # tile pixel (0, 0) is at global (0.5, 0.5). ndarray indices describe
        # sample centres, hence the half-pixel conversion here.
        col = global_x - self.origin_global_x_px - 0.5
        row = global_y - self.origin_global_y_px - 0.5
        finite = np.isfinite(col) & np.isfinite(row)
        inside = finite & (col >= 0.0) & (col <= self.rgb.shape[1] - 1) & (row >= 0.0) & (row <= self.rgb.shape[0] - 1)
        flat_row = np.where(finite, row, 0.0).ravel()
        flat_col = np.where(finite, col, 0.0).ravel()
        sampled = np.stack(
            [
                map_coordinates(self.rgb[..., channel], [flat_row, flat_col], order=1, mode="nearest")
                for channel in range(3)
            ],
            axis=-1,
        )
        sampled = np.clip(np.rint(sampled), 0.0, 255.0).astype(np.uint8).reshape((*east.shape, 3))
        sampled[~inside] = 0
        return sampled, inside.astype(np.bool_)

    def provenance(self) -> dict[str, Any]:
        manifest = {
            "schema": "peakle-orthophoto-tile-manifest-v1",
            "layer": self.layer,
            "time": self.time,
            "zoom": self.zoom,
            "tiles": self.tile_records,
        }
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        return {
            "kind": "orthophoto_wmts",
            "provider": "Swiss Federal Office of Topography swisstopo / geo.admin.ch",
            "layer": self.layer,
            "time": self.time,
            "zoom": self.zoom,
            "tile_count": len(self.tile_records),
            "tile_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
            "mosaic_rgb_sha256": self._mosaic_rgb_sha256,
            "mosaic_shape": list(self.rgb.shape),
            "local_frame_origin": self.frame.origin.model_dump(mode="json"),
            "mosaic_origin_global_pixel_edge": [self.origin_global_x_px, self.origin_global_y_px],
            "pixel_coordinate_convention": "XYZ edges; array samples at edge + 0.5 pixels",
            "service_url_template": self.service_url_template,
            "network_used_during_mosaic_build": self.network_used_during_build,
        }


@dataclass(frozen=True)
class SwissImageProvider:
    """Opt-in SWISSIMAGE WMTS tile cache and mosaic builder.

    ``download_missing`` defaults to ``False`` so an experiment cannot silently
    change its inputs or depend on network availability.  A separate audited
    provisioning run may set it to true and then persist the generated manifest.
    """

    cache_dir: Path
    zoom: int = 15
    time: str = "current"
    layer: str = DEFAULT_SWISSIMAGE_LAYER
    url_template: str = DEFAULT_SWISSIMAGE_URL
    download_missing: bool = False
    max_tiles: int = 1600
    timeout_s: float = 30.0
    download_workers: int = 6

    def mosaic_for_local_bounds(
        self,
        frame: LocalFrame,
        *,
        east_min_m: float,
        east_max_m: float,
        north_min_m: float,
        north_max_m: float,
    ) -> OrthophotoMosaic:
        """Load or provision every tile intersecting a local ENU rectangle."""

        if not 0 <= self.zoom <= 22:
            raise ValueError("SWISSIMAGE zoom must be in [0, 22]")
        if east_min_m >= east_max_m or north_min_m >= north_max_m:
            raise ValueError("orthophoto bounds must have positive area")
        corners_e = np.asarray([east_min_m, east_max_m, east_min_m, east_max_m], dtype=float)
        corners_n = np.asarray([north_min_m, north_min_m, north_max_m, north_max_m], dtype=float)
        latitude = frame.origin.latitude_deg + np.degrees(corners_n / EARTH_RADIUS_M)
        longitude = frame.origin.longitude_deg + np.degrees(
            corners_e / (EARTH_RADIUS_M * math.cos(math.radians(frame.origin.latitude_deg)))
        )
        global_x, global_y = _lon_lat_to_global_pixels(longitude, latitude, self.zoom)
        # Include the half-pixel interpolation support at each bound.  This is
        # normally free, but correctly pulls in the adjacent tile when a local
        # bound lands exactly on an XYZ tile edge.
        tile_x_min = int(math.floor((float(np.min(global_x)) - 0.5) / XYZ_TILE_SIZE_PX))
        tile_x_max = int(math.floor((float(np.max(global_x)) + 0.5) / XYZ_TILE_SIZE_PX))
        tile_y_min = int(math.floor((float(np.min(global_y)) - 0.5) / XYZ_TILE_SIZE_PX))
        tile_y_max = int(math.floor((float(np.max(global_y)) + 0.5) / XYZ_TILE_SIZE_PX))
        tile_count = (tile_x_max - tile_x_min + 1) * (tile_y_max - tile_y_min + 1)
        if tile_count > self.max_tiles:
            raise ValueError(
                f"orthophoto request needs {tile_count} tiles at zoom {self.zoom}; "
                f"the configured safety limit is {self.max_tiles}"
            )

        requested = [
            (tile_x, tile_y, self._tile_path(tile_x, tile_y))
            for tile_y in range(tile_y_min, tile_y_max + 1)
            for tile_x in range(tile_x_min, tile_x_max + 1)
        ]
        missing = [path for _x, _y, path in requested if not path.is_file()]
        if missing and not self.download_missing:
            raise MissingOrthophotoTiles(missing, requested=tile_count)
        if missing:
            downloads = [(tile_x, tile_y, path) for tile_x, tile_y, path in requested if not path.is_file()]
            if self.download_workers < 1:
                raise ValueError("orthophoto download_workers must be positive")
            with ThreadPoolExecutor(max_workers=self.download_workers) as executor:
                list(executor.map(lambda item: self._download_tile(*item), downloads))

        rows: list[NDArray[np.uint8]] = []
        records: list[dict[str, Any]] = []
        for tile_y in range(tile_y_min, tile_y_max + 1):
            images: list[NDArray[np.uint8]] = []
            for tile_x in range(tile_x_min, tile_x_max + 1):
                path = self._tile_path(tile_x, tile_y)
                content = path.read_bytes()
                with Image.open(io.BytesIO(content)) as image:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                if rgb.shape != (XYZ_TILE_SIZE_PX, XYZ_TILE_SIZE_PX, 3):
                    raise ValueError(f"unexpected SWISSIMAGE tile shape {rgb.shape} in {path}")
                images.append(rgb)
                records.append(
                    {
                        "z": self.zoom,
                        "x": tile_x,
                        "y": tile_y,
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "size_bytes": len(content),
                    }
                )
            rows.append(np.concatenate(images, axis=1))
        mosaic = np.concatenate(rows, axis=0)
        return OrthophotoMosaic(
            frame=frame,
            rgb=mosaic,
            zoom=self.zoom,
            origin_global_x_px=tile_x_min * XYZ_TILE_SIZE_PX,
            origin_global_y_px=tile_y_min * XYZ_TILE_SIZE_PX,
            layer=self.layer,
            time=self.time,
            tile_records=tuple(records),
            service_url_template=self.url_template,
            network_used_during_build=bool(missing),
        )

    def _tile_path(self, tile_x: int, tile_y: int) -> Path:
        safe_layer = _safe_component(self.layer)
        safe_time = _safe_component(self.time)
        return self.cache_dir / safe_layer / safe_time / "3857" / str(self.zoom) / str(tile_x) / f"{tile_y}.jpeg"

    def _tile_url(self, tile_x: int, tile_y: int) -> str:
        return self.url_template.format(
            layer=self.layer,
            time=self.time,
            zoom=self.zoom,
            x=tile_x,
            y=tile_y,
        )

    def _download_tile(self, tile_x: int, tile_y: int, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            self._tile_url(tile_x, tile_y),
            headers={"User-Agent": "Peakle pose-research/0.1 (orthophoto cache provisioning)"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310 - fixed HTTPS service
            content = response.read()
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(content)
        temporary.replace(path)


def _matching_coordinates(
    east_m: NDArray[np.float64],
    north_m: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    east = np.asarray(east_m, dtype=np.float64)
    north = np.asarray(north_m, dtype=np.float64)
    if east.shape != north.shape:
        raise ValueError(f"east/north coordinates must share a shape, got {east.shape} and {north.shape}")
    return east, north


def _lon_lat_to_global_pixels(
    longitude_deg: NDArray[np.float64],
    latitude_deg: NDArray[np.float64],
    zoom: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    longitude = np.asarray(longitude_deg, dtype=np.float64)
    latitude = np.clip(np.asarray(latitude_deg, dtype=np.float64), -WEB_MERCATOR_MAX_LAT_DEG, WEB_MERCATOR_MAX_LAT_DEG)
    scale = float(XYZ_TILE_SIZE_PX * (2**zoom))
    x_px = (longitude + 180.0) / 360.0 * scale
    latitude_rad = np.radians(latitude)
    y_px = (1.0 - np.arcsinh(np.tan(latitude_rad)) / math.pi) * 0.5 * scale
    return x_px, y_px


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not safe:
        raise ValueError("cache path component cannot be empty")
    return safe


def _update_array_digest(digest: Any, array: NDArray[Any]) -> None:
    """Hash an ndarray without losing dtype or shape boundaries."""

    contiguous = np.ascontiguousarray(array)
    descriptor = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest.update(len(descriptor).to_bytes(8, "big"))
    digest.update(descriptor)
    digest.update(memoryview(contiguous).cast("B"))
