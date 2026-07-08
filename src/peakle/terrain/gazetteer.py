"""Real summit names from OpenStreetMap.

For real-DEM scenes the detector finds prominent local maxima but cannot name
them. This module looks up named peaks (``natural=peak`` / ``volcano``) from the
OpenStreetMap Overpass API within the terrain's bounding box and matches them to
detected peaks by proximity. The Overpass result is cached to disk per bounding
box, so it is a one-time network fetch and works offline afterwards. Detected
peaks with no nearby named summit fall back to a cartographic spot height
("Pt 4153 m").
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

from peakle.domain.coordinates import GeoPoint, LocalPoint
from peakle.domain.peaks import Peak
from peakle.domain.terrain import TerrainMap

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
MATCH_RADIUS_M = 1200.0
EXTENDED_MATCH_RADIUS_M = 2200.0
_REQUEST_TIMEOUT_S = 30


def name_peaks_from_osm(
    peaks: list[Peak],
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
    terrain: TerrainMap | None = None,
) -> list[Peak]:
    """Renames detected peaks and optionally adds unmatched OSM summit labels.

    Args:
        peaks: Detected peaks (already ordered by prominence).
        bbox: ``(south, west, north, east)`` in degrees.
        cache_dir: Directory for the per-bbox Overpass cache file.
        terrain: Terrain grid used to project OSM-only summit labels. When
            omitted, only detected peaks are returned for backward compatibility.

    Returns:
        Detected peaks with real names where a nearby OSM summit exists, otherwise
        a spot height. When ``terrain`` is given, unmatched OSM summits inside the
        terrain are appended as zero-prominence label points.
    """

    named = _load_named_peaks(bbox, cache_dir)
    assignments = _match_named_peaks(peaks, named)
    renamed: list[Peak] = []
    for peak_index, peak in enumerate(peaks):
        summit = assignments.get(peak_index)
        if summit is None:
            renamed.append(peak.model_copy(update={"name": f"Pt {round(peak.elevation_m)} m"}))
        else:
            renamed.append(peak.model_copy(update={"name": summit["name"]}))
    if terrain is None:
        return renamed

    matched = {_summit_key(summit) for summit in assignments.values()}
    osm_labels: list[Peak] = []
    for summit_index, summit in enumerate(named, start=1):
        if _summit_key(summit) in matched:
            continue
        label = _summit_label_peak(summit, terrain, summit_index)
        if label is not None:
            osm_labels.append(label)
    osm_labels.sort(key=lambda peak: (-peak.elevation_m, peak.name, peak.id))
    renamed.extend(osm_labels)
    return renamed


def _match_named_peaks(peaks: list[Peak], named: list[dict]) -> dict[int, dict]:
    assignments: dict[int, dict] = {}
    used_summits: set[int] = set()
    _assign_nearest_pairs(peaks, named, assignments, used_summits, MATCH_RADIUS_M)
    _assign_nearest_pairs(peaks, named, assignments, used_summits, EXTENDED_MATCH_RADIUS_M)
    return assignments


def _assign_nearest_pairs(
    peaks: list[Peak],
    named: list[dict],
    assignments: dict[int, dict],
    used_summits: set[int],
    radius_m: float,
) -> None:
    candidates: list[tuple[float, int, int]] = []
    for peak_index, peak in enumerate(peaks):
        if peak_index in assignments:
            continue
        for summit_index, summit in enumerate(named):
            if summit_index in used_summits:
                continue
            distance = _distance_m(peak, summit)
            if distance <= radius_m:
                candidates.append((distance, peak_index, summit_index))
    for _distance, peak_index, summit_index in sorted(candidates):
        if peak_index in assignments or summit_index in used_summits:
            continue
        assignments[peak_index] = named[summit_index]
        used_summits.add(summit_index)


def _distance_m(peak: Peak, summit: dict) -> float:
    lat = peak.geo_position.latitude_deg
    lon = peak.geo_position.longitude_deg
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    north_m = (summit["lat"] - lat) * 111_320.0
    east_m = (summit["lon"] - lon) * meters_per_deg_lon
    return math.hypot(north_m, east_m)


def _summit_key(summit: dict) -> tuple[str, float, float]:
    return (str(summit["name"]), round(float(summit["lat"]), 7), round(float(summit["lon"]), 7))


def _summit_label_peak(summit: dict, terrain: TerrainMap, index: int) -> Peak | None:
    geo = GeoPoint(
        latitude_deg=float(summit["lat"]),
        longitude_deg=float(summit["lon"]),
        elevation_m=0.0,
    )
    local = terrain.frame.geo_to_local(geo)
    if not _inside_terrain(local, terrain):
        return None
    elevation = terrain.elevation_at(local.east_m, local.north_m)
    local_position = LocalPoint(east_m=local.east_m, north_m=local.north_m, up_m=elevation)
    geo_position = geo.model_copy(update={"elevation_m": elevation})
    return Peak(
        id=f"osm-{index:03d}",
        name=str(summit["name"]),
        local_position=local_position,
        geo_position=geo_position,
        elevation_m=elevation,
        prominence_m=0.0,
    )


def _inside_terrain(point: LocalPoint, terrain: TerrainMap) -> bool:
    x_min = min(float(terrain.x_m[0]), float(terrain.x_m[-1]))
    x_max = max(float(terrain.x_m[0]), float(terrain.x_m[-1]))
    y_min = min(float(terrain.y_m[0]), float(terrain.y_m[-1]))
    y_max = max(float(terrain.y_m[0]), float(terrain.y_m[-1]))
    return x_min <= point.east_m <= x_max and y_min <= point.north_m <= y_max


def _load_named_peaks(bbox: tuple[float, float, float, float], cache_dir: Path) -> list[dict]:
    south, west, north, east = bbox
    cache = Path(cache_dir) / f"osm_peaks_{south:.3f}_{west:.3f}_{north:.3f}_{east:.3f}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except OSError, ValueError:
            pass
    try:
        named = _query_overpass(bbox)
    except OSError, ValueError, TimeoutError:
        return []
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(named))
    except OSError:
        pass
    return named


def _query_overpass(bbox: tuple[float, float, float, float]) -> list[dict]:
    south, west, north, east = bbox
    area = f"({south},{west},{north},{east})"
    query = (
        f'[out:json][timeout:{_REQUEST_TIMEOUT_S}];(node["natural"="peak"]{area};node["natural"="volcano"]{area};);out;'
    )
    data = urllib.parse.urlencode({"data": query}).encode()
    request = urllib.request.Request(OVERPASS_URL, data=data, headers={"User-Agent": "peakle/0.1"})  # noqa: S310 - fixed https host
    with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_S) as response:  # noqa: S310
        payload = json.load(response)
    summits: list[dict] = []
    for element in payload.get("elements", []):
        tags = element.get("tags", {})
        name = tags.get("name")
        if not name or "lat" not in element or "lon" not in element:
            continue
        summits.append({"name": name, "lat": float(element["lat"]), "lon": float(element["lon"])})
    return summits
