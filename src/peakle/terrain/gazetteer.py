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

from peakle.domain.peaks import Peak

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
MATCH_RADIUS_M = 750.0
_REQUEST_TIMEOUT_S = 30


def name_peaks_from_osm(peaks: list[Peak], bbox: tuple[float, float, float, float], cache_dir: Path) -> list[Peak]:
    """Renames detected peaks using nearby OSM summit names.

    Args:
        peaks: Detected peaks (already ordered by prominence).
        bbox: ``(south, west, north, east)`` in degrees.
        cache_dir: Directory for the per-bbox Overpass cache file.

    Returns:
        Peaks with real names where a nearby OSM summit exists, otherwise a spot
        height. Order and all other fields are preserved.
    """

    named = _load_named_peaks(bbox, cache_dir)
    used: set[int] = set()
    renamed: list[Peak] = []
    for peak in peaks:
        best_index = _nearest_unused(peak, named, used)
        if best_index is None:
            renamed.append(peak.model_copy(update={"name": f"Pt {round(peak.elevation_m)} m"}))
        else:
            used.add(best_index)
            renamed.append(peak.model_copy(update={"name": named[best_index]["name"]}))
    return renamed


def _nearest_unused(peak: Peak, named: list[dict], used: set[int]) -> int | None:
    lat = peak.geo_position.latitude_deg
    lon = peak.geo_position.longitude_deg
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    best_index: int | None = None
    best_distance = MATCH_RADIUS_M
    for index, summit in enumerate(named):
        if index in used:
            continue
        north_m = (summit["lat"] - lat) * 111_320.0
        east_m = (summit["lon"] - lon) * meters_per_deg_lon
        distance = math.hypot(north_m, east_m)
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def _load_named_peaks(bbox: tuple[float, float, float, float], cache_dir: Path) -> list[dict]:
    south, west, north, east = bbox
    cache = Path(cache_dir) / f"osm_peaks_{south:.3f}_{west:.3f}_{north:.3f}_{east:.3f}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except (OSError, ValueError):
            pass
    try:
        named = _query_overpass(bbox)
    except (OSError, ValueError, TimeoutError):
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
        f"[out:json][timeout:{_REQUEST_TIMEOUT_S}];"
        f'(node["natural"="peak"]{area};node["natural"="volcano"]{area};);out;'
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
