"""OSM summit-name matching tests."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from peakle.domain.coordinates import GeoPoint, LocalFrame, LocalPoint
from peakle.domain.peaks import Peak
from peakle.domain.terrain import TerrainMap, TerrainSpec
from peakle.terrain import gazetteer

LAT = 46.0


def _lon_offset(east_m: float, lat: float = LAT) -> float:
    return east_m / (111_320.0 * math.cos(math.radians(lat)))


def _peak(name: str, east_m: float, elevation_m: float, prominence_m: float) -> Peak:
    return Peak(
        id=name,
        name=name,
        local_position=LocalPoint(east_m=east_m, north_m=0.0, up_m=elevation_m),
        geo_position=GeoPoint(latitude_deg=LAT, longitude_deg=_lon_offset(east_m), elevation_m=elevation_m),
        elevation_m=elevation_m,
        prominence_m=prominence_m,
    )


def _terrain() -> TerrainMap:
    origin = GeoPoint(latitude_deg=LAT, longitude_deg=0.0, elevation_m=0.0)
    frame = LocalFrame(origin=origin)
    x_m = np.linspace(-2500.0, 2500.0, 33)
    y_m = np.linspace(-2500.0, 2500.0, 33)
    x_grid, y_grid = np.meshgrid(x_m, y_m)
    elevation_m = 3200.0 + x_grid * 0.02 - y_grid * 0.01
    latitude_deg = np.zeros_like(elevation_m)
    longitude_deg = np.zeros_like(elevation_m)
    for row, north_m in enumerate(y_m):
        for col, east_m in enumerate(x_m):
            point = LocalPoint(east_m=float(east_m), north_m=float(north_m), up_m=float(elevation_m[row, col]))
            geo = frame.local_to_geo(point)
            latitude_deg[row, col] = geo.latitude_deg
            longitude_deg[row, col] = geo.longitude_deg
    return TerrainMap(
        spec=TerrainSpec(
            origin=origin,
            width_m=5000.0,
            height_m=5000.0,
            grid_width=33,
            grid_height=33,
            min_elevation_m=float(elevation_m.min()),
            max_elevation_m=float(elevation_m.max()),
            seed=1,
        ),
        x_m=x_m,
        y_m=y_m,
        elevation_m=elevation_m,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
    )


def test_osm_naming_assigns_closest_summit_before_extended_match(monkeypatch, tmp_path: Path) -> None:
    peaks = [
        _peak("high-shoulder", 0.0, 4200.0, 800.0),
        _peak("named-summit", 900.0, 4100.0, 500.0),
    ]
    named = [
        {"name": "Close Named Summit", "lat": LAT, "lon": _lon_offset(1000.0)},
        {"name": "Extended Shoulder Name", "lat": LAT, "lon": _lon_offset(-1800.0)},
    ]
    monkeypatch.setattr(gazetteer, "_load_named_peaks", lambda _bbox, _cache_dir: named)

    renamed = gazetteer.name_peaks_from_osm(peaks, (45.9, -0.1, 46.1, 0.1), tmp_path)

    assert [peak.name for peak in renamed] == ["Extended Shoulder Name", "Close Named Summit"]


def test_osm_naming_keeps_far_local_maximum_as_spot_height(monkeypatch, tmp_path: Path) -> None:
    peaks = [_peak("unnamed", 0.0, 3456.4, 300.0)]
    named = [{"name": "Too Far Away", "lat": LAT, "lon": _lon_offset(gazetteer.EXTENDED_MATCH_RADIUS_M + 50.0)}]
    monkeypatch.setattr(gazetteer, "_load_named_peaks", lambda _bbox, _cache_dir: named)

    renamed = gazetteer.name_peaks_from_osm(peaks, (45.9, -0.1, 46.1, 0.1), tmp_path)

    assert renamed[0].name == "Pt 3456 m"


def test_osm_naming_appends_unmatched_named_summits_when_terrain_is_given(monkeypatch, tmp_path: Path) -> None:
    terrain = _terrain()
    peaks = [_peak("detected", 0.0, 4200.0, 800.0)]
    named = [
        {"name": "Detected Name", "lat": LAT, "lon": _lon_offset(0.0)},
        {"name": "Unmatched Name", "lat": LAT, "lon": _lon_offset(1500.0)},
    ]
    monkeypatch.setattr(gazetteer, "_load_named_peaks", lambda _bbox, _cache_dir: named)

    renamed = gazetteer.name_peaks_from_osm(peaks, (45.9, -0.1, 46.1, 0.1), tmp_path, terrain=terrain)

    assert [peak.name for peak in renamed] == ["Detected Name", "Unmatched Name"]
    assert renamed[1].id == "osm-002"
    assert renamed[1].prominence_m == 0.0
    assert renamed[1].elevation_m == terrain.elevation_at(renamed[1].local_position.east_m, 0.0)
