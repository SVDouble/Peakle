"""GT Lab API helpers."""

import asyncio
import math
from types import SimpleNamespace

import numpy as np

from peakle.domain.coordinates import LocalPoint
from peakle.web.api import gtlab
from peakle.web.api.gtlab import LAYER_NAMES, _rows_contour, _scene_rows, _visible_peak_tags


def test_pfm_skyline_layer_is_registered() -> None:
    assert "pfm_sky" in LAYER_NAMES


def test_alignment_audit_endpoint_uses_all_records(monkeypatch) -> None:
    monkeypatch.setattr(
        gtlab,
        "_index",
        lambda: {
            "ok": {"name": "ok", "quality": "CLEAN", "sky_cons_px": 2.0, "pfm_cons_px": 2.0},
            "bad": {"name": "bad", "quality": "SUSPECT", "sky_cons_px": 40.0, "pfm_cons_px": 5.0},
        },
    )

    report = asyncio.run(gtlab.gt_alignment_audit(limit=10, include_clean=False))

    assert report["total"] == 2
    assert report["rows"][0]["name"] == "bad"


def test_alignment_audit_endpoint_can_enrich_metric_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        gtlab,
        "_index",
        lambda: {"bad": {"name": "bad", "quality": "SUSPECT", "sky_cons_px": 40.0, "pfm_cons_px": 5.0}},
    )
    monkeypatch.setattr(
        gtlab,
        "metric_skyline_errors_for_record",
        lambda _rec: {"sky_error_m": 120.0, "sky_error_p90_m": 250.0, "sky_range_median_m": 8000.0},
    )

    report = asyncio.run(gtlab.gt_alignment_audit(limit=10, include_clean=False, metric=True))

    assert report["rows"][0]["metrics"]["sky_error_m"] == 120.0
    assert report["rows"][0]["metrics"]["sky_range_median_m"] == 8000.0


def test_rows_contour_uses_finite_in_frame_rows() -> None:
    rows = np.asarray([10.0, math.nan, -1.0, 20.5, 99.0, 8.0])

    contour = _rows_contour(rows, 6, 50, source="gt_skyline")

    assert contour.source == "gt_skyline"
    assert [(p.x_px, p.y_px) for p in contour.points] == [(0.0, 10.0), (3.0, 20.5), (5.0, 8.0)]


def test_scene_rows_uses_current_terrain_frame_and_refined_offsets(monkeypatch) -> None:
    class Frame:
        def geo_to_local(self, point):
            assert point.latitude_deg == 46.0
            assert point.longitude_deg == 7.0
            return LocalPoint(east_m=10.0, north_m=20.0, up_m=0.0)

    origin = SimpleNamespace(latitude_deg=46.0, longitude_deg=7.0)
    terrain = SimpleNamespace(
        frame=Frame(),
        spec=SimpleNamespace(origin=origin),
        x_m=np.asarray([-100.0, 100.0]),
        y_m=np.asarray([-50.0, 50.0]),
    )
    scene = SimpleNamespace(terrain=terrain)
    rec = {
        "width": 4,
        "height": 3,
        "fov_deg": 30.0,
        "yaw_deg": 100.0,
        "de_m": 3.0,
        "dn_m": 4.0,
        "cam_z_m": 2100.0,
        "dv_px": 2.0,
    }

    def fake_dem_skyline(arg_terrain, cam_z, _az, _w, _h, _fov, de, dn, patch=None):
        assert arg_terrain is terrain
        assert cam_z == 2100.0
        assert de == 18.0
        assert dn == 30.0
        assert patch is None
        return np.asarray([1.0, math.nan, 3.5, 4.0])

    monkeypatch.setattr(gtlab, "dem_skyline", fake_dem_skyline)
    monkeypatch.setattr(gtlab, "_scene_skyline_patch", lambda _scene: None)

    rows = _scene_rows(scene, rec, 46.0, 7.0, dyaw=1.0, de=5.0, dn=6.0, dv=7.0)

    assert rows == {"rows": [10.0, None, 12.5, 13.0], "skyline_resolution_m": 200.0, "skyline_patch": None}


def test_scene_rows_uses_high_resolution_patch_when_available(monkeypatch) -> None:
    class Frame:
        def geo_to_local(self, _point):
            return LocalPoint(east_m=0.0, north_m=0.0, up_m=0.0)

    hires_patch = object()
    origin = SimpleNamespace(latitude_deg=46.0, longitude_deg=7.0)
    terrain = SimpleNamespace(
        frame=Frame(),
        spec=SimpleNamespace(origin=origin),
        x_m=np.asarray([-1.0, 1.0]),
        y_m=np.asarray([-1.0, 1.0]),
    )
    scene = SimpleNamespace(terrain=terrain)
    rec = {
        "width": 2,
        "height": 2,
        "fov_deg": 30.0,
        "yaw_deg": 100.0,
        "de_m": 0.0,
        "dn_m": 0.0,
        "cam_z_m": 2100.0,
        "dv_px": 0.0,
    }

    def fake_dem_skyline(*_args, patch=None):
        assert patch is hires_patch
        return np.asarray([1.0, 2.0])

    monkeypatch.setattr(gtlab, "dem_skyline", fake_dem_skyline)
    monkeypatch.setattr(gtlab, "_scene_skyline_patch", lambda _scene: hires_patch)

    rows = _scene_rows(scene, rec, 46.0, 7.0, dyaw=0.0, de=0.0, dn=0.0, dv=0.0)

    assert rows == {"rows": [1.0, 2.0], "skyline_resolution_m": 5.0, "skyline_patch": "swissalti3d"}


def test_visible_peak_tags_rank_names_in_camera_fov(monkeypatch) -> None:
    rec = {"yaw_deg": 270.0, "fov_deg": 30.0, "de_m": 0.0, "dn_m": 0.0}
    monkeypatch.setattr(
        gtlab,
        "_named_summits",
        lambda: (
            {"name": "Matterhorn", "lat": 46.0, "lon": 6.91},
            {"name": "Side Peak", "lat": 46.005, "lon": 6.93},
            {"name": "Behind Peak", "lat": 46.0, "lon": 7.1},
        ),
    )

    tags = _visible_peak_tags(rec, 46.0, 7.0)

    assert [tag["name"] for tag in tags] == ["Matterhorn", "Side Peak"]
    assert tags[0]["weight"] > tags[1]["weight"]
