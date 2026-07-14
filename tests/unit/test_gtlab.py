"""GT Lab API helpers."""

import asyncio
import math
from types import SimpleNamespace

import numpy as np
from PIL import Image

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


def test_open_gt_view_separates_photo_default_from_pfm_oracle(monkeypatch, tmp_path) -> None:
    photo_path = tmp_path / "photo.jpg"
    Image.new("RGB", (8, 4), (100, 140, 180)).save(photo_path)
    sample = SimpleNamespace(
        name="sample-a",
        photo_path=photo_path,
        depth_path=tmp_path / "depth.pfm",
        lat=46.0,
        lon=7.0,
        elev_m=1500.0,
        yaw_gt_deg=42.0,
        pitch_gt_deg=3.0,
        roll_gt_deg=2.0,
    )
    rec = {
        "name": "sample-a",
        "width": 8,
        "height": 4,
        "fov_deg": 50.0,
        "yaw_deg": 170.0,
        "de_m": 120.0,
        "dn_m": -80.0,
        "cam_z_m": 3100.0,
        "dv_px": 40.0,
    }

    class Frame:
        def geo_to_local(self, _point):
            return LocalPoint(east_m=5.0, north_m=6.0, up_m=0.0)

    terrain = SimpleNamespace(
        frame=Frame(),
        x_m=np.asarray([-100.0, 100.0]),
        y_m=np.asarray([-100.0, 100.0]),
        elevation_at=lambda _east, _north: 2000.0,
    )

    class Scene:
        def __init__(self):
            self.terrain = terrain
            self.captured = None

        def add_gt_view(self, *args, **kwargs):
            self.captured = (args, kwargs)
            return "view"

    scene = Scene()
    monkeypatch.setattr(gtlab, "_index", lambda: {"sample-a": rec})
    monkeypatch.setattr(gtlab, "load_sample", lambda _path: sample)
    monkeypatch.setattr(gtlab, "resampled_oracle_skyline", lambda *_args: np.asarray([1.0] * 8))
    photo_candidate = SimpleNamespace(rows=np.asarray([2.0] * 8), coverage=1.0, agreement=0.75)
    monkeypatch.setattr(gtlab, "extract_candidates", lambda *_args, **_kwargs: {"color": photo_candidate})

    assert gtlab._open_gt_view(scene, "sample-a") == "view"
    args, kwargs = scene.captured
    extrinsics = args[2]
    contour = args[3]
    prior = kwargs["prior"]
    assert extrinsics.yaw_deg == 42.0
    assert extrinsics.pitch_deg == 3.0
    assert extrinsics.position == LocalPoint(east_m=5.0, north_m=6.0, up_m=1500.0)
    assert contour.source == "photo_auto:color"
    assert prior.position == extrinsics.position
    assert prior.horizontal_sigma_m == 200.0
    assert prior.yaw_sigma_deg == 15.0
    assert kwargs["default_evidence_source"] == "photo_auto"
    assert kwargs["pitch_comparable"] is False
    assert kwargs["evidence_contours"]["photo_auto"].source == "photo_auto:color"
    assert kwargs["evidence_contours"]["pfm_oracle"].source == "pfm_oracle"
    assert kwargs["evidence_metadata"]["photo_auto"]["selection_uses_ground_truth"] is False
    assert kwargs["evidence_metadata"]["pfm_oracle"]["diagnostic"] is True


def test_open_gt_view_does_not_fall_back_when_photo_evidence_is_unavailable(monkeypatch, tmp_path) -> None:
    photo_path = tmp_path / "photo.jpg"
    Image.new("RGB", (8, 4), (100, 140, 180)).save(photo_path)
    sample = SimpleNamespace(
        photo_path=photo_path,
        depth_path=tmp_path / "depth.pfm",
        lat=46.0,
        lon=7.0,
        elev_m=1500.0,
        yaw_gt_deg=42.0,
        pitch_gt_deg=3.0,
        roll_gt_deg=0.0,
    )
    rec = {"name": "sample-a", "width": 8, "height": 4, "fov_deg": 50.0}

    class Frame:
        def geo_to_local(self, _point):
            return LocalPoint(east_m=0.0, north_m=0.0, up_m=0.0)

    terrain = SimpleNamespace(
        frame=Frame(),
        x_m=np.asarray([-1.0, 1.0]),
        y_m=np.asarray([-1.0, 1.0]),
    )

    class Scene:
        def __init__(self):
            self.terrain = terrain
            self.kwargs = None

        def add_gt_view(self, *_args, **kwargs):
            self.kwargs = kwargs
            return "view"

    scene = Scene()
    monkeypatch.setattr(gtlab, "_index", lambda: {"sample-a": rec})
    monkeypatch.setattr(gtlab, "load_sample", lambda _path: sample)
    monkeypatch.setattr(gtlab, "resampled_oracle_skyline", lambda *_args: np.asarray([1.0] * 8))
    monkeypatch.setattr(gtlab, "extract_candidates", lambda *_args, **_kwargs: {})

    assert gtlab._open_gt_view(scene, "sample-a") == "view"
    assert scene.kwargs["default_evidence_source"] == "photo_auto"
    assert "photo_auto" not in scene.kwargs["evidence_contours"]
    assert scene.kwargs["evidence_metadata"]["photo_auto"]["available"] is False
    assert "pfm_oracle" in scene.kwargs["evidence_contours"]


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
