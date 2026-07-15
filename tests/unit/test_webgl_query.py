"""Calibration and truth-firewall tests for the independent WebGL renderer."""

from __future__ import annotations

import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.coordinates import GeoPoint, LocalPoint
from peakle.domain.terrain import TerrainMap, TerrainSpec
from peakle.localize.synthetic_pipeline_bench import SyntheticSearchConfig
from peakle.rendering.rasterizer import SyntheticRenderer
from peakle.research import synthetic_query
from peakle.research.synthetic_query import build_synthetic_query_observations
from peakle.research.webgl_contract import QueryArtifactFile, freeze_webgl_query_artifact
from peakle.research.webgl_query import render_webgl_mesh_query, terrain_mesh_buffers

CHROMIUM_AVAILABLE = shutil.which("chromium") is not None


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        width_px=9,
        height_px=7,
        focal_length_px=4.0,
        principal_x_px=4.0,
        principal_y_px=3.0,
    )


def _camera(**changes: float) -> CameraExtrinsics:
    values = {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}
    values.update(changes)
    return CameraExtrinsics(
        position=LocalPoint(east_m=0.0, north_m=0.0, up_m=0.0),
        **values,
    )


def _front_plane(north_m: float) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(
        [
            [-20.0, north_m, -20.0],
            [20.0, north_m, -20.0],
            [-20.0, north_m, 20.0],
            [20.0, north_m, 20.0],
        ],
        dtype=np.float32,
    )
    triangles = np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.uint32)
    return positions, triangles


def _two_planes() -> tuple[np.ndarray, np.ndarray]:
    near_positions, plane_triangles = _front_plane(5.0)
    far_positions, _ = _front_plane(10.0)
    positions = np.concatenate((near_positions, far_positions))
    triangles = np.concatenate((plane_triangles, plane_triangles + 4))
    return positions, triangles


def _flat_terrain() -> TerrainMap:
    size = 32
    x_m = np.linspace(-100.0, 100.0, size)
    y_m = np.linspace(10.0, 210.0, size)
    x_grid, y_grid = np.meshgrid(x_m, y_m)
    spec = TerrainSpec(
        origin=GeoPoint(latitude_deg=46.0, longitude_deg=8.0, elevation_m=0.0),
        width_m=200.0,
        height_m=200.0,
        grid_width=size,
        grid_height=size,
        min_elevation_m=0.0,
        max_elevation_m=1.0,
        seed=0,
    )
    return TerrainMap(
        spec=spec,
        x_m=x_m,
        y_m=y_m,
        elevation_m=np.zeros((size, size)),
        latitude_deg=46.0 + y_grid / 111_320.0,
        longitude_deg=8.0 + x_grid / 80_000.0,
    )


def test_terrain_mesh_interchange_has_explicit_upward_wound_topology() -> None:
    positions, triangles = terrain_mesh_buffers(_flat_terrain())

    assert positions.dtype == np.dtype("float32")
    assert triangles.dtype == np.dtype("uint32")
    assert positions.shape == (32 * 32, 3)
    assert triangles.shape == (31 * 31 * 2, 3)
    assert triangles[:2].tolist() == [[0, 1, 32], [1, 33, 32]]
    first = positions[triangles[0]]
    assert np.cross(first[1] - first[0], first[2] - first[0])[2] > 0.0


def test_neutral_mesh_contract_rejects_invalid_topology() -> None:
    positions, triangles = _front_plane(5.0)

    with pytest.raises(ValueError, match="outside"):
        render_webgl_mesh_query(positions, np.asarray(((0, 1, 99),), dtype=np.uint32), _intrinsics(), _camera())
    with pytest.raises(ValueError, match="repeat"):
        render_webgl_mesh_query(positions, np.asarray(((0, 1, 1),), dtype=np.uint32), _intrinsics(), _camera())
    with pytest.raises(ValueError, match="non-zero area"):
        render_webgl_mesh_query(
            np.asarray(((0, 5, 0), (1, 5, 0), (2, 5, 0)), dtype=np.float32),
            np.asarray(((0, 1, 2),), dtype=np.uint32),
            _intrinsics(),
            _camera(),
        )
    with pytest.raises(ValueError, match="integer dtype"):
        render_webgl_mesh_query(positions, triangles.astype(np.float64), _intrinsics(), _camera())
    with pytest.raises(ValueError, match="non-negative"):
        render_webgl_mesh_query(positions, np.asarray(((-1, 1, 2),), dtype=np.int64), _intrinsics(), _camera())
    with pytest.raises(ValueError, match="uint32"):
        render_webgl_mesh_query(
            positions,
            np.asarray(((0, 1, 2**32),), dtype=np.int64),
            _intrinsics(),
            _camera(),
        )


def test_webgl_observation_provider_freezes_before_return_and_omits_shared_rgb_tracks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terrain = _flat_terrain()
    intrinsics = CameraIntrinsics.from_horizontal_fov(32, 18, 55.0)
    camera = CameraExtrinsics(
        position=LocalPoint(east_m=0.0, north_m=0.0, up_m=20.0),
        yaw_deg=0.0,
        pitch_deg=-20.0,
        roll_deg=0.0,
    )
    mask = np.zeros((18, 32), dtype=bool)
    mask[8:] = True
    fake_query = SimpleNamespace(
        skyline_profile=np.full(32, 8.0),
        rgb=np.zeros((18, 32, 3), dtype=np.uint8),
        forward_depth_m=np.where(mask, 100.0, np.nan),
        provenance=SimpleNamespace(model_dump=lambda **_kwargs: {"renderer_family": "test-webgl"}),
    )
    file_record = QueryArtifactFile(
        filename="query-scene.geometry.f32le",
        role="normal_xyz_depth_f32le",
        media_type="application/octet-stream",
        dtype="float32_little_endian",
        shape=(18, 32, 4),
        sha256="0" * 64,
        bytes=18 * 32 * 16,
    )
    fake_frozen = SimpleNamespace(files={file_record.filename: b"frozen"}, manifest=(file_record,))

    def reject_shared_renderer() -> None:
        raise AssertionError("webgl observations called the shared renderer")

    monkeypatch.setattr(synthetic_query, "render_terrain_webgl_query", lambda *_args, **_kwargs: fake_query)
    monkeypatch.setattr(synthetic_query, "freeze_webgl_query_artifact", lambda *_args, **_kwargs: fake_frozen)
    monkeypatch.setattr(synthetic_query, "SyntheticRenderer", reject_shared_renderer)

    observations = build_synthetic_query_observations(
        {"scene_id": "scene", "terrain": terrain, "truth": camera},
        intrinsics,
        SyntheticSearchConfig(render_stride=2),
        query_renderer="webgl",
    )

    assert set(observations.profiles) == {"oracle_mask"}
    assert observations.expected_actions == {"oracle_mask": "select"}
    assert observations.artifact_files == {file_record.filename: b"frozen"}
    assert observations.artifact_manifest == (file_record,)
    assert observations.metadata["oracle_mask"]["sealed_truth_side_query_provenance_ref"] == "scene"
    assert observations.metadata["oracle_mask"]["provenance_supplied_to_candidate_builder"] is False
    assert observations.query_provenance["renderer_family"] == "test-webgl"


@pytest.mark.skipif(not CHROMIUM_AVAILABLE, reason="system Chromium is an optional research runtime")
def test_webgl_query_renders_metric_depth_normals_and_nearest_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_shared_renderer(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the independent query called SyntheticRenderer")

    monkeypatch.setattr(SyntheticRenderer, "geometry", reject_shared_renderer)
    positions, triangles = _two_planes()
    rendered = render_webgl_mesh_query(positions, triangles, _intrinsics(), _camera())

    assert rendered.terrain_mask.all()
    assert np.allclose(rendered.forward_depth_m, 5.0, atol=2e-6)
    assert np.allclose(rendered.camera_normals, (0.0, 0.0, -1.0), atol=2e-6)
    assert rendered.skyline_profile.tolist() == [0.0] * 9
    assert rendered.rgb[0, 4].mean() > rendered.rgb[-1, 4].mean()
    assert rendered.provenance.independence_class == "independent_rasterizer_same_scene_model"
    assert rendered.provenance.calls_peakle_renderer_or_projection_helpers is False
    assert "SwiftShader" in (rendered.provenance.capabilities.unmasked_renderer or "")
    assert any(resource.name == "libvk_swiftshader.so" for resource in rendered.provenance.renderer_runtime_resources)
    assert rendered.rgb.flags.writeable is False
    assert rendered.forward_depth_m.flags.writeable is False
    frozen = freeze_webgl_query_artifact(rendered, "analytic-two-plane")
    assert len(frozen.files) == 4
    assert {record.sha256 for record in frozen.manifest} == {
        rendered.provenance.input_scene_sha256,
        rendered.provenance.output_hashes.rgb_rgba8_sha256,
        rendered.provenance.output_hashes.normal_xyz_depth_f32le_sha256,
        rendered.provenance.output_hashes.semantic_u8_sha256,
    }


@pytest.mark.skipif(not CHROMIUM_AVAILABLE, reason="system Chromium is an optional research runtime")
def test_webgl_query_is_byte_deterministic_and_reports_camera_axes() -> None:
    positions, triangles = _front_plane(10.0)

    first = render_webgl_mesh_query(positions, triangles, _intrinsics(), _camera(roll_deg=90.0))
    second = render_webgl_mesh_query(positions, triangles, _intrinsics(), _camera(roll_deg=90.0))

    assert first.provenance.output_hashes == second.provenance.output_hashes
    assert first.provenance.renderer_stratum_sha256 == second.provenance.renderer_stratum_sha256
    assert np.allclose(first.provenance.camera_world_axes.forward, (0.0, 1.0, 0.0), atol=1e-12)
    assert np.allclose(first.provenance.camera_world_axes.right, (0.0, 0.0, -1.0), atol=1e-12)
    assert np.allclose(first.provenance.camera_world_axes.down, (-1.0, 0.0, 0.0), atol=1e-12)
    assert first.provenance.camera_world_from_rdf_matrix_col_major == pytest.approx(
        (0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    )


@pytest.mark.skipif(not CHROMIUM_AVAILABLE, reason="system Chromium is an optional research runtime")
def test_webgl_query_honours_off_center_principal_point_and_pixel_centres() -> None:
    forward = np.asarray((3**0.5 / 2.0, 0.0, 0.5))
    right = np.asarray((0.0, -1.0, 0.0))
    down = np.asarray((0.5, 0.0, -(3**0.5 / 2.0)))
    centre = 10.0 * forward
    extent = 1.5
    positions = np.asarray(
        (
            centre - extent * right + extent * down,
            centre + extent * right + extent * down,
            centre - extent * right - extent * down,
            centre + extent * right - extent * down,
        ),
        dtype=np.float32,
    )
    triangles = np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.uint32)
    intrinsics = _intrinsics().model_copy(update={"principal_x_px": 2.0, "principal_y_px": 1.0})

    rendered = render_webgl_mesh_query(positions, triangles, intrinsics, _camera(yaw_deg=90.0, pitch_deg=30.0))

    assert np.argwhere(rendered.terrain_mask).tolist() == [[1, 2]]
    assert rendered.forward_depth_m[1, 2] == pytest.approx(10.0, abs=2e-6)
    assert rendered.provenance.camera_world_axes.forward == pytest.approx(forward, abs=1e-12)
    assert rendered.provenance.camera_world_axes.right == pytest.approx(right, abs=1e-12)
    assert rendered.provenance.camera_world_axes.down == pytest.approx(down, abs=1e-12)


@pytest.mark.skipif(not CHROMIUM_AVAILABLE, reason="system Chromium is an optional research runtime")
def test_webgl_query_slanted_plane_has_analytic_depth_and_camera_normal() -> None:
    # Plane north = 10 + east. At image column u, east/north = (u-cx)/f,
    # so forward depth is 10 / (1 - (u-cx)/f).
    positions = np.asarray(
        ((-5.0, 5.0, -20.0), (5.0, 15.0, -20.0), (-5.0, 5.0, 20.0), (5.0, 15.0, 20.0)),
        dtype=np.float32,
    )
    triangles = np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.uint32)

    rendered = render_webgl_mesh_query(positions, triangles, _intrinsics(), _camera())

    assert rendered.forward_depth_m[3, 3] == pytest.approx(8.0, abs=0.03)
    assert rendered.forward_depth_m[3, 4] == pytest.approx(10.0, abs=0.05)
    assert rendered.forward_depth_m[3, 5] == pytest.approx(40.0 / 3.0, abs=0.1)
    assert rendered.camera_normals[3, 4] == pytest.approx((2**-0.5, 0.0, -(2**-0.5)), abs=2e-6)


@pytest.mark.skipif(not CHROMIUM_AVAILABLE, reason="system Chromium is an optional research runtime")
def test_webgl_query_clips_a_triangle_crossing_the_near_plane() -> None:
    positions = np.asarray(((-4.0, 0.5, -4.0), (4.0, 5.0, -4.0), (0.0, 5.0, 4.0)), dtype=np.float32)
    triangles = np.asarray(((0, 1, 2),), dtype=np.uint32)

    rendered = render_webgl_mesh_query(positions, triangles, _intrinsics(), _camera())

    assert rendered.terrain_mask.any()
    assert np.nanmin(rendered.forward_depth_m) >= 1.0
