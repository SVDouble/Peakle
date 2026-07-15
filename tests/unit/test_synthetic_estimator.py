from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.coordinates import GeoPoint, LocalPoint
from peakle.domain.terrain import TerrainMap, TerrainSpec
from peakle.io.artifacts import write_once_bytes
from peakle.localize.synthetic_pipeline_bench import (
    SYNTHETIC_ARCHIVE_SCHEMA,
    SyntheticSearchConfig,
    canonical_json_bytes,
)
from peakle.rendering.rasterizer import SyntheticRenderer
from peakle.research import synthetic_estimator as worker
from peakle.research.synthetic_estimator import (
    ArtifactRef,
    EstimatorInputError,
    EstimatorIntrinsics,
    EstimatorPrior,
    EstimatorSearchConfig,
    SyntheticEstimatorRequest,
    decode_terrain_bundle,
    encode_terrain_bundle,
    execute_estimator_request,
    run_synthetic_estimator,
)


def _case() -> tuple[
    TerrainMap,
    CameraIntrinsics,
    CameraExtrinsics,
    SyntheticSearchConfig,
    bytes,
    bytes,
]:
    size = 32
    x_m = np.linspace(-500.0, 500.0, size, dtype=np.float64)
    y_m = np.linspace(-500.0, 500.0, size, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x_m, y_m)
    elevation = 40.0 + 180.0 * np.exp(-((x_grid / 180.0) ** 2 + ((y_grid - 120.0) / 150.0) ** 2))
    spec = TerrainSpec(
        origin=GeoPoint(latitude_deg=46.0, longitude_deg=7.0, elevation_m=40.0),
        width_m=1000.0,
        height_m=1000.0,
        grid_width=size,
        grid_height=size,
        min_elevation_m=40.0,
        max_elevation_m=220.0,
        seed=19,
    )
    terrain = TerrainMap(
        spec=spec,
        x_m=x_m,
        y_m=y_m,
        elevation_m=elevation,
        latitude_deg=np.full((size, size), 46.0, dtype=np.float64),
        longitude_deg=np.full((size, size), 7.0, dtype=np.float64),
    )
    intrinsics = CameraIntrinsics.from_horizontal_fov(32, 18, 60.0)
    prior = CameraExtrinsics(
        position=LocalPoint(east_m=0.0, north_m=-350.0, up_m=terrain.elevation_at(0.0, -350.0) + 2.5),
        yaw_deg=0.0,
        pitch_deg=9.0,
        roll_deg=0.0,
    )
    config = SyntheticSearchConfig(
        position_spacing_m=25.0,
        position_radius_steps=1,
        yaw_spacing_deg=4.0,
        yaw_radius_steps=1,
        render_stride=2,
        target_position_m=15.0,
        target_yaw_deg=2.0,
    )
    rendered = SyntheticRenderer().geometry(terrain, intrinsics, prior, stride=config.render_stride)
    semantic = np.asarray(rendered.terrain_mask, dtype=np.uint8)
    geometry = np.zeros((*semantic.shape, 4), dtype="<f4")
    geometry[..., 3] = np.where(semantic == 1, rendered.forward_depth_m, 0.0)
    assert np.any(semantic)
    return terrain, intrinsics, prior, config, semantic.tobytes(), geometry.tobytes()


def _request_payload(
    terrain: TerrainMap,
    intrinsics: CameraIntrinsics,
    prior: CameraExtrinsics,
    config: SyntheticSearchConfig,
    semantic: bytes,
    geometry: bytes,
) -> dict[str, Any]:
    request = SyntheticEstimatorRequest(
        estimator_terrain=ArtifactRef.for_bytes("terrain.json", encode_terrain_bundle(terrain)),
        webgl_semantic_u8=ArtifactRef.for_bytes("semantic.u8", semantic),
        webgl_geometry_f32le=ArtifactRef.for_bytes("geometry.f32le", geometry),
        intrinsics=EstimatorIntrinsics(**intrinsics.model_dump()),
        prior=EstimatorPrior(**prior.model_dump()),
        search_config=EstimatorSearchConfig(**asdict(config)),
        output_filename="archive.json",
    )
    return request.model_dump(mode="json", by_alias=True)


def test_terrain_bundle_roundtrip_is_exact_and_deterministic() -> None:
    terrain, *_rest = _case()

    encoded = encode_terrain_bundle(terrain)
    decoded = decode_terrain_bundle(encoded)

    assert encode_terrain_bundle(decoded) == encoded
    assert decoded.spec == terrain.spec
    for name in ("x_m", "y_m", "elevation_m", "latitude_deg", "longitude_deg"):
        assert np.array_equal(getattr(decoded, name), getattr(terrain, name))


@pytest.mark.parametrize("field", ["truth", "query_extrinsics", "camera_matrix", "expected_decision"])
def test_request_recursively_rejects_forbidden_truth_side_fields(field: str) -> None:
    case = _case()
    payload = _request_payload(*case)
    payload["intrinsics"][field] = {"nested": 1}

    with pytest.raises(ValueError, match="forbidden estimator request field"):
        SyntheticEstimatorRequest.model_validate(payload)


def test_request_is_strict_and_refs_are_safe() -> None:
    case = _case()
    payload = _request_payload(*case)
    payload["innocent_extra"] = 1
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        SyntheticEstimatorRequest.model_validate(payload)
    with pytest.raises(ValueError, match="safe path component"):
        ArtifactRef.for_bytes("../terrain.json", b"x")


def test_worker_rejects_content_tampering(tmp_path) -> None:
    terrain, intrinsics, prior, config, semantic, geometry = _case()
    terrain_bytes = encode_terrain_bundle(terrain)
    payload = _request_payload(terrain, intrinsics, prior, config, semantic, geometry)
    write_once_bytes(tmp_path / "terrain.json", terrain_bytes)
    write_once_bytes(tmp_path / "semantic.u8", semantic[:-1] + bytes([semantic[-1] ^ 1]))
    write_once_bytes(tmp_path / "geometry.f32le", geometry)
    request_path = tmp_path / "request.json"
    write_once_bytes(request_path, canonical_json_bytes(payload))

    with pytest.raises(EstimatorInputError, match="hash or byte count mismatch"):
        execute_estimator_request(request_path)
    assert not (tmp_path / "archive.json").exists()


def test_parent_runs_worker_and_returns_canonical_immutable_artifacts() -> None:
    terrain, intrinsics, prior, config, semantic, geometry = _case()

    result = run_synthetic_estimator(
        terrain,
        intrinsics,
        prior,
        config,
        semantic_u8=semantic,
        geometry_f32le=geometry,
        terrain_filename="terrain-scene-exact.json",
        semantic_filename="query-scene.semantic.u8",
        geometry_filename="query-scene.geometry.f32le",
        request_filename="request-scene-exact.json",
        output_filename="archive-scene-exact.json",
    )

    assert result.archive["schema"] == SYNTHETIC_ARCHIVE_SCHEMA
    assert result.archive["candidate_pool"]["candidate_count"] == 27
    assert result.archive_artifact.content == canonical_json_bytes(result.archive)
    assert result.request_artifact.content == canonical_json_bytes(
        result.request.model_dump(mode="json", by_alias=True)
    )
    assert result.terrain_artifact.content == encode_terrain_bundle(terrain)
    assert result.request.estimator_terrain.filename == "terrain-scene-exact.json"
    assert result.request.webgl_semantic_u8.filename == "query-scene.semantic.u8"
    assert result.request.webgl_geometry_f32le.filename == "query-scene.geometry.f32le"
    assert result.request_artifact.ref.filename == "request-scene-exact.json"
    assert result.archive_artifact.ref.filename == "archive-scene-exact.json"
    assert b'"truth"' not in result.request_artifact.content


def test_parent_monkeypatch_cannot_cross_subprocess_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    terrain, intrinsics, prior, config, semantic, geometry = _case()

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("parent-process builder was called")

    monkeypatch.setattr(worker, "build_synthetic_candidate_archive", fail_if_called)
    result = run_synthetic_estimator(
        terrain,
        intrinsics,
        prior,
        config,
        semantic_u8=semantic,
        geometry_f32le=geometry,
    )

    assert result.archive["candidate_pool"]["candidate_count"] == 27
