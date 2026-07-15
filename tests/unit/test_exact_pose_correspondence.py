"""Registered exact-pose correspondence grading tests."""

from dataclasses import replace

import numpy as np
import pytest

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.coordinates import LocalPoint
from peakle.localize.correspondence import MatchSet
from peakle.rendering.terrain_view import TerrainRenderBundle, unproject_pinhole_depth
from peakle.research.exact_pose_correspondence import (
    EXACT_POSE_CASE_GATE,
    cross_render_calibration,
    freeze_match_artifact,
    grade_frozen_exact_pose_correspondences,
    identical_query_calibration,
    load_frozen_match_artifact,
)
from peakle.research.webgl_contract import LoadedWebGLQueryArtifact


def _geometry(
    depth_m: float = 100.0,
) -> tuple[
    LoadedWebGLQueryArtifact,
    TerrainRenderBundle,
    CameraIntrinsics,
    CameraExtrinsics,
]:
    size = 32
    intrinsics = CameraIntrinsics(
        width_px=size,
        height_px=size,
        focal_length_px=20.0,
        principal_x_px=15.5,
        principal_y_px=15.5,
    )
    extrinsics = CameraExtrinsics(
        position=LocalPoint(east_m=3.0, north_m=-5.0, up_m=7.0),
        yaw_deg=12.0,
        pitch_deg=-4.0,
        roll_deg=1.0,
    )
    mask = np.ones((size, size), dtype=np.bool_)
    depth = np.full((size, size), depth_m, dtype=np.float64)
    query = LoadedWebGLQueryArtifact(
        rgb=np.zeros((size, size, 3), dtype=np.uint8),
        terrain_mask=mask,
        forward_depth_m=depth,
        camera_normals=np.zeros((size, size, 3), dtype=np.float64),
    )
    render = TerrainRenderBundle(
        rgb=np.zeros((size, size, 3), dtype=np.uint8),
        forward_depth_m=depth,
        world_xyz_m=unproject_pinhole_depth(depth, intrinsics, extrinsics),
        world_normals=np.zeros((size, size, 3), dtype=np.float64),
        terrain_mask=mask,
        appearance_mask=mask,
        skyline_profile=np.zeros(size, dtype=np.float64),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        modality="hillshade",
        provenance={"schema": "exact_pose_test_v1"},
    )
    return query, render, intrinsics, extrinsics


def _matches(query_xy: np.ndarray, render_xy: np.ndarray | None = None) -> MatchSet:
    count = query_xy.shape[0]
    return MatchSet(
        query_xy_px=np.asarray(query_xy, dtype=np.float64),
        render_xy_px=np.asarray(query_xy if render_xy is None else render_xy, dtype=np.float64),
        confidence=np.linspace(0.2, 0.9, count),
        selected=np.ones(count, dtype=np.bool_),
    )


def test_frozen_match_artifact_detects_corruption_and_is_read_only() -> None:
    frozen = freeze_match_artifact(_matches(np.asarray([[8.0, 9.0], [12.0, 13.0]])), "roma.case-01")
    loaded = load_frozen_match_artifact(frozen.manifest, frozen.files)

    assert loaded.count == 2
    assert loaded.query_xy_px.flags.writeable is False
    assert loaded.render_xy_px.flags.writeable is False
    assert loaded.confidence.flags.writeable is False
    assert loaded.selected is not None
    assert loaded.selected.flags.writeable is False

    filename = frozen.manifest[0].filename
    damaged = bytearray(frozen.files[filename])
    damaged[0] ^= 1
    with pytest.raises(ValueError, match="SHA-256"):
        load_frozen_match_artifact(frozen.manifest, {filename: bytes(damaged)})
    with pytest.raises(ValueError, match="artifact_id"):
        freeze_match_artifact(_matches(np.empty((0, 2))), "unsafe/name")


def test_exact_pixel_match_is_geographically_correct() -> None:
    query, render, intrinsics, extrinsics = _geometry()
    frozen = freeze_match_artifact(_matches(np.asarray([[8.25, 9.5]])), "exact")

    grade = grade_frozen_exact_pose_correspondences(
        frozen.manifest, frozen.files, query, intrinsics, extrinsics, render
    )

    assert grade["raw_count"] == grade["selected_count"] == 1
    assert grade["query_lift_valid_count"] == grade["render_lift_valid_count"] == 1
    assert grade["correct_count"] == 1
    assert grade["precision"] == 1.0
    assert grade["confidence"]["minimum"] == pytest.approx(0.2)


def test_pixel_match_with_wrong_geography_is_not_correct() -> None:
    query, render, intrinsics, extrinsics = _geometry()
    frozen = freeze_match_artifact(
        _matches(np.asarray([[8.0, 9.0]]), np.asarray([[16.0, 9.0]])),
        "wrong",
    )

    grade = grade_frozen_exact_pose_correspondences(
        frozen.manifest, frozen.files, query, intrinsics, extrinsics, render
    )

    assert grade["both_lifts_valid_count"] == 1
    assert grade["correct_count"] == 0
    assert grade["precision"] == 0.0


@pytest.mark.parametrize(
    ("query_depth_m", "render_depth_m", "expected_correct"),
    ((100.0, 125.0, True), (100.0, 125.1, False), (5_000.0, 5_049.0, True), (5_000.0, 5_051.0, False)),
)
def test_world_distance_gate_uses_maximum_of_25m_and_one_percent_range(
    query_depth_m: float,
    render_depth_m: float,
    expected_correct: bool,
) -> None:
    query, render, intrinsics, extrinsics = _geometry(query_depth_m)
    render_depth = np.full_like(render.forward_depth_m, render_depth_m)
    render = replace(
        render,
        forward_depth_m=render_depth,
        world_xyz_m=unproject_pinhole_depth(render_depth, intrinsics, extrinsics),
    )
    frozen = freeze_match_artifact(_matches(np.asarray([[15.5, 15.5]])), "distance-threshold")

    grade = grade_frozen_exact_pose_correspondences(
        frozen.manifest, frozen.files, query, intrinsics, extrinsics, render
    )

    assert bool(grade["correct_count"]) is expected_correct


def test_query_sky_endpoint_is_invalid_and_counts_as_incorrect() -> None:
    query, render, intrinsics, extrinsics = _geometry()
    query = replace(
        query,
        terrain_mask=np.zeros_like(query.terrain_mask),
        forward_depth_m=np.full_like(query.forward_depth_m, np.nan),
    )
    frozen = freeze_match_artifact(_matches(np.asarray([[15.5, 15.5]])), "query-sky")

    grade = grade_frozen_exact_pose_correspondences(
        frozen.manifest, frozen.files, query, intrinsics, extrinsics, render
    )

    assert grade["query_lift_valid_count"] == 0
    assert grade["render_lift_valid_count"] == 1
    assert grade["correct_count"] == 0


def test_exact_pose_case_gate_includes_every_frozen_boundary() -> None:
    def passes(
        selected: int = 24,
        lift: float = 0.80,
        correct: int = 18,
        precision: float = 0.70,
        cells: int = 6,
        x_span: float = 0.40,
        y_span: float = 0.15,
    ) -> bool:
        return EXACT_POSE_CASE_GATE.evaluate(
            selected_count=selected,
            render_lift_valid_fraction=lift,
            correct_count=correct,
            precision=precision,
            occupied_cells=cells,
            x_span_fraction=x_span,
            y_span_fraction=y_span,
        )["passed"]

    assert passes()
    assert not passes(selected=23)
    assert not passes(lift=0.7999)
    assert not passes(correct=17)
    assert not passes(precision=0.6999)
    assert not passes(cells=5)
    assert not passes(x_span=0.3999)
    assert not passes(y_span=0.1499)


def test_cross_render_and_identical_query_calibration_gates() -> None:
    query, render, _intrinsics, _extrinsics = _geometry()
    assert cross_render_calibration(query, render)["passed"]

    mismatched_depth = np.asarray(render.forward_depth_m) * np.exp(0.011)
    mismatched_render = replace(
        render,
        forward_depth_m=mismatched_depth,
        world_xyz_m=unproject_pinhole_depth(mismatched_depth, render.intrinsics, render.extrinsics),
    )
    assert not cross_render_calibration(query, mismatched_render)["passed"]

    points = np.asarray([(x, y) for y in (2.0, 10.0, 18.0) for x in (2.0, 10.0, 18.0, 26.0)])
    frozen = freeze_match_artifact(_matches(points), "identical")
    calibration = identical_query_calibration(frozen.manifest, frozen.files, width_px=32, height_px=32)
    assert calibration["selected_count"] == 12
    assert calibration["within_1px_fraction"] == 1.0
    assert calibration["occupied_4x4_cells"] == 12
    assert calibration["coverage_fraction"] == 0.75
    assert calibration["passed"]

    shifted = points.copy()
    shifted[0, 0] += 2.0
    failed = freeze_match_artifact(_matches(points, shifted), "not-identical")
    assert not identical_query_calibration(failed.manifest, failed.files, width_px=32, height_px=32)["passed"]
