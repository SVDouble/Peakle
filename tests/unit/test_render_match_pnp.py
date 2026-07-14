from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from skimage import data

import peakle.localize.render_match_pnp as render_match_module
from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics, CameraModel
from peakle.domain.coordinates import LocalPoint
from peakle.domain.pose import PosePrior
from peakle.localize.candidate_validation import (
    candidate_zbuffer_visibility as _candidate_zbuffer_visibility,
)
from peakle.localize.candidate_validation import (
    exact_binomial_lower_bound as _exact_binomial_lower_bound,
)
from peakle.localize.correspondence import (
    CORRESPONDENCE_CACHE_ENTRY_SCHEMA,
    WORKER_RENDER_ID_POLICY,
    WORKER_RENDER_ID_POLICY_VERSION,
    MatcherUnavailable,
    MatchSet,
    SiftMatcher,
    WorkerMatcher,
    match_image_fan,
)
from peakle.localize.pnp import (
    GROUND_ELEVATION_SOURCE_SCHEMA,
    RANSAC_SAMPLING_METHOD,
    RANSAC_SAMPLING_SCHEMA,
    WORLD_CONSENSUS_GEOMETRY_SCHEMA,
    PoseRansacConfig,
    PoseRansacResult,
    fit_pose_ransac,
    project_world_points,
    required_finite_population_ransac_trials,
    required_ransac_trials,
)
from peakle.localize.render_match_pnp import (
    CANDIDATE_VALIDATION_SCHEMA,
    MATCH_SELECTION_METHOD,
    MATCH_SELECTION_SCHEMA,
    QUERY_PADDING_MASK_METHOD,
    QUERY_PADDING_MASK_SCHEMA,
    RENDER_SEED_BATCH_SCHEMA,
    RENDER_SEED_SCHEMA,
    CandidateValidationConfig,
    IdentifiedRenderSeed,
    RenderMatchConfig,
    RenderSeed,
    _balanced_match_cap_indices,
    _candidate_render_extrinsics,
    _FrameAttempt,
    _match_and_fit_frame,
    _match_selection_record,
    _query_holdout_fold,
    _query_spatial_holdout_mask,
    _query_warp_padding_mask,
    _validate_candidate_pose,
    render_fan_yaws,
    solve_render_match_pose,
    solve_render_match_pose_batch,
)
from peakle.localize.swissdem import Patch
from peakle.rendering.pinhole import camera_axes
from peakle.rendering.terrain_view import TerrainRenderBundle, TerrainViewRenderer, lift_render_pixels
from peakle.scene.scene import Scene
from peakle.scripts.roma_match_worker import _joint_grid_select


def _placed_pose(scene: Scene) -> CameraExtrinsics:
    east_m = 50.0
    north_m = -3_400.0
    return CameraExtrinsics(
        position=LocalPoint(
            east_m=east_m,
            north_m=north_m,
            up_m=scene.terrain.elevation_at(east_m, north_m) + 3.0,
        ),
        yaw_deg=4.0,
        pitch_deg=7.0,
        roll_deg=0.0,
    )


def _visible_elevation_patch(scene: Scene, *, nodata: bool = False) -> Patch:
    x_m = np.linspace(-800.0, 800.0, 41)
    y_m = np.linspace(-2_500.0, 500.0, 61)
    if nodata:
        elevation_m = np.full((len(y_m), len(x_m)), np.nan)
    else:
        x_grid, y_grid = np.meshgrid(x_m, y_m)
        base = np.asarray(
            [[scene.terrain.elevation_at(east, north) for east in x_m] for north in y_m],
            dtype=np.float64,
        )
        hill = 350.0 * np.exp(-((x_grid / 250.0) ** 2 + ((y_grid + 500.0) / 500.0) ** 2))
        elevation_m = base + hill
    return Patch(x_m=x_m, y_m=y_m, elevation_m=elevation_m)


def _exact_cyltan_ground_case(
    ground_elevation_m: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    CameraModel,
    CameraExtrinsics,
    PosePrior,
    PoseRansacConfig,
]:
    rng = np.random.default_rng(911)
    truth = CameraExtrinsics(
        position=LocalPoint(east_m=0.0, north_m=-800.0, up_m=ground_elevation_m + 8.0),
        yaw_deg=7.0,
        pitch_deg=5.0,
        roll_deg=0.0,
    )
    camera = CameraModel(width_px=640, height_px=420, horizontal_fov_deg=60.0, projection="cyltan")
    distance_m = rng.uniform(1_500.0, 5_000.0, 64)
    azimuth_rad = np.radians(truth.yaw_deg + rng.uniform(-26.0, 26.0, len(distance_m)))
    world = np.column_stack(
        (
            truth.position.east_m + distance_m * np.sin(azimuth_rad),
            truth.position.north_m + distance_m * np.cos(azimuth_rad),
            truth.position.up_m
            + distance_m * (math.tan(math.radians(truth.pitch_deg)) + rng.uniform(-0.18, 0.18, len(distance_m))),
        )
    )
    image, visible = project_world_points(world, camera, truth)
    assert visible.all()
    prior = PosePrior(
        position=truth.position,
        yaw_deg=truth.yaw_deg,
        pitch_deg=truth.pitch_deg,
        horizontal_sigma_m=50.0,
        vertical_sigma_m=25.0,
        yaw_sigma_deg=10.0,
        pitch_sigma_deg=8.0,
    )
    config = PoseRansacConfig(
        iterations=2,
        max_iterations=2,
        sample_size=6,
        horizontal_search_radius_m=100.0,
        yaw_search_radius_deg=15.0,
        clearance_constraint_policy="prior_ground_coupled",
        seed=21,
    )
    return world, image, camera, truth, prior, config


def _write_fake_cache_worker(tmp_path: Path) -> tuple[Path, Path]:
    worker_script = tmp_path / "cache_worker.py"
    request_log = tmp_path / "cache_worker.requests.jsonl"
    worker_script.write_text(
        """
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--request', required=True)
args = parser.parse_args()
request_path = Path(args.request)
request = json.loads(request_path.read_text())
root = request_path.parent
log_path = Path(__file__).with_name('cache_worker.requests.jsonl')
with log_path.open('a') as handle:
    handle.write(json.dumps({
        'ids': [render['id'] for render in request['renders']],
        'offline': {
            'hf_hub': os.environ.get('HF_HUB_OFFLINE'),
            'network': os.environ.get('PEAKLE_MATCHER_NETWORK_ALLOWED'),
        },
    }, sort_keys=True) + '\\n')
records = []
outputs = []
for render in request['renders']:
    seed_material = json.dumps(
        {'seed': request['seed'], 'render_id': render['id']},
        sort_keys=True,
        separators=(',', ':'),
    ).encode()
    marker = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], 'big') % 40 + 1
    output_path = root / render['matches_npz']
    np.savez(
        output_path,
        query_xy_px=np.asarray([[float(marker), 20.0], [70.0, 50.0]], np.float64),
        render_xy_px=np.asarray([[30.0, float(marker)], [45.0, 35.0]], np.float64),
        confidence=np.asarray([0.9, 0.1], np.float64),
        selected=np.asarray([True, False]),
    )
    output_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
    records.append({
        'id': render['id'],
        'runtime_s': 0.25,
        'input_sha256': render['sha256'],
        'input_shape': render['shape'],
        'output_sha256': output_sha,
    })
    outputs.append({'path': output_path.name, 'sha256': output_sha})
implementation_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(root / request['outputs']['result_json']).write_text(json.dumps({
    'schema': 'peakle_match_batch_result_v1',
    'status': 'ok',
    'matcher_id': request['matcher_id'],
    'request_sha256': hashlib.sha256(request_path.read_bytes()).hexdigest(),
    'query': request['query'],
    'renders': records,
    'runtime': {'total_s': 0.5},
    'provenance': {
        'worker': {
            'implementation_sha256': implementation_sha,
            'network_allowed': False,
            'implicit_model_downloads_allowed': False,
        },
    },
    'outputs': outputs,
}))
""".strip()
        + "\n"
    )
    return worker_script, request_log


def test_metric_render_depth_round_trips_to_integer_pixel_centres(scene: Scene) -> None:
    pose = _placed_pose(scene)
    bundle = TerrainViewRenderer().render(
        scene.terrain,
        scene.intrinsics,
        pose,
        modality="normal",
        terrain_stride=1,
    )
    rows_columns = np.argwhere(bundle.terrain_mask)
    rows_columns = rows_columns[:: max(1, len(rows_columns) // 200)][:200]
    world = bundle.world_xyz_m[rows_columns[:, 0], rows_columns[:, 1]]
    camera = CameraModel.from_intrinsics(scene.intrinsics)

    projected, valid = project_world_points(world, camera, pose)

    expected = np.column_stack((rows_columns[:, 1], rows_columns[:, 0]))
    assert valid.all()
    assert np.max(np.linalg.norm(projected - expected, axis=1)) < 1e-8
    assert bundle.provenance["depth_semantics"].startswith("positive camera-forward")
    assert bundle.provenance["uses_source_depth_pfm"] is False
    assert bundle.provenance["native_high_resolution_patch_used"] is False
    assert len(bundle.provenance["render_content_sha256"]) == 64
    assert bundle.forward_depth_m.flags.writeable is False


def test_spatial_holdout_is_deterministic_disjoint_and_interleaved() -> None:
    camera = CameraModel(width_px=800, height_px=600, horizontal_fov_deg=60.0, projection="pinhole")
    config = CandidateValidationConfig()
    query_xy = np.asarray(
        [
            [
                (column + 0.5) * camera.width_px / config.query_grid_columns,
                (row + 0.5) * camera.height_px / config.query_grid_rows,
            ]
            for row in range(config.query_grid_rows)
            for column in range(config.query_grid_columns)
        ],
        dtype=np.float64,
    )

    masks = [
        _query_spatial_holdout_mask(query_xy, camera, config, holdout_fold) for holdout_fold in range(config.folds)
    ]

    np.testing.assert_array_equal(np.sum(masks, axis=0), np.ones(len(query_xy), dtype=np.int64))
    assert [int(mask.sum()) for mask in masks] == [12, 12, 12, 12]
    for mask in masks:
        selected = query_xy[mask]
        assert np.ptp(selected[:, 0]) > camera.width_px * 0.5
        assert np.ptp(selected[:, 1]) > camera.height_px * 0.5


def test_candidate_validation_rejects_non_finite_depth_tolerance() -> None:
    with pytest.raises(ValueError, match="maximum candidate-render depth tolerance"):
        CandidateValidationConfig(maximum_depth_tolerance_m=float("nan")).validate()


def test_candidate_zbuffer_uses_inverse_depth_and_rejects_depth_order_conflicts(scene: Scene) -> None:
    pose = _placed_pose(scene)
    validation_intrinsics = CameraIntrinsics.from_horizontal_fov(
        scene.intrinsics.width_px * 2,
        scene.intrinsics.height_px * 2,
        CameraModel.from_intrinsics(scene.intrinsics).horizontal_fov_deg,
    )
    bundle = TerrainViewRenderer().render(
        scene.terrain,
        validation_intrinsics,
        pose,
        modality="normal",
        terrain_stride=1,
    )
    depth = bundle.forward_depth_m
    chosen: tuple[int, int, np.ndarray] | None = None
    for row in range(1, depth.shape[0] - 1):
        for column in range(1, depth.shape[1] - 1):
            support = depth[row - 1 : row + 2, column - 1 : column + 2]
            if not np.all(np.isfinite(support)) or np.any(support <= 0.0):
                continue
            span = float(np.ptp(support))
            maximum_span = min(250.0, 0.08 * float(np.median(support)))
            if 5.0 < span <= maximum_span:
                chosen = row, column, support
                break
        if chosen is not None:
            break
    assert chosen is not None
    row, column, support = chosen
    u_px = column + 0.25
    v_px = row + 0.25
    inverse_support = 1.0 / depth[row : row + 2, column : column + 2]
    sampled_inverse = (
        0.75 * 0.75 * inverse_support[0, 0]
        + 0.25 * 0.75 * inverse_support[0, 1]
        + 0.75 * 0.25 * inverse_support[1, 0]
        + 0.25 * 0.25 * inverse_support[1, 1]
    )
    visible_depth = 1.0 / float(sampled_inverse)
    right, down, forward = camera_axes(pose)
    x_ratio = (u_px - bundle.intrinsics.principal_x_px) / bundle.intrinsics.focal_length_px
    y_ratio = (v_px - bundle.intrinsics.principal_y_px) / bundle.intrinsics.focal_length_px
    ray_per_forward_m = forward + x_ratio * right + y_ratio * down
    position = np.asarray(pose.position.as_tuple(), dtype=np.float64)
    visible_world = position + visible_depth * ray_per_forward_m
    displacement = 50.0
    world = np.stack(
        (
            visible_world,
            visible_world + displacement * ray_per_forward_m,
            visible_world - displacement * ray_per_forward_m,
        )
    )

    classified = _candidate_zbuffer_visibility(
        bundle,
        world,
        max_absolute_depth_span_m=250.0,
        max_relative_depth_span=0.08,
        minimum_depth_tolerance_m=1.0,
        maximum_depth_tolerance_m=3.0,
        relative_depth_tolerance=1e-4,
    )

    np.testing.assert_array_equal(classified.testable, [True, True, True])
    np.testing.assert_array_equal(classified.consistent, [True, False, False])
    np.testing.assert_array_equal(classified.occluded, [False, True, False])
    np.testing.assert_array_equal(classified.in_front, [False, False, True])
    assert classified.signed_depth_residual_m[0] == pytest.approx(0.0, abs=1e-8)


def test_candidate_gate_keeps_cyltan_query_and_auxiliary_pinhole_depth_separate(scene: Scene) -> None:
    candidate = _placed_pose(scene)
    bundle = TerrainViewRenderer().render(
        scene.terrain,
        scene.intrinsics,
        candidate,
        modality="normal",
        terrain_stride=1,
    )
    source_intrinsics = CameraIntrinsics.from_horizontal_fov(
        scene.intrinsics.width_px // 2,
        scene.intrinsics.height_px // 2,
        CameraModel.from_intrinsics(scene.intrinsics).horizontal_fov_deg,
    )
    source_bundle = TerrainViewRenderer().render(
        scene.terrain,
        source_intrinsics,
        candidate,
        modality="normal",
        terrain_stride=1,
    )
    query_camera = CameraModel(
        width_px=scene.intrinsics.width_px,
        height_px=scene.intrinsics.height_px,
        horizontal_fov_deg=CameraModel.from_intrinsics(scene.intrinsics).horizontal_fov_deg,
        projection="cyltan",
    )
    rows_columns = np.argwhere(bundle.terrain_mask)
    rows_columns = rows_columns[:: max(1, len(rows_columns) // 500)]
    world = bundle.world_xyz_m[rows_columns[:, 0], rows_columns[:, 1]]
    query_xy, query_valid = project_world_points(world, query_camera, candidate)
    query_inside = (
        query_valid
        & (query_xy[:, 0] >= 0.0)
        & (query_xy[:, 0] <= query_camera.width_px - 1.0)
        & (query_xy[:, 1] >= 0.0)
        & (query_xy[:, 1] <= query_camera.height_px - 1.0)
    )
    world = world[query_inside]
    query_xy = query_xy[query_inside]
    assert len(world) >= 100
    empty_matches = MatchSet(
        query_xy_px=np.empty((0, 2)),
        render_xy_px=np.empty((0, 2)),
        confidence=np.empty(0),
    )
    source = _FrameAttempt(
        index=2,
        render=source_bundle,
        match_set=empty_matches,
        lifted_world=np.empty((0, 3)),
        query_xy=np.empty((0, 2)),
        confidence=np.empty(0),
        holdout_world=world,
        holdout_query_xy=query_xy,
        holdout_confidence=np.ones(len(world)),
        pnp_result=None,
        record={},
    )
    settings = RenderMatchConfig(refinement_passes=0)
    auxiliary_pose = _candidate_render_extrinsics(query_camera, candidate)
    assert auxiliary_pose.pitch_deg == candidate.pitch_deg
    assert auxiliary_pose.roll_deg == 0.0

    accepted = _validate_candidate_pose(
        source,
        bundle,
        query_camera,
        candidate,
        settings,
        holdout_fold=1,
    )
    wrong_intrinsics = CameraIntrinsics.from_horizontal_fov(
        bundle.intrinsics.width_px,
        bundle.intrinsics.height_px,
        bundle.intrinsics.horizontal_fov_deg() + 1.0,
    )
    with pytest.raises(RuntimeError, match="changed the auxiliary camera intrinsics"):
        _validate_candidate_pose(
            source,
            replace(bundle, intrinsics=wrong_intrinsics),
            query_camera,
            candidate,
            settings,
            holdout_fold=1,
        )
    wrong_extrinsics = candidate.model_copy(update={"yaw_deg": candidate.yaw_deg + 1.0})
    with pytest.raises(RuntimeError, match="does not use the selected candidate pose"):
        _validate_candidate_pose(
            source,
            replace(bundle, extrinsics=wrong_extrinsics),
            query_camera,
            candidate,
            settings,
            holdout_fold=1,
        )
    corrupted = replace(source, holdout_query_xy=query_xy + np.asarray([0.0, 20.0]))
    rejected = _validate_candidate_pose(
        corrupted,
        bundle,
        query_camera,
        candidate,
        settings,
        holdout_fold=1,
    )

    assert accepted["schema"] == CANDIDATE_VALIDATION_SCHEMA
    assert accepted["passed"] is True, accepted
    assert accepted["uses_reference_truth"] is False
    assert accepted["withheld_from_geometric_pose_fit"] is True
    assert accepted["withheld_from_geometric_frame_ranking"] is True
    assert accepted["matcher_used_full_query_image"] is True
    assert accepted["worker_candidate_selection_precedes_holdout"] is True
    assert accepted["projection_separation"] == {
        "query_reprojection": "cyltan",
        "visibility_render": "auxiliary_pinhole",
        "depth_comparison": (
            "heldout world-point forward depth and z-buffer depth are both computed in the same "
            "auxiliary pinhole camera"
        ),
        "cross_projection_depth_comparison": False,
        "outside_auxiliary_render_coverage": "untestable",
    }
    assert accepted["counts"]["query_reprojection_inliers"] == len(world)
    assert accepted["counts"]["candidate_render_testable"] + accepted["counts"]["outside_auxiliary_frustum"] + accepted[
        "counts"
    ]["missing_or_sky_depth_support"] + accepted["counts"]["discontinuous_depth_support"] == len(world)
    assert accepted["counts"]["joint_support"] > len(world) * 0.5
    assert accepted["render_contract"]["resolution_multiplier"] == 2
    assert accepted["render_contract"]["surface_identity_matches"] is True
    assert accepted["render_contract"]["candidate_intrinsics_match"] is True
    assert accepted["render_contract"]["candidate_extrinsics_match"] is True
    assert accepted["joint_world_geometry_gate"]["passed"] is True
    assert "independent Bernoulli" in accepted["gates"]["binomial_model_note"]
    assert rejected["passed"] is False
    assert "heldout_joint_consensus_below_acceptance_gate" in rejected["failures"]
    assert _exact_binomial_lower_bound(0, 100, 0.95) == 0.0


def test_native_elevation_patch_is_decimated_and_composited_over_regional_terrain(scene: Scene) -> None:
    pose = _placed_pose(scene)
    renderer = TerrainViewRenderer()
    base = renderer.render(scene.terrain, scene.intrinsics, pose, terrain_stride=1)

    rendered = renderer.render(
        scene.terrain,
        scene.intrinsics,
        pose,
        terrain_stride=1,
        native_elevation_patch=_visible_elevation_patch(scene),
        native_patch_stride=4,
    )

    assert rendered.provenance["native_high_resolution_patch_used"] is True
    patch_record = rendered.provenance["native_high_resolution_patch"]
    assert len(patch_record["source_content_sha256"]) == 64
    assert patch_record["source_shape"] == [61, 41]
    assert patch_record["source_spacing_m"] == pytest.approx({"east": 40.0, "north": 50.0})
    assert patch_record["render_stride"] == 4
    assert patch_record["render_mesh_shape"] == [16, 11]
    assert patch_record["render_effective_mesh_spacing_m"] == pytest.approx({"east": 160.0, "north": 200.0})
    assert patch_record["rendered_at_source_spacing"] is False
    assert patch_record["visible_z_buffer_pixels"] > 0
    assert patch_record["depth_composition_policy"].startswith(
        "the native mesh replaces the coarse representation at matching supported world XY"
    )
    assert rendered.provenance["terrain_composition_policy"].startswith(
        "suppress regional pixels only when their visible world XY lies in finite native-triangle support"
    )
    assert "not a claim of source-resolution rendering" in rendered.provenance["terrain_surface_note"]
    assert rendered.provenance["render_surface_identity_sha256"] != base.provenance["render_surface_identity_sha256"]
    assert rendered.provenance["render_content_sha256"] != base.provenance["render_content_sha256"]
    shared = np.isfinite(base.forward_depth_m) & np.isfinite(rendered.forward_depth_m)
    assert np.max(np.abs(base.forward_depth_m[shared] - rendered.forward_depth_m[shared])) > 1.0


def test_native_patch_nodata_never_punches_holes_in_regional_fallback(scene: Scene) -> None:
    pose = _placed_pose(scene)
    renderer = TerrainViewRenderer()
    base = renderer.render(scene.terrain, scene.intrinsics, pose, terrain_stride=1)

    rendered = renderer.render(
        scene.terrain,
        scene.intrinsics,
        pose,
        terrain_stride=1,
        native_elevation_patch=_visible_elevation_patch(scene, nodata=True),
        native_patch_stride=4,
    )

    assert rendered.provenance["native_high_resolution_patch_used"] is False
    patch_record = rendered.provenance["native_high_resolution_patch"]
    assert patch_record["source_finite_elevation_samples"] == 0
    assert patch_record["source_nodata_samples"] == 61 * 41
    assert patch_record["visible_z_buffer_pixels"] == 0
    assert np.array_equal(rendered.terrain_mask, base.terrain_mask)
    assert np.array_equal(rendered.forward_depth_m, base.forward_depth_m, equal_nan=True)
    # The request identity records the supplied source even when every output
    # pixel conservatively falls back to the regular regional DEM.
    assert rendered.provenance["render_surface_identity_sha256"] != base.provenance["render_surface_identity_sha256"]
    assert rendered.provenance["render_content_sha256"] == base.provenance["render_content_sha256"]


def test_render_lifting_rejects_depth_edges_and_preserves_safe_pixels(scene: Scene) -> None:
    pose = _placed_pose(scene)
    bundle = TerrainViewRenderer().render(scene.terrain, scene.intrinsics, pose, terrain_stride=1)
    rows, columns = np.nonzero(bundle.terrain_mask)
    sample = np.column_stack((columns[::50], rows[::50])).astype(float)

    lifted = lift_render_pixels(bundle, sample, max_relative_depth_span=0.08)

    assert lifted.valid.any()
    accepted_xy = sample[lifted.valid].astype(int)
    expected = bundle.world_xyz_m[accepted_xy[:, 1], accepted_xy[:, 0]]
    assert lifted.world_xyz_m[lifted.valid] == pytest.approx(expected)
    assert lifted.rejection_counts["accepted"] == int(lifted.valid.sum())
    assert sum(lifted.rejection_counts.values()) == len(sample)

    fractional = sample + np.asarray([0.27, 0.31])
    fractional_lifted = lift_render_pixels(bundle, fractional, max_relative_depth_span=0.08)
    assert fractional_lifted.valid.any()
    camera = CameraModel.from_intrinsics(scene.intrinsics)
    reprojected, projected_valid = project_world_points(
        fractional_lifted.world_xyz_m[fractional_lifted.valid],
        camera,
        pose,
    )
    assert projected_valid.all()
    assert reprojected == pytest.approx(fractional[fractional_lifted.valid], abs=1e-8)

    accepted_index = int(np.flatnonzero(lifted.valid)[0])
    accepted_x, accepted_y = sample[accepted_index].astype(int)
    appearance_mask = bundle.appearance_mask.copy()
    appearance_mask[accepted_y, accepted_x] = False
    masked_bundle = replace(bundle, appearance_mask=appearance_mask)
    appearance_rejected = lift_render_pixels(masked_bundle, sample[[accepted_index]])
    assert appearance_rejected.valid.tolist() == [False]
    assert appearance_rejected.rejection_counts["invalid_appearance"] == 1


def test_query_warp_padding_mask_floods_black_border_wedge_and_antialias_fringe() -> None:
    image = np.full((48, 80, 3), 140, dtype=np.uint8)
    core_widths: list[int] = []
    for row in range(image.shape[0]):
        core_width = 5 + abs(row - image.shape[0] // 2) // 4
        core_widths.append(core_width)
        image[row, :core_width] = 0
        image[row, core_width] = 18

    padding = _query_warp_padding_mask(image)

    assert padding.record["schema"] == QUERY_PADDING_MASK_SCHEMA
    assert padding.record["method"] == QUERY_PADDING_MASK_METHOD
    assert padding.record["truth_free"] is True
    assert padding.record["active"] is True
    assert padding.record["connected_components"]["retained_as_meaningful"] == 1
    assert padding.record["antialias_fringe_pixels"] > 0
    assert padding.record["masked_pixel_fraction"] == pytest.approx(padding.record["masked_pixels"] / (48 * 80))
    for row, core_width in enumerate(core_widths):
        assert padding.mask[row, : core_width + 1].all()
        assert not padding.mask[row, core_width + 1]


def test_query_warp_padding_mask_preserves_disconnected_interior_black_region() -> None:
    image = np.full((48, 80, 3), 120, dtype=np.uint8)
    image[:, :3] = 0
    image[:, 3] = 18
    image[18:30, 30:45] = 0

    padding = _query_warp_padding_mask(image)

    assert padding.record["active"] is True
    assert padding.record["no_op_reason"] is None
    assert padding.record["connected_components"] == {
        "near_black_total": 2,
        "touching_border": 1,
        "retained_as_meaningful": 1,
    }
    assert padding.mask[:, :4].all()
    assert not padding.mask[18:30, 30:45].any()


def test_query_warp_padding_mask_is_a_noop_for_an_unpadded_photo() -> None:
    rows, columns = np.indices((37, 53))
    base = 35 + (rows * 3 + columns * 5) % 180
    image = np.repeat(base[..., None], 3, axis=2).astype(np.uint8)

    padding = _query_warp_padding_mask(image)

    assert padding.record["active"] is False
    assert padding.record["masked_pixels"] == 0
    assert padding.record["masked_pixel_fraction"] == 0.0
    assert padding.record["connected_components"]["near_black_total"] == 0
    assert not padding.mask.any()


def test_balanced_match_cap_prevents_a_high_confidence_cell_from_consuming_the_budget() -> None:
    cluster_count = 200
    spread_count = 100
    cluster = np.column_stack(
        (
            np.linspace(1.0, 14.0, cluster_count),
            np.linspace(1.0, 14.0, cluster_count),
        )
    )
    columns = np.arange(spread_count) % 10
    rows = np.arange(spread_count) // 10
    spread = np.column_stack((24.0 + columns * 20.0, 24.0 + rows * 20.0))
    query_xy = np.vstack((cluster, spread)).astype(np.float64)
    render_xy = query_xy.copy()
    confidence = np.concatenate(
        (
            np.linspace(1.0, 0.8, cluster_count),
            np.linspace(0.7, 0.6, spread_count),
        )
    )

    selected = _balanced_match_cap_indices(
        query_xy,
        render_xy,
        confidence,
        max_matches=100,
        cell_px=16,
    )

    global_top = np.argsort(-confidence, kind="stable")[:100]
    balanced_cells = np.unique(np.floor(query_xy[selected] / 16).astype(int), axis=0)
    global_cells = np.unique(np.floor(query_xy[global_top] / 16).astype(int), axis=0)
    assert selected.size == 100
    assert len(balanced_cells) >= 90
    assert len(global_cells) == 1
    np.testing.assert_array_equal(
        selected,
        _balanced_match_cap_indices(
            query_xy,
            render_xy,
            confidence,
            max_matches=100,
            cell_px=16,
        ),
    )


def test_cached_worker_candidates_reselect_to_the_direct_progressive_prefix() -> None:
    rng = np.random.default_rng(20260714)
    query_xy = rng.uniform([0.0, 0.0], [775.0, 593.0], size=(7_000, 2))
    render_xy = rng.uniform([0.0, 0.0], [320.0, 256.0], size=(7_000, 2))
    confidence = rng.random(7_000)

    cached_5k, _ = _joint_grid_select(
        query_xy,
        render_xy,
        confidence,
        max_matches=5_000,
        cell_px=16,
    )
    cached_selection = _balanced_match_cap_indices(
        query_xy[cached_5k],
        render_xy[cached_5k],
        confidence[cached_5k],
        max_matches=800,
        cell_px=16,
    )
    direct_800, _ = _joint_grid_select(
        query_xy,
        render_xy,
        confidence,
        max_matches=800,
        cell_px=16,
    )

    np.testing.assert_array_equal(cached_5k[cached_selection], direct_800)


def test_match_selection_record_exposes_versioned_before_and_after_distribution() -> None:
    cluster = np.column_stack((np.linspace(1.0, 14.0, 20), np.linspace(1.0, 14.0, 20)))
    spread = np.asarray([[40.0, 40.0], [80.0, 40.0], [40.0, 80.0], [80.0, 80.0]])
    query_xy = np.vstack((cluster, spread))
    matches = MatchSet(
        query_xy_px=query_xy,
        render_xy_px=query_xy,
        confidence=np.concatenate((np.linspace(1.0, 0.8, 20), np.linspace(0.7, 0.4, 4))),
    )
    selected = _balanced_match_cap_indices(
        matches.query_xy_px,
        matches.render_xy_px,
        matches.confidence,
        max_matches=5,
        cell_px=16,
    )

    record = _match_selection_record(
        matches,
        selected,
        query_camera=CameraModel(width_px=100, height_px=100, horizontal_fov_deg=60.0),
        render_shape=(100, 100, 3),
        max_matches=5,
        cell_px=16,
    )

    assert record["schema"] == MATCH_SELECTION_SCHEMA
    assert record["method"] == MATCH_SELECTION_METHOD
    assert record["cap_stage"] == "after_render_lifting"
    assert record["worker_selected_matches"] == 24
    assert record["lift_valid_matches"] == 24
    assert record["rejected_by_lifting_before_cap"] == 0
    assert record["input_matches"] == 24
    assert record["selected_matches"] == 5
    assert record["cap_applied"] is True
    assert record["selected_distribution"]["query"]["occupied_selection_cells"] == 5
    assert record["input_distribution"]["query"]["occupied_selection_cells"] == 5


@pytest.mark.parametrize("invalid_surface", ["sky", "nodata"])
def test_invalid_render_matches_cannot_consume_the_post_lift_budget(
    scene: Scene,
    invalid_surface: str,
) -> None:
    pose = _placed_pose(scene)
    bundle = TerrainViewRenderer().render(
        scene.terrain,
        scene.intrinsics,
        pose,
        modality="normal",
        terrain_stride=1,
    )
    camera = CameraModel.from_intrinsics(scene.intrinsics)

    invalid_rows_columns = np.argwhere(~bundle.terrain_mask if invalid_surface == "sky" else bundle.terrain_mask)
    safe_invalid = invalid_rows_columns[
        (invalid_rows_columns[:, 0] >= 2)
        & (invalid_rows_columns[:, 0] < scene.intrinsics.height_px - 2)
        & (invalid_rows_columns[:, 1] >= 2)
        & (invalid_rows_columns[:, 1] < scene.intrinsics.width_px - 2)
    ]
    invalid_by_cell: dict[tuple[int, int], np.ndarray] = {}
    for row, column in safe_invalid:
        invalid_by_cell.setdefault((int(column // 16), int(row // 16)), np.asarray([column, row]))
    invalid_render_xy = np.asarray(list(invalid_by_cell.values())[:80], dtype=np.float64)
    assert len(invalid_render_xy) >= 60

    if invalid_surface == "nodata":
        depth = bundle.forward_depth_m.copy()
        world = bundle.world_xyz_m.copy()
        normals = bundle.world_normals.copy()
        terrain_mask = bundle.terrain_mask.copy()
        appearance_mask = bundle.appearance_mask.copy()
        invalid_columns = invalid_render_xy[:, 0].astype(int)
        invalid_rows = invalid_render_xy[:, 1].astype(int)
        depth[invalid_rows, invalid_columns] = np.nan
        world[invalid_rows, invalid_columns] = np.nan
        normals[invalid_rows, invalid_columns] = 0.0
        terrain_mask[invalid_rows, invalid_columns] = False
        appearance_mask[invalid_rows, invalid_columns] = False
        bundle = replace(
            bundle,
            forward_depth_m=depth,
            world_xyz_m=world,
            world_normals=normals,
            terrain_mask=terrain_mask,
            appearance_mask=appearance_mask,
            provenance={**bundle.provenance, "test_nodata_holes": len(invalid_render_xy)},
        )

    terrain_rows_columns = np.argwhere(bundle.terrain_mask)[::10]
    terrain_xy = terrain_rows_columns[:, ::-1].astype(np.float64)
    terrain_lift = lift_render_pixels(bundle, terrain_xy)
    valid_render_xy = terrain_xy[terrain_lift.valid][:20]
    assert len(valid_render_xy) == 20

    render_xy = np.vstack((invalid_render_xy, valid_render_xy))
    # High-confidence invalid sky/nodata points occupy more joint cells than
    # the budget. A pre-lift cap would select only those points and discard
    # every usable terrain correspondence.
    confidence = np.concatenate(
        (
            np.linspace(1.0, 0.8, len(invalid_render_xy)),
            np.linspace(0.2, 0.1, len(valid_render_xy)),
        )
    )
    matches = MatchSet(
        query_xy_px=render_xy.copy(),
        render_xy_px=render_xy,
        confidence=confidence,
    )
    prior = PosePrior(
        position=pose.position,
        yaw_deg=pose.yaw_deg,
        pitch_deg=pose.pitch_deg,
        horizontal_sigma_m=100.0,
        vertical_sigma_m=50.0,
        yaw_sigma_deg=15.0,
        pitch_sigma_deg=10.0,
    )
    settings = RenderMatchConfig(
        max_matches_per_frame=50,
        min_lifted_matches_per_frame=30,
        refinement_passes=0,
        candidate_validation=CandidateValidationConfig(enabled=False),
    )

    attempt = _match_and_fit_frame(
        0,
        bundle,
        matches,
        camera,
        prior,
        scene.terrain,
        True,
        True,
        settings,
        17,
    )

    assert attempt.pnp_result is None
    assert len(attempt.lifted_world) == 20
    assert attempt.record["matcher_matches"] == len(matches.query_xy_px)
    assert attempt.record["lift_valid_matches_before_cap"] == 20
    assert attempt.record["matches_after_cap"] == 20
    assert attempt.record["lifting"]["accepted"] == 20
    selection = attempt.record["match_selection"]
    assert selection["cap_stage"] == "after_render_lifting"
    assert selection["worker_selected_matches"] == len(matches.query_xy_px)
    assert selection["lift_valid_matches"] == 20
    assert selection["rejected_by_lifting_before_cap"] == len(invalid_render_xy)
    assert selection["raw_worker_exceeds_budget"] is True
    assert selection["cap_applied"] is False


def test_query_padding_matches_are_rejected_before_render_lifting_and_spatial_cap(scene: Scene) -> None:
    pose = _placed_pose(scene)
    bundle = TerrainViewRenderer().render(
        scene.terrain,
        scene.intrinsics,
        pose,
        modality="normal",
        terrain_stride=1,
    )
    camera = CameraModel.from_intrinsics(scene.intrinsics)
    query = np.full((camera.height_px, camera.width_px, 3), 130, dtype=np.uint8)
    query[:, :5] = 0
    query[:, 5] = 18
    query_padding = _query_warp_padding_mask(query)
    assert query_padding.record["active"] is True

    terrain_rows_columns = np.argwhere(bundle.terrain_mask)
    render_candidates = terrain_rows_columns[:, ::-1].astype(np.float64)
    candidate_lift = lift_render_pixels(bundle, render_candidates)
    render_xy = render_candidates[candidate_lift.valid][:24]
    assert len(render_xy) == 24
    padding_y = np.linspace(2.0, camera.height_px - 3.0, 12)
    valid_y = np.linspace(2.0, camera.height_px - 3.0, 12)
    query_xy = np.vstack(
        (
            np.column_stack((np.tile([2.0, 5.0], 6), padding_y)),
            np.column_stack((np.full(12, camera.width_px / 2.0), valid_y)),
        )
    )
    matches = MatchSet(
        query_xy_px=query_xy,
        render_xy_px=render_xy,
        confidence=np.linspace(1.0, 0.5, 24),
    )
    prior = PosePrior(
        position=pose.position,
        yaw_deg=pose.yaw_deg,
        pitch_deg=pose.pitch_deg,
        horizontal_sigma_m=100.0,
        vertical_sigma_m=50.0,
        yaw_sigma_deg=15.0,
        pitch_sigma_deg=10.0,
    )
    settings = RenderMatchConfig(
        max_matches_per_frame=50,
        min_lifted_matches_per_frame=30,
        refinement_passes=0,
        candidate_validation=CandidateValidationConfig(enabled=False),
    )

    attempt = _match_and_fit_frame(
        0,
        bundle,
        matches,
        camera,
        prior,
        scene.terrain,
        True,
        True,
        settings,
        17,
        query_padding=query_padding,
    )

    assert matches.count == 24  # The worker/cache result remains untouched.
    assert attempt.pnp_result is None
    assert attempt.record["match_stage_counts"] == {
        "worker_selected": 24,
        "after_query_padding_rejection": 12,
        "after_render_lifting": 12,
        "after_spatial_cap": 12,
        "training_after_spatial_cap": 12,
        "holdout_after_spatial_cap": 0,
        "total_after_independent_caps": 12,
    }
    padding_record = attempt.record["query_padding_filter"]
    assert padding_record["schema"] == QUERY_PADDING_MASK_SCHEMA
    assert padding_record["rejected_matches"] == 12
    assert padding_record["matches_after_filter"] == 12
    assert padding_record["masked_pixel_fraction"] > 0.0
    assert padding_record["changes_matcher_or_cache_inputs"] is False
    assert attempt.record["lifting"]["accepted"] == 12
    selection = attempt.record["match_selection"]
    assert selection["worker_selected_matches"] == 24
    assert selection["query_padding_valid_matches"] == 12
    assert selection["rejected_by_query_padding_before_lifting"] == 12
    assert selection["lift_valid_matches"] == 12
    assert selection["rejected_by_lifting_before_cap"] == 0
    assert selection["selected_matches"] == 12


@pytest.mark.parametrize("projection", ["pinhole", "cyltan"])
def test_projection_aware_ransac_recovers_pose_with_one_third_outliers(projection: str) -> None:
    rng = np.random.default_rng(44)
    truth = CameraExtrinsics(
        position=LocalPoint(east_m=100.0, north_m=-80.0, up_m=1_500.0),
        yaw_deg=179.0,
        pitch_deg=4.0,
        roll_deg=0.0,
    )
    camera = CameraModel(width_px=640, height_px=420, horizontal_fov_deg=60.0, projection=projection)
    azimuth = np.radians(truth.yaw_deg + rng.uniform(-25.0, 25.0, 120))
    distance = rng.uniform(1_000.0, 10_000.0, 120)
    elevation = rng.uniform(-0.05, 0.18, 120)
    world = np.column_stack(
        (
            truth.position.east_m + distance * np.sin(azimuth),
            truth.position.north_m + distance * np.cos(azimuth),
            truth.position.up_m + distance * np.tan(elevation),
        )
    )
    image, visible = project_world_points(world, camera, truth)
    world = world[visible]
    image = image[visible] + rng.normal(0.0, 0.35, (int(visible.sum()), 2))
    outliers = rng.choice(len(image), size=len(image) // 3, replace=False)
    image[outliers, 0] = rng.uniform(0.0, camera.width_px - 1, len(outliers))
    image[outliers, 1] = rng.uniform(0.0, camera.height_px - 1, len(outliers))
    confidence = np.ones(len(image))
    confidence[outliers] = 0.35
    initial = CameraExtrinsics(
        position=LocalPoint(east_m=250.0, north_m=-190.0, up_m=1_570.0),
        yaw_deg=-166.0,
        pitch_deg=10.0,
        roll_deg=0.0,
    )
    config = PoseRansacConfig(
        iterations=60,
        seed=7,
        horizontal_search_radius_m=600.0,
        vertical_search_radius_m=250.0,
        yaw_search_radius_deg=45.0,
        min_query_y_span_fraction=0.03,
    )

    result = fit_pose_ransac(world, image, confidence, camera, initial, config=config)

    assert result.solved
    assert result.extrinsics is not None
    position_error = math.sqrt(
        (result.extrinsics.position.east_m - truth.position.east_m) ** 2
        + (result.extrinsics.position.north_m - truth.position.north_m) ** 2
        + (result.extrinsics.position.up_m - truth.position.up_m) ** 2
    )
    yaw_error = abs((result.extrinsics.yaw_deg - truth.yaw_deg + 180.0) % 360.0 - 180.0)
    assert position_error < 3.0
    assert yaw_error < 0.1
    assert result.diagnostics["inliers"] >= 70
    sampling = result.diagnostics["ransac_sampling"]
    assert sampling["sample_size"] == 3
    assert sampling["executed_total_trials"] >= config.iterations


def test_ransac_budget_recovers_at_the_declared_twenty_percent_inlier_floor() -> None:
    rng = np.random.default_rng(20260714)
    truth = CameraExtrinsics(
        position=LocalPoint(east_m=100.0, north_m=-80.0, up_m=1_500.0),
        yaw_deg=27.0,
        pitch_deg=4.0,
        roll_deg=0.0,
    )
    camera = CameraModel(width_px=640, height_px=420, horizontal_fov_deg=60.0, projection="cyltan")
    point_count = 200
    azimuth = np.radians(truth.yaw_deg + rng.uniform(-25.0, 25.0, point_count))
    distance = rng.uniform(1_000.0, 10_000.0, point_count)
    elevation = rng.uniform(-0.05, 0.18, point_count)
    world = np.column_stack(
        (
            truth.position.east_m + distance * np.sin(azimuth),
            truth.position.north_m + distance * np.cos(azimuth),
            truth.position.up_m + distance * np.tan(elevation),
        )
    )
    image, visible = project_world_points(world, camera, truth)
    assert visible.all()

    inlier_mask = np.zeros(point_count, dtype=np.bool_)
    inlier_mask[rng.choice(point_count, size=point_count // 5, replace=False)] = True
    image[~inlier_mask, 0] = rng.uniform(0.0, camera.width_px - 1, int((~inlier_mask).sum()))
    image[~inlier_mask, 1] = rng.uniform(0.0, camera.height_px - 1, int((~inlier_mask).sum()))
    permutation = rng.permutation(point_count)
    world = world[permutation]
    image = image[permutation]
    initial = CameraExtrinsics(
        position=LocalPoint(east_m=250.0, north_m=-190.0, up_m=1_570.0),
        yaw_deg=42.0,
        pitch_deg=10.0,
        roll_deg=0.0,
    )
    config = PoseRansacConfig(
        iterations=48,
        seed=71,
        horizontal_search_radius_m=600.0,
        vertical_search_radius_m=250.0,
        yaw_search_radius_deg=45.0,
        min_query_y_span_fraction=0.03,
    )

    result = fit_pose_ransac(
        world,
        image,
        np.ones(point_count),
        camera,
        initial,
        config=config,
    )

    assert result.solved
    assert result.extrinsics is not None
    assert result.diagnostics["inliers"] == point_count // 5
    assert result.diagnostics["inlier_ratio"] == pytest.approx(0.20)
    assert math.dist(result.extrinsics.position.as_tuple(), truth.position.as_tuple()) < 0.01
    assert abs((result.extrinsics.yaw_deg - truth.yaw_deg + 180.0) % 360.0 - 180.0) < 0.01
    sampling = result.diagnostics["ransac_sampling"]
    assert sampling["schema"] == RANSAC_SAMPLING_SCHEMA
    assert sampling["method"] == RANSAC_SAMPLING_METHOD
    assert sampling["assumed_inlier_count_floor"] == 40
    assert sampling["required_uniform_trials_at_assumed_floor"] == 610
    assert sampling["asymptotic_required_uniform_trials_at_assumed_floor"] == 574
    assert sampling["executed_uniform_trials"] == 610
    assert sampling["executed_guided_trials"] > 0
    assert sampling["budget_meets_assumed_floor"] is True
    assert sampling["stopping_reason"] == "adaptive_uniform_target_met"


def test_ransac_trial_formula_matches_the_declared_floor_budget() -> None:
    assert required_ransac_trials(0.20, 3, 0.99) == 574
    assert required_finite_population_ransac_trials(40, 200, 3, 0.99) == 610
    assert required_ransac_trials(1.0, 3, 0.99) == 1


def test_world_consensus_gate_rejects_collinear_false_consensus() -> None:
    truth = CameraExtrinsics(
        position=LocalPoint(east_m=100.0, north_m=-80.0, up_m=1_500.0),
        yaw_deg=12.0,
        pitch_deg=3.0,
        roll_deg=0.0,
    )
    camera = CameraModel(width_px=640, height_px=420, horizontal_fov_deg=60.0, projection="pinhole")
    right, down, forward = camera_axes(truth)
    position = np.asarray(truth.position.as_tuple())
    offset = np.linspace(-1.0, 1.0, 24)
    # This 3D line projects across several query grid cells and produces a
    # perfect reprojection consensus at the seed pose, but cannot independently
    # constrain a full absolute pose.
    world = position[None, :] + 3_000.0 * forward[None, :] + offset[:, None] * (900.0 * right + 500.0 * down)[None, :]
    image, visible = project_world_points(world, camera, truth)
    assert visible.all()
    config = PoseRansacConfig(
        iterations=2,
        max_iterations=2,
        sample_size=6,
        min_correspondences=12,
        min_inliers=10,
        prior_weight_px=0.0,
        ground_weight_px=0.0,
        seed=1,
    )

    result = fit_pose_ransac(world, image, np.ones(len(world)), camera, truth, config=config)

    assert result.solved is False
    assert result.inlier_mask.all()
    assert result.diagnostics["abstain_reason"] == "degenerate_world_consensus"
    assert result.diagnostics["world_degeneracy_failures"] == ["essentially_collinear_world_points"]
    geometry = result.diagnostics["inlier_world_geometry"]
    assert geometry["schema"] == WORLD_CONSENSUS_GEOMETRY_SCHEMA
    assert geometry["unique_point_count"] == 24
    assert geometry["camera_angular_span_deg"] > 30.0
    assert geometry["second_to_first_singular_ratio"] < config.min_world_second_singular_ratio
    assert result.diagnostics["world_degeneracy_gate"]["uses_reference_truth"] is False


def test_world_consensus_gate_rejects_duplicate_inlier_inflation() -> None:
    truth = CameraExtrinsics(
        position=LocalPoint(east_m=100.0, north_m=-80.0, up_m=1_500.0),
        yaw_deg=12.0,
        pitch_deg=3.0,
        roll_deg=0.0,
    )
    camera = CameraModel(width_px=640, height_px=420, horizontal_fov_deg=60.0, projection="pinhole")
    right, down, forward = camera_axes(truth)
    position = np.asarray(truth.position.as_tuple())
    horizontal_offsets = np.asarray([-900.0, -300.0, 300.0, 900.0, -700.0, 700.0])
    vertical_offsets = np.asarray([-450.0, 450.0, -400.0, 400.0, 100.0, -100.0])
    unique_world = (
        position[None, :]
        + 3_000.0 * forward[None, :]
        + horizontal_offsets[:, None] * right[None, :]
        + vertical_offsets[:, None] * down[None, :]
    )
    world = np.repeat(unique_world, 3, axis=0)
    image, visible = project_world_points(world, camera, truth)
    assert visible.all()
    config = PoseRansacConfig(
        iterations=2,
        max_iterations=2,
        sample_size=6,
        min_correspondences=12,
        min_inliers=10,
        prior_weight_px=0.0,
        ground_weight_px=0.0,
        seed=1,
    )

    result = fit_pose_ransac(world, image, np.ones(len(world)), camera, truth, config=config)

    assert result.solved is False
    assert result.inlier_mask.all()
    assert result.diagnostics["abstain_reason"] == "degenerate_world_consensus"
    assert result.diagnostics["world_degeneracy_failures"] == ["excessive_duplicate_world_points"]
    geometry = result.diagnostics["inlier_world_geometry"]
    assert geometry["point_count"] == 18
    assert geometry["unique_point_count"] == 6
    assert geometry["unique_point_fraction"] == pytest.approx(1.0 / 3.0)
    gate = result.diagnostics["world_degeneracy_gate"]
    assert gate["thresholds"]["minimum_unique_point_fraction"] == 0.5


def test_world_consensus_gate_allows_an_ordinary_planar_mountain_surface() -> None:
    truth = CameraExtrinsics(
        position=LocalPoint(east_m=0.0, north_m=-1_000.0, up_m=1_500.0),
        yaw_deg=0.0,
        pitch_deg=5.0,
        roll_deg=0.0,
    )
    camera = CameraModel(width_px=640, height_px=420, horizontal_fov_deg=60.0, projection="pinhole")
    east_grid, north_grid = np.meshgrid(
        np.linspace(-800.0, 800.0, 6),
        np.linspace(1_800.0, 5_000.0, 5),
    )
    # A sloped but exactly planar terrain surface: the third singular value is
    # zero by construction and must not be treated as a degeneracy.
    up_grid = 1_000.0 + 0.20 * east_grid + 0.25 * north_grid
    world = np.column_stack((east_grid.ravel(), north_grid.ravel(), up_grid.ravel()))
    image, visible = project_world_points(world, camera, truth)
    assert visible.all()
    config = PoseRansacConfig(
        iterations=2,
        max_iterations=2,
        sample_size=6,
        min_correspondences=12,
        min_inliers=10,
        prior_weight_px=0.0,
        ground_weight_px=0.0,
        seed=1,
    )

    result = fit_pose_ransac(world, image, np.ones(len(world)), camera, truth, config=config)

    assert result.solved
    geometry = result.diagnostics["inlier_world_geometry"]
    assert geometry["unique_point_fraction"] == 1.0
    assert geometry["second_to_first_singular_ratio"] > 0.4
    assert geometry["third_to_first_singular_ratio"] < 1e-9
    gate = result.diagnostics["world_degeneracy_gate"]
    assert gate["passed"] is True
    assert gate["accepts_planar_surfaces"] is True
    assert gate["thresholds"]["third_singular_value_is_gated"] is False


def test_prior_ground_coupling_breaks_cyltan_height_crop_shift_degeneracy(scene: Scene) -> None:
    rng = np.random.default_rng(3)
    terrain = scene.terrain
    true_east_m = 100.0
    true_north_m = -800.0
    truth = CameraExtrinsics(
        position=LocalPoint(
            east_m=true_east_m,
            north_m=true_north_m,
            up_m=terrain.elevation_at(true_east_m, true_north_m) + 3.0,
        ),
        yaw_deg=10.0,
        pitch_deg=25.0,
        roll_deg=0.0,
    )
    camera = CameraModel(width_px=640, height_px=420, horizontal_fov_deg=60.0, projection="cyltan")

    # Every point is the same horizontal distance from the camera. In cyltan,
    # camera-up and the exact global tan(pitch) crop shift are then vertically
    # degenerate, while azimuth still strongly determines horizontal pose/yaw.
    point_count = 120
    horizontal_range_m = 3_000.0
    azimuth = np.radians(truth.yaw_deg + rng.uniform(-27.0, 27.0, point_count))
    vertical_terms = rng.uniform(-0.18, 0.18, point_count)
    world = np.column_stack(
        (
            true_east_m + horizontal_range_m * np.sin(azimuth),
            true_north_m + horizontal_range_m * np.cos(azimuth),
            truth.position.up_m + horizontal_range_m * (math.tan(math.radians(truth.pitch_deg)) + vertical_terms),
        )
    )
    image, visible = project_world_points(world, camera, truth)
    world = world[visible]
    image = image[visible]

    initial_east_m = true_east_m + 120.0
    initial_north_m = true_north_m - 80.0
    initial = CameraExtrinsics(
        position=LocalPoint(
            east_m=initial_east_m,
            north_m=initial_north_m,
            up_m=terrain.elevation_at(initial_east_m, initial_north_m) + 220.0,
        ),
        yaw_deg=25.0,
        pitch_deg=25.0,
        roll_deg=0.0,
    )
    prior = PosePrior(
        position=initial.position,
        yaw_deg=initial.yaw_deg,
        pitch_deg=initial.pitch_deg,
        horizontal_sigma_m=150.0,
        vertical_sigma_m=100.0,
        yaw_sigma_deg=20.0,
        pitch_sigma_deg=20.0,
    )
    config = PoseRansacConfig(
        iterations=12,
        sample_size=6,
        min_query_y_span_fraction=0.03,
        horizontal_search_radius_m=500.0,
        vertical_search_radius_m=400.0,
        yaw_search_radius_deg=40.0,
        maximum_clearance_m=None,
        ground_weight_px=0.0,
        seed=9,
    )

    unconstrained = fit_pose_ransac(
        world,
        image,
        np.ones(len(world)),
        camera,
        initial,
        prior=prior,
        terrain=terrain,
        use_position_prior=True,
        use_orientation_prior=True,
        config=replace(config, clearance_constraint_policy="free_up"),
    )
    coupled = fit_pose_ransac(
        world,
        image,
        np.ones(len(world)),
        camera,
        initial,
        prior=prior,
        terrain=terrain,
        use_position_prior=True,
        use_orientation_prior=True,
        config=config,
    )

    assert unconstrained.solved
    assert unconstrained.diagnostics["final_clearance_m"] > 150.0
    assert coupled.solved
    assert coupled.extrinsics is not None
    horizontal_error_m = math.hypot(
        coupled.extrinsics.position.east_m - true_east_m,
        coupled.extrinsics.position.north_m - true_north_m,
    )
    assert horizontal_error_m < 0.1
    assert abs((coupled.extrinsics.yaw_deg - truth.yaw_deg + 180.0) % 360.0 - 180.0) < 0.01
    assert 0.5 <= coupled.diagnostics["final_clearance_m"] <= 12.0
    constraint = coupled.diagnostics["clearance_constraint"]
    assert constraint["resolved_policy"] == "prior_ground_coupled"
    assert constraint["raw_prior_clearance_m"] == pytest.approx(220.0)
    assert constraint["prior_clearance_plausible_bounds_m"] == pytest.approx([0.5, 12.0])
    assert constraint["anchor_policy"] == "neutral_ground_camera_fallback"
    assert constraint["fallback_applied"] is True
    assert constraint["fallback_reason"] == "raw_prior_clearance_above_plausible_input_range"
    assert constraint["configured_fallback_clearance_m"] == pytest.approx(2.0)
    assert constraint["configured_fallback_bounds_m"] == pytest.approx([0.5, 12.0])
    assert constraint["anchor_clearance_m"] == pytest.approx(2.0)
    assert constraint["clearance_bounds_m"] == pytest.approx([0.5, 12.0])
    assert "crop shift" in coupled.diagnostics["pitch_semantics"]
    assert coupled.diagnostics["candidate_delta_from_initial"]["horizontal_m"] > 100.0


def test_prior_ground_coupling_uses_neutral_fallback_for_below_ground_prior(scene: Scene) -> None:
    terrain = scene.terrain
    ground_m = terrain.elevation_at(0.0, -800.0)
    world, image, camera, truth, _plausible_prior, config = _exact_cyltan_ground_case(ground_m)
    below_ground_position = LocalPoint(east_m=0.0, north_m=-800.0, up_m=ground_m - 25.0)
    initial = truth.model_copy(update={"position": below_ground_position})
    prior = PosePrior(
        position=below_ground_position,
        yaw_deg=truth.yaw_deg,
        pitch_deg=truth.pitch_deg,
        horizontal_sigma_m=50.0,
        vertical_sigma_m=25.0,
        yaw_sigma_deg=10.0,
        pitch_sigma_deg=8.0,
    )

    result = fit_pose_ransac(
        world,
        image,
        np.ones(len(world)),
        camera,
        initial,
        prior=prior,
        terrain=terrain,
        use_position_prior=True,
        use_orientation_prior=True,
        config=config,
    )

    assert result.solved
    constraint = result.diagnostics["clearance_constraint"]
    assert constraint["raw_prior_clearance_m"] == pytest.approx(-25.0)
    assert constraint["anchor_policy"] == "neutral_ground_camera_fallback"
    assert constraint["fallback_applied"] is True
    assert constraint["fallback_reason"] == "raw_prior_clearance_below_plausible_input_range"
    assert constraint["anchor_clearance_m"] == pytest.approx(2.0)
    assert constraint["clearance_bounds_m"] == pytest.approx([0.5, 12.0])
    assert 0.5 <= result.diagnostics["final_clearance_m"] <= 12.0


def test_pnp_clearance_uses_finer_native_ground_instead_of_regional_dem(scene: Scene) -> None:
    terrain = scene.terrain
    x_m = np.linspace(-400.0, 400.0, 9)
    y_m = np.linspace(-1_200.0, -400.0, 9)
    native_elevation = np.asarray(
        [[terrain.elevation_at(east_m, north_m) + 120.0 for east_m in x_m] for north_m in y_m],
        dtype=np.float64,
    )
    native_patch = Patch(x_m=x_m, y_m=y_m, elevation_m=native_elevation)
    native_ground_m = float(native_elevation[4, 4])
    regional_ground_m = terrain.elevation_at(0.0, -800.0)
    assert native_ground_m - regional_ground_m == pytest.approx(120.0)
    world, image, camera, initial, prior, config = _exact_cyltan_ground_case(native_ground_m)

    result = fit_pose_ransac(
        world,
        image,
        np.ones(len(world)),
        camera,
        initial,
        prior=prior,
        terrain=terrain,
        native_elevation_patch=native_patch,
        use_position_prior=True,
        use_orientation_prior=True,
        config=config,
    )

    assert result.solved
    ground_record = result.diagnostics["ground_elevation_source"]
    assert ground_record["schema"] == GROUND_ELEVATION_SOURCE_SCHEMA
    assert ground_record["native_patch_sampled_at_source_resolution"] is True
    assert ground_record["native_patch"]["source_shape"] == [9, 9]
    assert len(ground_record["native_patch"]["source_content_sha256"]) == 64
    constraint = result.diagnostics["clearance_constraint"]
    assert constraint["initial_ground_sample"]["source"] == "native_patch_bilinear"
    assert constraint["initial_ground_sample"]["elevation_m"] == pytest.approx(native_ground_m)
    assert constraint["prior_ground_sample"]["source"] == "native_patch_bilinear"
    assert constraint["prior_ground_sample"]["elevation_m"] == pytest.approx(native_ground_m)
    assert constraint["raw_prior_clearance_m"] == pytest.approx(8.0)
    assert constraint["anchor_policy"] == "supplied_prior_clearance"
    assert constraint["fallback_applied"] is False
    assert constraint["fallback_reason"] is None
    assert constraint["anchor_clearance_m"] == pytest.approx(8.0)
    assert result.diagnostics["final_ground_sample"]["source"] == "native_patch_bilinear"
    assert result.diagnostics["final_clearance_m"] == pytest.approx(8.0, abs=1e-3)


def test_pnp_native_nodata_cells_fall_back_to_regional_ground(scene: Scene) -> None:
    terrain = scene.terrain
    x_m = np.linspace(-400.0, 400.0, 9)
    y_m = np.linspace(-1_200.0, -400.0, 9)
    native_patch = Patch(
        x_m=x_m,
        y_m=y_m,
        elevation_m=np.full((len(y_m), len(x_m)), np.nan, dtype=np.float64),
    )
    regional_ground_m = terrain.elevation_at(0.0, -800.0)
    world, image, camera, initial, prior, config = _exact_cyltan_ground_case(regional_ground_m)

    result = fit_pose_ransac(
        world,
        image,
        np.ones(len(world)),
        camera,
        initial,
        prior=prior,
        terrain=terrain,
        native_elevation_patch=native_patch,
        use_position_prior=True,
        use_orientation_prior=True,
        config=config,
    )

    assert result.solved
    ground_record = result.diagnostics["ground_elevation_source"]
    assert ground_record["native_patch"]["source_finite_elevation_samples"] == 0
    assert ground_record["native_patch"]["finite_bilinear_support_cells"] == 0
    constraint = result.diagnostics["clearance_constraint"]
    assert constraint["initial_ground_sample"]["source"] == "regional_terrain_fallback"
    assert constraint["initial_ground_sample"]["elevation_m"] == pytest.approx(regional_ground_m)
    assert constraint["prior_ground_sample"]["source"] == "regional_terrain_fallback"
    assert constraint["prior_ground_sample"]["elevation_m"] == pytest.approx(regional_ground_m)
    assert constraint["raw_prior_clearance_m"] == pytest.approx(8.0)
    assert constraint["anchor_policy"] == "supplied_prior_clearance"
    assert constraint["fallback_applied"] is False
    assert constraint["fallback_reason"] is None
    assert constraint["anchor_clearance_m"] == pytest.approx(8.0)
    assert result.diagnostics["final_ground_sample"]["source"] == "regional_terrain_fallback"
    assert result.diagnostics["final_clearance_m"] == pytest.approx(8.0, abs=1e-3)


def test_projection_aware_ransac_abstains_from_implausible_camera_clearance(scene: Scene) -> None:
    rng = np.random.default_rng(81)
    ground = scene.terrain.elevation_at(0.0, -3_000.0)
    truth = CameraExtrinsics(
        position=LocalPoint(east_m=0.0, north_m=-3_000.0, up_m=ground + 220.0),
        yaw_deg=8.0,
        pitch_deg=3.0,
        roll_deg=0.0,
    )
    camera = CameraModel(width_px=640, height_px=420, horizontal_fov_deg=60.0, projection="pinhole")
    right, down, forward = camera_axes(truth)
    depth = rng.uniform(1_500.0, 5_000.0, 80)
    world = (
        np.asarray(truth.position.as_tuple())[None, :]
        + rng.uniform(-0.28, 0.28, (80, 1)) * depth[:, None] * right[None, :]
        + rng.uniform(-0.16, 0.16, (80, 1)) * depth[:, None] * down[None, :]
        + depth[:, None] * forward[None, :]
    )
    image, visible = project_world_points(world, camera, truth)
    config = PoseRansacConfig(
        iterations=8,
        sample_size=8,
        min_correspondences=12,
        min_inliers=10,
        maximum_clearance_m=50.0,
        ground_weight_px=0.0,
        seed=4,
    )

    result = fit_pose_ransac(
        world[visible],
        image[visible],
        np.ones(int(visible.sum())),
        camera,
        truth,
        terrain=scene.terrain,
        config=config,
    )

    assert result.solved is False
    assert result.diagnostics["abstain_reason"] == "implausible_camera_clearance"
    assert result.diagnostics["estimate_clearance_m"] > 200.0
    assert result.diagnostics["candidate_pose"]["position"]["up_m"] > ground + 200.0
    assert result.diagnostics["candidate_delta_from_initial"]["horizontal_m"] < 0.01


def test_sift_control_returns_original_resolution_coordinates() -> None:
    gray = data.camera()
    image = np.repeat(gray[..., None], 3, axis=2)

    matches = SiftMatcher(max_dimension_px=160, max_matches=100).match(image, image)

    assert matches.count >= 30
    assert np.median(np.linalg.norm(matches.query_xy_px - matches.render_xy_px, axis=1)) < 1e-6
    assert matches.provenance["cross_modal"] is False


def test_external_matcher_fan_uses_one_batch_worker_process(tmp_path: Path) -> None:
    worker_script = tmp_path / "worker.py"
    worker_script.write_text(
        """
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--request', required=True)
args = parser.parse_args()
request_path = Path(args.request)
request = json.loads(request_path.read_text())
root = request_path.parent
records = []
for index, render in enumerate(request['renders']):
    np.savez(
        root / render['matches_npz'],
        query_xy_px=np.asarray([[10.0 + index, 20.0], [70.0, 50.0]], np.float32),
        render_xy_px=np.asarray([[30.0, 40.0 + index], [45.0, 35.0]], np.float32),
        confidence=np.asarray([0.9, 0.1], np.float32),
        selected=np.asarray([True, False]),
    )
    records.append({'id': render['id'], 'runtime_s': 0.01})
(root / request['outputs']['result_json']).write_text(json.dumps({
    'schema': 'peakle_match_batch_result_v1',
    'status': 'ok',
    'matcher_id': request['matcher_id'],
    'request_sha256': hashlib.sha256(request_path.read_bytes()).hexdigest(),
    'renders': records,
    'provenance': {
        'worker': {
            'network_allowed': False,
            'implicit_model_downloads_allowed': False,
        },
    },
    'offline_environment': {
        'hf_hub': os.environ.get('HF_HUB_OFFLINE'),
        'transformers': os.environ.get('TRANSFORMERS_OFFLINE'),
        'peakle_network': os.environ.get('PEAKLE_MATCHER_NETWORK_ALLOWED'),
    },
}))
""".strip()
        + "\n"
    )
    matcher = WorkerMatcher(
        command=(sys.executable, str(worker_script)),
        matcher_id="fake-batch",
        model_manifest={"artifacts": []},
    )
    query = np.zeros((60, 80, 3), dtype=np.uint8)
    renders = [np.full((40, 50, 3), index, dtype=np.uint8) for index in range(3)]

    results = match_image_fan(matcher, query, renders)

    assert len(results) == 3
    assert matcher.identity()["correspondence_cache"]["enabled"] is False
    assert all(result.count == 2 for result in results)
    assert [result.query_xy_px[0, 0] for result in results] == [10.0, 11.0, 12.0]
    assert [result.diagnostics["id"] for result in results] == [0, 1, 2]
    assert all("batch" not in result.diagnostics for result in results)
    worker_batch = results[0].provenance["worker_batch"]
    assert worker_batch["status"] == "ok"
    assert worker_batch["offline_environment"] == {
        "hf_hub": "1",
        "transformers": "1",
        "peakle_network": "0",
    }
    assert worker_batch["render_count"] == 3
    assert "renders" not in worker_batch
    assert "outputs" not in worker_batch
    selected = results[0].chosen()
    assert selected.count == 1
    assert selected.diagnostics["worker_total"] == 2
    assert selected.diagnostics["worker_selected"] == 1


def test_correspondence_cache_batches_only_misses_and_restores_render_order(tmp_path: Path) -> None:
    worker_script, request_log = _write_fake_cache_worker(tmp_path)
    cache_dir = tmp_path / "correspondences"
    matcher = WorkerMatcher(
        command=(sys.executable, str(worker_script)),
        matcher_id="fake-cache",
        model_manifest={
            "schema": "fake_manifest_v1",
            "matcher_id": "fake-cache",
            "artifacts": [],
            "inference": {"selection_cell_px": 16, "min_confidence": 0.1},
        },
        seed=41,
        cache_dir=cache_dir,
    )
    query = np.zeros((60, 80, 3), dtype=np.uint8)
    render_a = np.full((40, 50, 3), 10, dtype=np.uint8)
    render_b = np.full((40, 50, 3), 20, dtype=np.uint8)
    render_c = np.full((40, 50, 3), 30, dtype=np.uint8)

    cold = matcher.match_many(query, [render_a, render_b])
    warm_reordered = matcher.match_many(query, [render_b, render_a])
    partial = matcher.match_many(query, [render_a, render_c, render_b])

    requests = [json.loads(line) for line in request_log.read_text().splitlines()]
    assert [len(request["ids"]) for request in requests] == [2, 1]
    assert all(len(render_id) == 64 for request in requests for render_id in request["ids"])
    assert all(request["offline"] == {"hf_hub": "1", "network": "0"} for request in requests)
    np.testing.assert_array_equal(warm_reordered[0].query_xy_px, cold[1].query_xy_px)
    np.testing.assert_array_equal(warm_reordered[1].query_xy_px, cold[0].query_xy_px)
    np.testing.assert_array_equal(warm_reordered[0].selected, cold[1].selected)
    assert [match.diagnostics["id"] for match in warm_reordered] == [0, 1]
    assert [match.diagnostics["cache"]["status"] for match in cold] == ["miss", "miss"]
    assert [match.diagnostics["cache"]["status"] for match in warm_reordered] == ["hit", "hit"]
    assert [match.diagnostics["cache"]["status"] for match in partial] == ["hit", "miss", "hit"]
    assert partial[0].provenance["worker_batch"] is partial[1].provenance["worker_batch"]
    cache_batch = partial[0].provenance["worker_batch"]
    assert cache_batch["cache"]["hits"] == 2
    assert cache_batch["cache"]["misses"] == 1
    assert cache_batch["cache"]["unique_worker_misses"] == 1
    assert cache_batch["runtime"]["worker_invoked_this_call"] is True
    assert partial[0].diagnostics["cache"]["current_worker_runtime_s"] == 0.0
    assert partial[1].diagnostics["cache"]["current_worker_runtime_s"] == 0.25

    metadata_files = sorted(cache_dir.rglob("*.json"))
    assert len(metadata_files) == 3
    metadata = json.loads(metadata_files[0].read_text())
    assert metadata["schema"] == CORRESPONDENCE_CACHE_ENTRY_SCHEMA
    key_record = metadata["key_record"]
    assert key_record["query"]["raw_rgb_sha256"]
    assert key_record["render"]["raw_rgb_sha256"]
    assert key_record["worker"]["request_seed"] == 41
    assert key_record["worker"]["normalized_manifest_sha256"]
    assert key_record["worker"]["inference_config"]["selection_cell_px"] == 16
    assert len(key_record["worker"]["worker_command"]["files"]) >= 2
    assert key_record["contract"]["coordinates"].startswith("original-resolution")
    assert not list(cache_dir.rglob("*.tmp"))


def test_worker_render_ids_and_arrays_do_not_depend_on_cache_toggle(tmp_path: Path) -> None:
    worker_script, request_log = _write_fake_cache_worker(tmp_path)
    manifest = {
        "schema": "fake_manifest_v1",
        "matcher_id": "fake-cache",
        "artifacts": [],
        "inference": {"selection_cell_px": 16, "min_confidence": 0.1},
    }
    query = np.zeros((60, 80, 3), dtype=np.uint8)
    render_a = np.full((40, 50, 3), 10, dtype=np.uint8)
    render_b = np.full((40, 50, 3), 20, dtype=np.uint8)
    ordered_renders = [render_a, render_a.copy(), render_b]
    uncached_matcher = WorkerMatcher(
        command=(sys.executable, str(worker_script)),
        matcher_id="fake-cache",
        model_manifest=manifest,
        seed=41,
    )

    uncached = uncached_matcher.match_many(query, ordered_renders)

    assert not any(path.name == CORRESPONDENCE_CACHE_ENTRY_SCHEMA for path in tmp_path.rglob("*"))
    cache_dir = tmp_path / "correspondences"
    cached_matcher = WorkerMatcher(
        command=(sys.executable, str(worker_script)),
        matcher_id="fake-cache",
        model_manifest=manifest,
        seed=41,
        cache_dir=cache_dir,
    )
    cached_cold = cached_matcher.match_many(query, ordered_renders)

    requests = [json.loads(line) for line in request_log.read_text().splitlines()]
    assert len(requests) == 2
    assert requests[0]["ids"] == requests[1]["ids"]
    assert len(requests[0]["ids"]) == 2
    assert len(set(requests[0]["ids"])) == 2
    assert all(len(render_id) == 64 for render_id in requests[0]["ids"])
    assert [match.diagnostics["id"] for match in uncached] == [0, 1, 2]
    assert [match.diagnostics["id"] for match in cached_cold] == [0, 1, 2]
    assert uncached[0].diagnostics["source_worker_render_id"] == uncached[1].diagnostics["source_worker_render_id"]
    assert (
        cached_cold[0].diagnostics["source_worker_render_id"] == cached_cold[1].diagnostics["source_worker_render_id"]
    )
    for direct, cached in zip(uncached, cached_cold, strict=True):
        np.testing.assert_array_equal(direct.query_xy_px, cached.query_xy_px)
        np.testing.assert_array_equal(direct.render_xy_px, cached.render_xy_px)
        np.testing.assert_array_equal(direct.confidence, cached.confidence)
        np.testing.assert_array_equal(direct.selected, cached.selected)
    direct_batch = uncached[0].provenance["worker_batch"]
    assert all(match.provenance["worker_batch"] is direct_batch for match in uncached)
    assert direct_batch["content_addressed_pairs"] == {
        "render_id_policy": WORKER_RENDER_ID_POLICY,
        "render_id_policy_version": WORKER_RENDER_ID_POLICY_VERSION,
        "requested_occurrences": 3,
        "unique_pairs_executed": 2,
        "duplicate_occurrences": 1,
        "ordered_render_ids": [requests[0]["ids"][0], requests[0]["ids"][0], requests[0]["ids"][1]],
        "cache_enabled": False,
    }
    assert cached_cold[0].provenance["worker_batch"]["cache"]["unique_worker_misses"] == 2
    assert len(list(cache_dir.rglob("*.json"))) == 2


def test_corrupt_correspondence_cache_entry_is_recomputed_and_recorded(tmp_path: Path) -> None:
    worker_script, request_log = _write_fake_cache_worker(tmp_path)
    cache_dir = tmp_path / "correspondences"
    matcher = WorkerMatcher(
        command=(sys.executable, str(worker_script)),
        matcher_id="fake-cache",
        model_manifest={"artifacts": [], "inference": {"max_matches": 100}},
        seed=7,
        cache_dir=cache_dir,
    )
    query = np.zeros((60, 80, 3), dtype=np.uint8)
    render = np.full((40, 50, 3), 50, dtype=np.uint8)
    first = matcher.match(query, render)
    key = first.diagnostics["cache"]["key"]
    metadata_path = next(cache_dir.rglob(f"{key}.json"))
    metadata = json.loads(metadata_path.read_text())
    (metadata_path.parent / metadata["npz_file"]).write_bytes(b"truncated")

    recomputed = matcher.match(query, render)
    warm = matcher.match(query, render)

    assert len(request_log.read_text().splitlines()) == 2
    assert recomputed.diagnostics["cache"]["status"] == "miss"
    assert recomputed.diagnostics["cache"]["reason"].startswith("corrupt_recomputed:ValueError")
    assert recomputed.provenance["worker_batch"]["cache"]["corrupt_entries_recomputed"] == 1
    assert warm.diagnostics["cache"]["status"] == "hit"
    np.testing.assert_array_equal(recomputed.query_xy_px, warm.query_xy_px)
    assert not list(cache_dir.rglob("*.tmp"))


def test_correspondence_cache_never_crosses_seed_manifest_or_worker_implementation(tmp_path: Path) -> None:
    worker_script, request_log = _write_fake_cache_worker(tmp_path)
    cache_dir = tmp_path / "correspondences"
    query = np.zeros((60, 80, 3), dtype=np.uint8)
    render = np.full((40, 50, 3), 60, dtype=np.uint8)

    def matcher(seed: int, max_matches: int) -> WorkerMatcher:
        return WorkerMatcher(
            command=(sys.executable, str(worker_script)),
            matcher_id="fake-cache",
            model_manifest={"artifacts": [], "inference": {"max_matches": max_matches}},
            seed=seed,
            cache_dir=cache_dir,
        )

    first = matcher(7, 100).match(query, render)
    changed_seed = matcher(8, 100).match(query, render)
    changed_manifest = matcher(7, 200).match(query, render)
    worker_script.write_text(worker_script.read_text() + "# implementation revision\n")
    changed_worker = matcher(7, 100).match(query, render)

    assert len(request_log.read_text().splitlines()) == 4
    keys = {result.diagnostics["cache"]["key"] for result in (first, changed_seed, changed_manifest, changed_worker)}
    assert len(keys) == 4
    assert all(
        result.diagnostics["cache"]["status"] == "miss"
        for result in (
            first,
            changed_seed,
            changed_manifest,
            changed_worker,
        )
    )


def test_orientation_guided_fan_allows_an_exact_seed_heading() -> None:
    config = RenderMatchConfig(orientation_prior_half_span_deg=0.0)

    config.validate()

    assert render_fan_yaws(181.0, True, config) == (-179.0,)
    assert RenderMatchConfig().expand_pnp_search_radii_from_prior_sigma is True


def test_explicit_render_seed_places_fan_independently_of_pose_prior(scene: Scene) -> None:
    rendered_poses: list[CameraExtrinsics] = []

    class RecordingRenderer(TerrainViewRenderer):
        def render(self, terrain, intrinsics, extrinsics, **kwargs):
            rendered_poses.append(extrinsics)
            return super().render(terrain, intrinsics, extrinsics, **kwargs)

    class EmptyMatcher:
        def identity(self):
            return {"id": "empty-render-seed-test"}

        def match(self, query_rgb, render_rgb):
            del query_rgb, render_rgb
            return MatchSet(
                query_xy_px=np.empty((0, 2)),
                render_xy_px=np.empty((0, 2)),
                confidence=np.empty(0),
            )

    prior_pose = _placed_pose(scene)
    prior = PosePrior(
        position=prior_pose.position,
        yaw_deg=prior_pose.yaw_deg,
        pitch_deg=prior_pose.pitch_deg,
        horizontal_sigma_m=75.0,
        vertical_sigma_m=40.0,
        yaw_sigma_deg=12.0,
        pitch_sigma_deg=8.0,
    )
    seed_position = LocalPoint(
        east_m=2_400.0,
        north_m=900.0,
        up_m=scene.terrain.elevation_at(2_400.0, 900.0) + 6.0,
    )
    render_seed = RenderSeed(position=seed_position, yaw_deg=112.0, pitch_deg=-3.0)
    camera = CameraModel(width_px=80, height_px=60, horizontal_fov_deg=60.0)

    result = solve_render_match_pose(
        scene.terrain,
        np.full((60, 80, 3), 128, dtype=np.uint8),
        camera,
        prior,
        EmptyMatcher(),
        use_position_prior=False,
        use_orientation_prior=True,
        render_seed=render_seed,
        config=RenderMatchConfig(
            render_width_px=64,
            render_height_px=64,
            yaw_step_deg=30.0,
            orientation_prior_half_span_deg=30.0,
            refinement_passes=0,
        ),
        renderer=RecordingRenderer(),
    )

    assert result.status == "abstained"
    assert len(rendered_poses) == 3
    assert [pose.yaw_deg for pose in rendered_poses] == pytest.approx([82.0, 112.0, 142.0])
    assert all(pose.position == seed_position for pose in rendered_poses)
    assert all(pose.position != prior.position for pose in rendered_poses)
    seed_record = result.diagnostics["render_seed"]
    assert seed_record["schema"] == RENDER_SEED_SCHEMA
    assert seed_record["source"] == "explicit_argument"
    assert seed_record["position"] == seed_position.model_dump(mode="json")
    assert result.diagnostics["pose_prior"]["position"] == prior.position.model_dump(mode="json")
    assert result.diagnostics["pose_prior"]["position_regularization_enabled"] is False


def test_exact_heading_seed_batch_is_one_ordered_match_call_and_never_stops_early(
    scene: Scene,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_by_east = {
        100.0: "candidate-a",
        200.0: "candidate-b",
        300.0: "candidate-c",
    }
    marker_by_candidate = {candidate_id: index for index, candidate_id in enumerate(candidate_by_east.values(), 1)}
    events: list[str] = []
    batch_marker_orders: list[tuple[int, ...]] = []
    pnp_calls: list[tuple[str, int, PosePrior, CameraExtrinsics]] = []
    validation_calls: list[str] = []
    fan_call_count = 0

    def candidate_id(position: LocalPoint) -> str:
        return candidate_by_east[position.east_m]

    class RecordingRenderer(TerrainViewRenderer):
        def render(self, terrain, intrinsics, extrinsics, **kwargs):
            del terrain, kwargs
            identifier = candidate_id(extrinsics.position)
            stage = "initial" if intrinsics.width_px == 64 else "validation"
            events.append(f"render-{stage}:{identifier}:{extrinsics.yaw_deg}")
            height = intrinsics.height_px
            width = intrinsics.width_px
            marker = marker_by_candidate[identifier]
            return TerrainRenderBundle(
                rgb=np.full((height, width, 3), marker, dtype=np.uint8),
                forward_depth_m=np.full((height, width), 1_000.0),
                world_xyz_m=np.zeros((height, width, 3)),
                world_normals=np.zeros((height, width, 3)),
                terrain_mask=np.ones((height, width), dtype=np.bool_),
                appearance_mask=np.ones((height, width), dtype=np.bool_),
                skyline_profile=np.zeros(width),
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                modality="hillshade",
                provenance={"candidate_id": identifier, "stage": stage},
            )

    render_x = np.tile(np.linspace(5.0, 58.0, 8), 4)
    render_y = np.repeat(np.linspace(5.0, 58.0, 4), 8)
    query_x = np.tile(np.linspace(5.0, 74.0, 8), 4)
    query_y = np.repeat(np.linspace(5.0, 54.0, 4), 8)

    class OneBatchMatcher:
        def identity(self):
            return {"id": "one-batch-test"}

        def match_many(self, _query, renders):
            markers = tuple(int(render[0, 0, 0]) for render in renders)
            batch_marker_orders.append(markers)
            events.append("match-many:" + ",".join(map(str, markers)))
            # The production batch must isolate its frozen query/render inputs
            # even from an ill-behaved in-process matcher.
            _query.fill(0)
            for render in renders:
                render.fill(0)
            return [
                MatchSet(
                    query_xy_px=np.column_stack((query_x, query_y)),
                    render_xy_px=np.column_stack((render_x, render_y)),
                    confidence=np.ones(len(render_x)),
                )
                for _render in renders
            ]

        def match(self, query_rgb, render_rgb):
            del query_rgb, render_rgb
            raise AssertionError("the exact-heading batch must not use scalar matching")

    prior_pose = _placed_pose(scene)
    prior = PosePrior(
        position=prior_pose.position,
        yaw_deg=prior_pose.yaw_deg,
        pitch_deg=prior_pose.pitch_deg,
        horizontal_sigma_m=80.0,
        vertical_sigma_m=40.0,
        yaw_sigma_deg=12.0,
        pitch_sigma_deg=7.0,
    )
    hypotheses = (
        IdentifiedRenderSeed(
            "candidate-a",
            RenderSeed(LocalPoint(east_m=100.0, north_m=-500.0, up_m=1_500.0), 10.0, -2.0),
        ),
        IdentifiedRenderSeed(
            "candidate-b",
            RenderSeed(LocalPoint(east_m=200.0, north_m=-600.0, up_m=1_600.0), 181.0, 3.0),
        ),
        IdentifiedRenderSeed(
            "candidate-c",
            RenderSeed(LocalPoint(east_m=300.0, north_m=-700.0, up_m=1_700.0), -40.0, 1.0),
        ),
    )
    real_match_image_fan = render_match_module.match_image_fan

    def recording_match_image_fan(matcher, query, renders):
        nonlocal fan_call_count
        fan_call_count += 1
        return real_match_image_fan(matcher, query, renders)

    def solved_fit_pose_ransac(*args, **kwargs):
        initial = args[4]
        fit_prior = kwargs["prior"]
        fit_config = kwargs["config"]
        assert isinstance(initial, CameraExtrinsics)
        assert isinstance(fit_prior, PosePrior)
        assert isinstance(fit_config, PoseRansacConfig)
        identifier = candidate_id(initial.position)
        events.append(f"pnp:{identifier}")
        pnp_calls.append((identifier, fit_config.seed, fit_prior, initial))
        count = len(args[0])
        return PoseRansacResult(
            status="solved",
            extrinsics=initial,
            inlier_mask=np.ones(count, dtype=np.bool_),
            reprojection_error_px=np.zeros(count),
            diagnostics={
                "inliers": count,
                "inlier_ratio": 1.0,
                "median_reprojection_error_px": 0.0,
            },
        )

    def recording_candidate_validation(source, *_args, **_kwargs):
        identifier = candidate_id(source.render.extrinsics.position)
        events.append(f"validate:{identifier}")
        validation_calls.append(identifier)
        passed = identifier != "candidate-a"
        return {
            "schema": CANDIDATE_VALIDATION_SCHEMA,
            "enabled": True,
            "passed": passed,
            "failures": [] if passed else ["deliberate_first_candidate_rejection"],
            "uses_reference_truth": False,
        }

    monkeypatch.setattr(render_match_module, "match_image_fan", recording_match_image_fan)
    monkeypatch.setattr(render_match_module, "fit_pose_ransac", solved_fit_pose_ransac)
    monkeypatch.setattr(render_match_module, "_validate_candidate_pose", recording_candidate_validation)
    camera = CameraModel(width_px=80, height_px=60, horizontal_fov_deg=60.0)
    config = RenderMatchConfig(
        render_width_px=64,
        render_height_px=64,
        orientation_prior_half_span_deg=0.0,
        refinement_passes=0,
    )
    query = np.full((60, 80, 3), 128, dtype=np.uint8)

    first = solve_render_match_pose_batch(
        scene.terrain,
        query,
        camera,
        prior,
        OneBatchMatcher(),
        hypotheses,
        use_position_prior=True,
        use_orientation_prior=False,
        config=config,
        seed=99,
        renderer=RecordingRenderer(),
    )

    assert first.ordered_candidate_ids == ("candidate-a", "candidate-b", "candidate-c")
    assert [item.result.status for item in first.results] == ["abstained", "solved", "solved"]
    assert batch_marker_orders == [(1, 2, 3)]
    assert fan_call_count == 1
    assert events[:4] == [
        "render-initial:candidate-a:10.0",
        "render-initial:candidate-b:-179.0",
        "render-initial:candidate-c:-40.0",
        "match-many:1,2,3",
    ]
    assert [item[0] for item in pnp_calls] == ["candidate-a", "candidate-b", "candidate-c"]
    assert validation_calls == ["candidate-a", "candidate-b", "candidate-c"]
    stable_fit_prior = pnp_calls[0][2]
    assert stable_fit_prior is not prior
    assert stable_fit_prior.model_dump(mode="json") == prior.model_dump(mode="json")
    assert all(fit_prior is stable_fit_prior for _, _, fit_prior, _ in pnp_calls)
    assert [initial.position for _, _, _, initial in pnp_calls] == [
        hypothesis.render_seed.position for hypothesis in hypotheses
    ]
    assert all(initial.position != fit_prior.position for _, _, fit_prior, initial in pnp_calls)
    assert [initial.pitch_deg for _, _, _, initial in pnp_calls] == pytest.approx([-2.0, 3.0, 1.0])
    assert np.all(query == 128)
    assert first.provenance["schema"] == RENDER_SEED_BATCH_SCHEMA
    assert first.provenance["matching"] == {
        "match_image_fan_invocations": 1,
        "matcher_match_many_invocations": 1,
        "render_count_in_single_invocation": 3,
        "all_seed_renders_completed_before_invocation": True,
        "scalar_fallback_allowed": False,
        "matcher_received_detached_mutable_rgb_copies": True,
    }
    assert first.provenance["shared_pose_prior"]["position"] == prior.position.model_dump(mode="json")
    assert first.provenance["exact_heading"]["yaw_fan_enabled"] is False
    first_pnp_seeds = {identifier: rng_seed for identifier, rng_seed, _, _ in pnp_calls}
    first_candidate_seeds = {item.candidate_id: item.rng_seed for item in first.results}

    events.clear()
    pnp_calls.clear()
    validation_calls.clear()
    reordered = solve_render_match_pose_batch(
        scene.terrain,
        query,
        camera,
        prior,
        OneBatchMatcher(),
        (hypotheses[2], hypotheses[0], hypotheses[1]),
        use_position_prior=True,
        use_orientation_prior=False,
        config=config,
        seed=99,
        renderer=RecordingRenderer(),
    )

    assert reordered.ordered_candidate_ids == ("candidate-c", "candidate-a", "candidate-b")
    assert batch_marker_orders[-1] == (3, 1, 2)
    assert fan_call_count == 2
    assert {identifier: rng_seed for identifier, rng_seed, _, _ in pnp_calls} == first_pnp_seeds
    assert {item.candidate_id: item.rng_seed for item in reordered.results} == first_candidate_seeds
    assert {item.candidate_id: item for item in reordered.results} == {
        item.candidate_id: item for item in first.results
    }
    reordered_stable_prior = pnp_calls[0][2]
    assert reordered_stable_prior is not prior
    assert all(fit_prior is reordered_stable_prior for _, _, fit_prior, _ in pnp_calls)
    assert np.all(query == 128)


def test_exact_heading_seed_batch_rejects_invalid_identity_fans_refinement_and_scalar_matchers(
    scene: Scene,
) -> None:
    pose = _placed_pose(scene)
    prior = PosePrior(
        position=pose.position,
        yaw_deg=pose.yaw_deg,
        pitch_deg=pose.pitch_deg,
        horizontal_sigma_m=50.0,
        vertical_sigma_m=30.0,
        yaw_sigma_deg=10.0,
        pitch_sigma_deg=5.0,
    )
    seed = RenderSeed(position=pose.position, yaw_deg=pose.yaw_deg, pitch_deg=pose.pitch_deg)
    valid = IdentifiedRenderSeed("candidate", seed)
    query = np.full((60, 80, 3), 128, dtype=np.uint8)
    camera = CameraModel(width_px=80, height_px=60, horizontal_fov_deg=60.0)

    class NeverBatchMatcher:
        def identity(self):
            return {"id": "must-not-run"}

        def match_many(self, _query, _renders):
            raise AssertionError("invalid batches must fail before matching")

        def match(self, query_rgb, render_rgb):
            del query_rgb, render_rgb
            raise AssertionError("invalid batches must fail before matching")

    base_config = RenderMatchConfig(
        render_width_px=64,
        render_height_px=64,
        orientation_prior_half_span_deg=0.0,
        refinement_passes=0,
    )
    invalid_batches: tuple[tuple[IdentifiedRenderSeed, ...], ...] = (
        (),
        (IdentifiedRenderSeed("", seed),),
        (IdentifiedRenderSeed(" candidate", seed),),
        (valid, IdentifiedRenderSeed("candidate", seed)),
        (cast(IdentifiedRenderSeed, object()),),
    )
    for invalid in invalid_batches:
        with pytest.raises(ValueError):
            solve_render_match_pose_batch(
                scene.terrain,
                query,
                camera,
                prior,
                NeverBatchMatcher(),
                invalid,
                use_position_prior=True,
                use_orientation_prior=True,
                config=base_config,
            )

    with pytest.raises(ValueError, match="refinement_passes=0"):
        solve_render_match_pose_batch(
            scene.terrain,
            query,
            camera,
            prior,
            NeverBatchMatcher(),
            (valid,),
            use_position_prior=True,
            use_orientation_prior=True,
            config=replace(base_config, refinement_passes=1),
        )
    with pytest.raises(ValueError, match="cannot render a yaw fan"):
        solve_render_match_pose_batch(
            scene.terrain,
            query,
            camera,
            prior,
            NeverBatchMatcher(),
            (valid,),
            use_position_prior=True,
            use_orientation_prior=True,
            config=replace(base_config, orientation_prior_half_span_deg=30.0),
        )

    class ScalarOnlyMatcher:
        def identity(self):
            return {"id": "scalar-only"}

        def match(self, query_rgb, render_rgb):
            del query_rgb, render_rgb
            raise AssertionError("scalar fallback must never run")

    with pytest.raises(ValueError, match="callable match_many"):
        solve_render_match_pose_batch(
            scene.terrain,
            query,
            camera,
            prior,
            ScalarOnlyMatcher(),
            (valid,),
            use_position_prior=True,
            use_orientation_prior=True,
            config=base_config,
        )


def test_render_seed_initializes_pnp_but_prior_still_anchors_regularization_and_clearance(
    scene: Scene,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior_east_m = 0.0
    prior_north_m = -800.0
    prior_ground_m = scene.terrain.elevation_at(prior_east_m, prior_north_m)
    prior = PosePrior(
        position=LocalPoint(east_m=prior_east_m, north_m=prior_north_m, up_m=prior_ground_m + 8.0),
        yaw_deg=5.0,
        pitch_deg=2.0,
        horizontal_sigma_m=60.0,
        vertical_sigma_m=30.0,
        yaw_sigma_deg=10.0,
        pitch_sigma_deg=6.0,
    )
    seed_east_m = 1_500.0
    seed_north_m = -2_500.0
    seed_ground_m = scene.terrain.elevation_at(seed_east_m, seed_north_m)
    render_seed = RenderSeed(
        position=LocalPoint(east_m=seed_east_m, north_m=seed_north_m, up_m=seed_ground_m + 30.0),
        yaw_deg=25.0,
        pitch_deg=-4.0,
    )
    intrinsics = CameraIntrinsics.from_horizontal_fov(128, 96, 60.0)
    render = TerrainViewRenderer().render(
        scene.terrain,
        intrinsics,
        CameraExtrinsics(
            position=render_seed.position,
            yaw_deg=render_seed.yaw_deg,
            pitch_deg=-8.0,
            roll_deg=0.0,
        ),
        modality="normal",
        terrain_stride=2,
    )
    terrain_rows_columns = np.argwhere(render.terrain_mask)
    render_candidates = terrain_rows_columns[:, ::-1].astype(np.float64)
    candidate_lift = lift_render_pixels(render, render_candidates)
    render_xy = render_candidates[candidate_lift.valid][:16]
    assert len(render_xy) == 16
    matches = MatchSet(
        # Deliberately degenerate so the real fitter records the prior/clearance
        # policy and then abstains before expensive nonlinear RANSAC.
        query_xy_px=np.tile([64.0, 48.0], (len(render_xy), 1)),
        render_xy_px=render_xy,
        confidence=np.ones(len(render_xy)),
    )
    query_camera = CameraModel(width_px=128, height_px=96, horizontal_fov_deg=60.0, projection="cyltan")
    settings = RenderMatchConfig(
        min_lifted_matches_per_frame=12,
        expand_pnp_search_radii_from_prior_sigma=False,
        refinement_passes=0,
        candidate_validation=CandidateValidationConfig(enabled=False),
        pnp=PoseRansacConfig(
            horizontal_search_radius_m=40.0,
            vertical_search_radius_m=25.0,
            prior_weight_px=9.0,
        ),
    )
    captured: dict[str, object] = {}
    real_fit_pose_ransac = render_match_module.fit_pose_ransac

    def recording_fit_pose_ransac(*args, **kwargs):
        captured["initial"] = args[4]
        captured.update(kwargs)
        return real_fit_pose_ransac(*args, **kwargs)

    monkeypatch.setattr(render_match_module, "fit_pose_ransac", recording_fit_pose_ransac)

    attempt = render_match_module._match_and_fit_frame(
        0,
        render,
        matches,
        query_camera,
        prior,
        scene.terrain,
        True,
        True,
        settings,
        17,
        render_seed=render_seed,
    )

    assert captured["prior"] is prior
    assert captured["use_position_prior"] is True
    assert captured["use_orientation_prior"] is True
    captured_config = captured["config"]
    assert isinstance(captured_config, PoseRansacConfig)
    assert captured_config.prior_weight_px == pytest.approx(9.0)
    assert captured_config.horizontal_search_radius_m == pytest.approx(40.0)
    assert captured_config.vertical_search_radius_m == pytest.approx(25.0)
    initial = captured["initial"]
    assert isinstance(initial, CameraExtrinsics)
    assert initial.position == render_seed.position
    assert initial.pitch_deg == pytest.approx(render_seed.pitch_deg)
    assert attempt.pnp_result is not None
    constraint = attempt.pnp_result.diagnostics["clearance_constraint"]
    assert constraint["initial_clearance_m"] == pytest.approx(30.0)
    assert constraint["raw_prior_clearance_m"] == pytest.approx(8.0)
    assert constraint["anchor_clearance_m"] == pytest.approx(8.0)
    assert constraint["prior_ground_sample"]["elevation_m"] == pytest.approx(prior_ground_m)
    assert attempt.record["pose_prior_usage"]["clearance_anchor_uses_supplied_prior"] is True


def test_refinement_moves_render_seed_without_replacing_pose_prior(
    scene: Scene,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundles: dict[int, TerrainRenderBundle] = {}
    rendered_poses: list[CameraExtrinsics] = []

    class RecordingRenderer(TerrainViewRenderer):
        def render(self, terrain, intrinsics, extrinsics, **kwargs):
            bundle = super().render(terrain, intrinsics, extrinsics, **kwargs)
            rendered_poses.append(extrinsics)
            bundles[id(bundle.rgb)] = bundle
            return bundle

    class LiftableMatcher:
        def identity(self):
            return {"id": "liftable-refinement-test"}

        def match(self, query_rgb, render_rgb):
            bundle = bundles[id(render_rgb)]
            rows_columns = np.argwhere(bundle.terrain_mask)
            candidates = rows_columns[:, ::-1].astype(np.float64)
            lifted = lift_render_pixels(bundle, candidates)
            render_xy = candidates[lifted.valid][:20]
            assert len(render_xy) == 20
            count = len(render_xy)
            query_xy = np.column_stack(
                (
                    np.linspace(5.0, query_rgb.shape[1] - 6.0, count),
                    np.linspace(5.0, query_rgb.shape[0] - 6.0, count),
                )
            )
            return MatchSet(
                query_xy_px=query_xy,
                render_xy_px=render_xy,
                confidence=np.ones(count),
            )

    prior_pose = _placed_pose(scene)
    prior = PosePrior(
        position=prior_pose.position,
        yaw_deg=prior_pose.yaw_deg,
        pitch_deg=prior_pose.pitch_deg,
        horizontal_sigma_m=80.0,
        vertical_sigma_m=35.0,
        yaw_sigma_deg=12.0,
        pitch_sigma_deg=7.0,
    )
    seed_position = LocalPoint(
        east_m=2_000.0,
        north_m=-1_000.0,
        up_m=scene.terrain.elevation_at(2_000.0, -1_000.0) + 5.0,
    )
    render_seed = RenderSeed(position=seed_position, yaw_deg=100.0, pitch_deg=-2.0)
    pnp_initials: list[CameraExtrinsics] = []
    pnp_priors: list[PosePrior] = []
    pnp_configs: list[PoseRansacConfig] = []
    pnp_prior_flags: list[tuple[bool, bool]] = []

    def solved_fit_pose_ransac(*args, **kwargs):
        initial = args[4]
        assert isinstance(initial, CameraExtrinsics)
        fit_prior = kwargs["prior"]
        fit_config = kwargs["config"]
        assert isinstance(fit_prior, PosePrior)
        assert isinstance(fit_config, PoseRansacConfig)
        pnp_initials.append(initial)
        pnp_priors.append(fit_prior)
        pnp_configs.append(fit_config)
        pnp_prior_flags.append((kwargs["use_position_prior"], kwargs["use_orientation_prior"]))
        if len(pnp_initials) <= 3:
            estimate = initial.model_copy(
                update={
                    "position": LocalPoint(
                        east_m=initial.position.east_m + 25.0,
                        north_m=initial.position.north_m - 15.0,
                        up_m=initial.position.up_m + 1.0,
                    ),
                    "yaw_deg": initial.yaw_deg + 1.0,
                }
            )
        else:
            estimate = initial
        correspondence_count = len(args[0])
        return PoseRansacResult(
            status="solved",
            extrinsics=estimate,
            inlier_mask=np.ones(correspondence_count, dtype=np.bool_),
            reprojection_error_px=np.zeros(correspondence_count),
            diagnostics={
                "inliers": correspondence_count,
                "inlier_ratio": 1.0,
                "median_reprojection_error_px": 0.0,
                "candidate_pose": estimate.model_dump(mode="json"),
            },
        )

    monkeypatch.setattr(render_match_module, "fit_pose_ransac", solved_fit_pose_ransac)
    camera = CameraModel(width_px=120, height_px=80, horizontal_fov_deg=60.0)
    result = solve_render_match_pose(
        scene.terrain,
        np.full((80, 120, 3), 128, dtype=np.uint8),
        camera,
        prior,
        LiftableMatcher(),
        use_position_prior=True,
        use_orientation_prior=True,
        render_seed=render_seed,
        config=RenderMatchConfig(
            render_width_px=96,
            render_height_px=64,
            yaw_step_deg=30.0,
            orientation_prior_half_span_deg=30.0,
            terrain_stride=2,
            refinement_passes=1,
            candidate_validation=CandidateValidationConfig(enabled=False),
            pnp=PoseRansacConfig(prior_weight_px=4.0),
        ),
        renderer=RecordingRenderer(),
    )

    assert result.solved
    assert len(pnp_initials) == 4
    assert all(fit_prior is prior for fit_prior in pnp_priors)
    assert pnp_prior_flags == [(True, True)] * 4
    assert all(fit_config.prior_weight_px == pytest.approx(4.0) for fit_config in pnp_configs)
    assert all(initial.position == seed_position for initial in pnp_initials[:3])
    refinement_initial = pnp_initials[-1]
    assert refinement_initial.position != seed_position
    assert rendered_poses[-1].position == refinement_initial.position
    assert result.diagnostics["refinement"]["render_seed"]["source"] == "first_pass_estimate"
    assert result.diagnostics["pose_prior"]["position"] == prior.position.model_dump(mode="json")


def test_render_pipeline_serializes_shared_worker_batch_once(scene: Scene) -> None:
    batch = {
        "schema": "peakle_match_batch_result_v1",
        "status": "ok",
        "request_sha256": "request-only-once",
        "provenance": {"worker": {"implementation_sha256": "worker-hash"}},
    }

    class EmptyBatchMatcher:
        def identity(self):
            return {"id": "empty-external-worker"}

        def match_many(self, _query, renders):
            return [
                MatchSet(
                    query_xy_px=np.empty((0, 2)),
                    render_xy_px=np.empty((0, 2)),
                    confidence=np.empty(0),
                    diagnostics={"id": index},
                    provenance={**self.identity(), "worker_batch": batch},
                )
                for index, _render in enumerate(renders)
            ]

    pose = _placed_pose(scene)
    prior = PosePrior(
        position=pose.position,
        yaw_deg=pose.yaw_deg,
        pitch_deg=pose.pitch_deg,
        horizontal_sigma_m=50.0,
        vertical_sigma_m=30.0,
        yaw_sigma_deg=10.0,
        pitch_sigma_deg=5.0,
    )
    camera = CameraModel(width_px=80, height_px=60, horizontal_fov_deg=60.0, projection="pinhole")
    result = solve_render_match_pose(
        scene.terrain,
        np.zeros((60, 80, 3), dtype=np.uint8),
        camera,
        prior,
        EmptyBatchMatcher(),
        use_position_prior=True,
        use_orientation_prior=True,
        config=RenderMatchConfig(
            render_width_px=64,
            render_height_px=64,
            orientation_prior_half_span_deg=30.0,
            refinement_passes=0,
        ),
    )

    assert result.diagnostics["matcher_batch"] == batch
    assert all("worker_batch" not in frame["matcher_diagnostics"] for frame in result.diagnostics["frames"])
    assert json.dumps(result.diagnostics).count("request-only-once") == 1


def test_external_matcher_preserves_a_structured_missing_model_skip(tmp_path: Path) -> None:
    worker_script = tmp_path / "missing_worker.py"
    worker_script.write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--request', required=True)
args = parser.parse_args()
request_path = Path(args.request)
request = json.loads(request_path.read_text())
(request_path.parent / request['outputs']['result_json']).write_text(json.dumps({
    'schema': 'peakle_match_batch_result_v1',
    'status': 'SKIPPED_MISSING_MODEL',
    'message': 'checkpoint absent',
}))
""".strip()
        + "\n"
    )
    matcher = WorkerMatcher(
        command=(sys.executable, str(worker_script)),
        matcher_id="missing-test",
        model_manifest={"artifacts": []},
        cache_dir=tmp_path / "missing-result-cache",
    )

    with pytest.raises(MatcherUnavailable, match="checkpoint absent"):
        matcher.match(
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.zeros((32, 32, 3), dtype=np.uint8),
        )
    assert not list((tmp_path / "missing-result-cache").rglob("*.json"))


def test_render_match_pipeline_improves_a_perturbed_position_with_injected_matches(scene: Scene) -> None:
    truth = _placed_pose(scene)
    camera = CameraModel(width_px=320, height_px=180, horizontal_fov_deg=60.0, projection="pinhole")

    class Shared:
        bundles = {}

    shared = Shared()

    class RecordingRenderer:
        def __init__(self) -> None:
            self.delegate = TerrainViewRenderer()

        def render(self, *args, **kwargs):
            bundle = self.delegate.render(*args, **kwargs)
            shared.bundles[id(bundle.rgb)] = bundle
            return bundle

    class GeometryMatcher:
        def __init__(self, *, corrupt_holdout: bool = False) -> None:
            self.corrupt_holdout = corrupt_holdout

        def identity(self):
            return {
                "id": "test_geometry_correspondences",
                "ranking_eligible": False,
                "corrupt_holdout": self.corrupt_holdout,
            }

        def match(self, query_rgb, render_rgb):
            bundle = shared.bundles[id(render_rgb)]
            rows, columns = np.nonzero(bundle.terrain_mask)
            selected = (
                (columns % 8 == 0)
                & (rows % 6 == 0)
                & (columns > 4)
                & (columns < render_rgb.shape[1] - 5)
                & (rows > 4)
                & (rows < render_rgb.shape[0] - 5)
            )
            rows = rows[selected]
            columns = columns[selected]
            query_xy, valid = project_world_points(bundle.world_xyz_m[rows, columns], camera, truth)
            inside = (
                valid
                & (query_xy[:, 0] > 3)
                & (query_xy[:, 0] < camera.width_px - 4)
                & (query_xy[:, 1] > 3)
                & (query_xy[:, 1] < camera.height_px - 4)
            )
            retained_query_xy = query_xy[inside].copy()
            if self.corrupt_holdout:
                validation_config = CandidateValidationConfig()
                query_sha256 = hashlib.sha256(memoryview(np.ascontiguousarray(query_rgb)).cast("B")).hexdigest()
                heldout_fold = _query_holdout_fold(query_sha256, validation_config)
                heldout = _query_spatial_holdout_mask(
                    retained_query_xy,
                    camera,
                    validation_config,
                    heldout_fold,
                )
                cell_width = camera.width_px / validation_config.query_grid_columns
                cell_left = np.floor(retained_query_xy[:, 0] / cell_width) * cell_width
                shift_right = retained_query_xy[:, 0] - cell_left < cell_width - 6.5
                retained_query_xy[heldout, 0] += np.where(shift_right[heldout], 6.0, -6.0)
                np.testing.assert_array_equal(
                    _query_spatial_holdout_mask(
                        retained_query_xy,
                        camera,
                        validation_config,
                        heldout_fold,
                    ),
                    heldout,
                )
            render_xy = np.column_stack((columns[inside], rows[inside])).astype(float)
            return MatchSet(
                query_xy_px=retained_query_xy,
                render_xy_px=render_xy,
                confidence=np.ones(int(inside.sum())),
                provenance=self.identity(),
            )

    perturbed_position = LocalPoint(
        east_m=truth.position.east_m + 120.0,
        north_m=truth.position.north_m - 80.0,
        up_m=truth.position.up_m + 35.0,
    )
    prior = PosePrior(
        position=perturbed_position,
        yaw_deg=truth.yaw_deg + 15.0,
        pitch_deg=truth.pitch_deg + 3.0,
        horizontal_sigma_m=150.0,
        vertical_sigma_m=60.0,
        yaw_sigma_deg=15.0,
        pitch_sigma_deg=8.0,
    )
    config = RenderMatchConfig(
        render_width_px=220,
        render_height_px=150,
        terrain_stride=2,
        orientation_prior_half_span_deg=30.0,
        refinement_passes=0,
        candidate_validation=CandidateValidationConfig(enabled=False),
        pnp=PoseRansacConfig(
            iterations=25,
            sample_size=6,
            min_correspondences=10,
            min_inliers=8,
            min_query_y_span_fraction=0.03,
            horizontal_search_radius_m=500.0,
            vertical_search_radius_m=150.0,
            yaw_search_radius_deg=40.0,
            seed=9,
        ),
    )

    result = solve_render_match_pose(
        scene.terrain,
        np.full((camera.height_px, camera.width_px, 3), 128, dtype=np.uint8),
        camera,
        prior,
        GeometryMatcher(),
        use_position_prior=True,
        use_orientation_prior=True,
        config=config,
        renderer=RecordingRenderer(),
        seed=8,
    )

    assert result.solved
    assert result.extrinsics is not None
    prior_error = math.hypot(
        prior.position.east_m - truth.position.east_m,
        prior.position.north_m - truth.position.north_m,
    )
    solved_error = math.hypot(
        result.extrinsics.position.east_m - truth.position.east_m,
        result.extrinsics.position.north_m - truth.position.north_m,
    )
    assert prior_error > 140.0
    assert solved_error < 0.01
    assert result.diagnostics["estimator_inputs"]["source_depth_pfm"] is False

    gated_prior = prior.model_copy(update={"position": truth.position})
    gated_result = solve_render_match_pose(
        scene.terrain,
        np.full((camera.height_px, camera.width_px, 3), 128, dtype=np.uint8),
        camera,
        gated_prior,
        GeometryMatcher(),
        use_position_prior=True,
        use_orientation_prior=True,
        config=replace(config, candidate_validation=CandidateValidationConfig()),
        renderer=RecordingRenderer(),
        seed=8,
    )

    assert gated_result.solved, gated_result.diagnostics.get("candidate_validation")
    gate = gated_result.diagnostics["candidate_validation"]
    assert gate["passed"] is True
    assert gate["withheld_from_geometric_pose_fit"] is True
    assert gate["withheld_from_geometric_frame_ranking"] is True
    assert gate["matcher_used_full_query_image"] is True
    assert gate["worker_candidate_selection_precedes_holdout"] is True
    assert gate["runner_up_retry_attempted"] is False

    rejected_result = solve_render_match_pose(
        scene.terrain,
        np.full((camera.height_px, camera.width_px, 3), 128, dtype=np.uint8),
        camera,
        gated_prior,
        GeometryMatcher(corrupt_holdout=True),
        use_position_prior=True,
        use_orientation_prior=True,
        config=replace(config, candidate_validation=CandidateValidationConfig()),
        renderer=RecordingRenderer(),
        seed=8,
    )

    assert rejected_result.status == "abstained"
    assert rejected_result.extrinsics is None
    assert rejected_result.candidates == ()
    assert rejected_result.diagnostics["final_pnp"]["candidate_pose"] is not None
    assert rejected_result.diagnostics["abstain_reason"] == "candidate_pose_holdout_validation_failed"
    assert rejected_result.diagnostics["runner_up_retry_attempted"] is False
    rejected_gate = rejected_result.diagnostics["candidate_validation"]
    assert rejected_gate["withheld_from_geometric_frame_ranking"] is True
    assert rejected_gate["matcher_used_full_query_image"] is True
    assert "heldout_joint_consensus_below_acceptance_gate" in rejected_gate["failures"]
