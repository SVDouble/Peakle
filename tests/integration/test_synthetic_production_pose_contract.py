from __future__ import annotations

import json

import numpy as np
import pytest

from peakle.domain.camera import CameraExtrinsics
from peakle.domain.coordinates import GeoPoint, LocalPoint
from peakle.domain.terrain import TerrainMap, TerrainSpec
from peakle.localize.gtrefine import crop_az_deg, dem_depth_image
from peakle.localize.pfm_geometry_rerank import (
    PfmGeometryRerankConfig,
    build_pfm_geometry_rerank,
    evaluate_pfm_geometry_rerank,
)
from peakle.localize.skyline_atlas import (
    SkylineAtlasConfig,
    build_skyline_atlas,
    evaluate_skyline_atlas,
)
from peakle.localize.solve import HorizonProfile

WIDTH_PX = 64
HEIGHT_PX = 48
FOV_DEG = 80.0


@pytest.fixture(scope="module")
def asymmetric_terrain() -> TerrainMap:
    size = 65
    x_m = np.linspace(-2_400.0, 2_400.0, size)
    y_m = np.linspace(-2_400.0, 2_400.0, size)
    x_grid, y_grid = np.meshgrid(x_m, y_m)
    elevation = np.full_like(x_grid, 100.0)
    for east_m, north_m, height_m, sigma_m in (
        (1_050.0, 700.0, 950.0, 240.0),
        (-1_150.0, 850.0, 680.0, 320.0),
        (-450.0, -1_200.0, 820.0, 260.0),
        (1_300.0, -700.0, 520.0, 180.0),
    ):
        distance_squared = (x_grid - east_m) ** 2 + (y_grid - north_m) ** 2
        elevation += height_m * np.exp(-distance_squared / (2.0 * sigma_m**2))
    return TerrainMap(
        spec=TerrainSpec(
            origin=GeoPoint(latitude_deg=46.0, longitude_deg=8.0, elevation_m=100.0),
            width_m=4_800.0,
            height_m=4_800.0,
            grid_width=size,
            grid_height=size,
            min_elevation_m=100.0,
            max_elevation_m=1_100.0,
            seed=0,
        ),
        x_m=x_m,
        y_m=y_m,
        elevation_m=elevation,
        latitude_deg=46.0 + y_grid / 111_320.0,
        longitude_deg=8.0 + x_grid / 80_000.0,
    )


def _truth(terrain: TerrainMap, east_m: float, north_m: float, yaw_deg: float) -> CameraExtrinsics:
    return CameraExtrinsics(
        position=LocalPoint(
            east_m=east_m,
            north_m=north_m,
            up_m=terrain.elevation_at(east_m, north_m) + 2.5,
        ),
        yaw_deg=yaw_deg,
        pitch_deg=0.0,
        roll_deg=0.0,
    )


def _skyline(terrain: TerrainMap, truth: CameraExtrinsics, *, ray_step_m: float) -> np.ndarray:
    return HorizonProfile(
        terrain,
        truth.position.up_m,
        step=ray_step_m,
        cam_e=truth.position.east_m,
        cam_n=truth.position.north_m,
    ).rows_cyl_tan(WIDTH_PX, HEIGHT_PX, FOV_DEG, truth.yaw_deg)


def _camera_ray_depth(terrain: TerrainMap, truth: CameraExtrinsics) -> np.ndarray:
    horizontal_depth, _hit, elevation_pixels, _elevations, _distances, _rows = dem_depth_image(
        terrain,
        truth.position.up_m,
        crop_az_deg(WIDTH_PX, FOV_DEG, truth.yaw_deg),
        WIDTH_PX,
        HEIGHT_PX,
        FOV_DEG,
        0.0,
        truth.position.east_m,
        truth.position.north_m,
        0.0,
        sub=1,
    )
    cosine = np.cos(elevation_pixels)
    return np.divide(
        horizontal_depth,
        cosine,
        out=np.full_like(horizontal_depth, np.nan),
        where=np.isfinite(horizontal_depth) & np.isfinite(cosine) & (cosine > 1e-6),
    )


def test_production_atlas_recovers_a_distinctive_synthetic_pose_after_freeze(
    asymmetric_terrain: TerrainMap,
) -> None:
    config = SkylineAtlasConfig(
        radius_m=100.0,
        spacing_m=100.0,
        yaw_step_deg=30.0,
        yaw_modes_per_position=3,
        yaw_mode_separation_deg=30.0,
        max_observed_columns=WIDTH_PX,
        ray_step_m=40.0,
    )
    truth = _truth(asymmetric_terrain, 100.0, -100.0, 60.0)
    observed = _skyline(asymmetric_terrain, truth, ray_step_m=config.ray_step_m)

    archive = build_skyline_atlas(
        asymmetric_terrain,
        observed,
        HEIGHT_PX,
        FOV_DEG,
        LocalPoint(east_m=0.0, north_m=0.0, up_m=-999.0),
        config=config,
    )
    frozen_record = json.dumps(archive.to_record(), sort_keys=True)

    evaluation = evaluate_skyline_atlas(archive, truth, top_ks=(1, len(archive.candidates)))

    assert json.dumps(archive.to_record(), sort_keys=True) == frozen_record
    assert archive.to_record()["numeric_evaluation_reference_used"] is False
    assert archive.selected.position.as_tuple() == pytest.approx(truth.position.as_tuple(), abs=1e-9)
    assert archive.selected.yaw_deg == pytest.approx(truth.yaw_deg)
    assert evaluation.winner_errors.errors.horizontal_position_m == 0.0
    assert evaluation.winner_errors.errors.yaw_deg == 0.0


def test_self_render_depth_is_an_upper_bound_for_reranking_an_ambiguous_frozen_atlas(
    asymmetric_terrain: TerrainMap,
) -> None:
    """Same-raycaster depth is an identity/upper-bound check, not independent evidence."""

    config = SkylineAtlasConfig(
        radius_m=100.0,
        spacing_m=100.0,
        yaw_step_deg=90.0,
        yaw_modes_per_position=4,
        yaw_mode_separation_deg=90.0,
        max_observed_columns=WIDTH_PX,
        ray_step_m=40.0,
    )
    truth = _truth(asymmetric_terrain, 0.0, 0.0, 0.0)
    complete_skyline = _skyline(asymmetric_terrain, truth, ray_step_m=config.ray_step_m)
    ambiguous_skyline = np.full_like(complete_skyline, np.nan)
    ambiguous_skyline[WIDTH_PX // 2] = complete_skyline[WIDTH_PX // 2]

    atlas = build_skyline_atlas(
        asymmetric_terrain,
        ambiguous_skyline,
        HEIGHT_PX,
        FOV_DEG,
        LocalPoint(east_m=0.0, north_m=0.0, up_m=-999.0),
        config=config,
    )
    frozen_atlas = atlas.to_record()
    frozen_atlas_json = json.dumps(frozen_atlas, sort_keys=True)

    atlas_evaluation = evaluate_skyline_atlas(atlas, truth, top_ks=(1, len(atlas.candidates)))

    assert atlas_evaluation.winner_errors.errors.horizontal_position_m > 100.0
    assert atlas_evaluation.winner_errors.errors.yaw_deg > 5.0
    assert atlas_evaluation.shortlist_gt_oracle.normalized_joint_error == 0.0
    assert atlas_evaluation.shortlist_top_k_reach[1] is False
    assert atlas_evaluation.shortlist_top_k_reach[len(atlas.candidates)] is True
    assert json.dumps(atlas.to_record(), sort_keys=True) == frozen_atlas_json

    # This source depth is rendered by the same terrain/raycast model used for
    # candidates.  It checks plumbing and the attainable identity ceiling only.
    source_depth = _camera_ray_depth(asymmetric_terrain, truth)
    rerank = build_pfm_geometry_rerank(
        asymmetric_terrain,
        None,
        source_depth,
        frozen_atlas,
        config=PfmGeometryRerankConfig(
            subsample=2,
            min_depth_m=50.0,
            min_source_family_px=2,
            min_common_depth_px=4,
        ),
    )
    frozen_rerank_json = json.dumps(rerank.to_record(), sort_keys=True)

    rerank_evaluation = evaluate_pfm_geometry_rerank(rerank, truth, top_ks=(1, len(rerank.candidates)))

    assert rerank.to_record()["numeric_evaluation_reference_used"] is False
    assert rerank.selected.candidate_id == atlas_evaluation.shortlist_gt_oracle.candidate_id
    assert rerank_evaluation.winner_errors.errors.horizontal_position_m == 0.0
    assert rerank_evaluation.winner_errors.errors.yaw_deg == 0.0
    assert rerank.selected.terms.depth_shape_loss == 0.0
    assert rerank.selected.terms.depth_overlap_f1 == 1.0
    assert json.dumps(rerank.to_record(), sort_keys=True) == frozen_rerank_json
