from __future__ import annotations

import json

import numpy as np
import pytest

from peakle.domain.camera import CameraExtrinsics
from peakle.domain.coordinates import GeoPoint, LocalPoint
from peakle.domain.terrain import TerrainMap, TerrainSpec
from peakle.localize.atlas_geometry import render_cyltan_candidate_depth
from peakle.localize.gtrefine import crop_az_deg, dem_depth_image
from peakle.localize.pfm_geometry_rerank import (
    PfmGeometryRerankConfig,
    build_pfm_geometry_rerank,
    evaluate_pfm_geometry_rerank,
)
from peakle.localize.photo_geometry_verifier import (
    PhotoGeometryVerifierConfig,
    build_photo_geometry_verifier,
    evaluate_photo_geometry_verifier,
    extract_photo_geometry_evidence,
)
from peakle.localize.skyline_atlas import (
    SkylineAtlasArchive,
    SkylineAtlasConfig,
    build_skyline_atlas,
    evaluate_skyline_atlas,
)
from peakle.localize.solve import HorizonProfile
from peakle.localize.typed_outlines import extract_typed_outlines

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


class _EncodedSyntheticEdges:
    """Decode an authoritative synthetic edge channel from photo RGB."""

    name = "synthetic-rgb-edge-channel"

    def detect(self, rgb: np.ndarray) -> np.ndarray:
        return (rgb[..., 0] > 0.5).astype(np.float64)


class _EncodedSyntheticDepth:
    """Decode far-is-large relative depth from the synthetic photo's green channel."""

    name = "synthetic-rgb-relative-depth-channel"

    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        return rgb[..., 1].astype(np.float64)


def _photo_model_provenance(name: str, digest_character: str) -> dict[str, object]:
    return {
        "name": name,
        "aggregate_sha256": digest_character * 64,
        "offline": True,
        "input": "photo_rgb",
        "synthetic_authoritative_channel": True,
    }


def _photo_observation_provenance(
    atlas: SkylineAtlasArchive,
    *,
    shared_candidate_render_encoded: bool,
) -> dict[str, object]:
    return {
        "track": "photo_auto",
        "source": "synthetic_rgb_authoritative_channel_control",
        "candidate": "predeclared_synthetic_scene",
        "selection_uses_reference_truth": False,
        # The verifier receives model outputs extracted from RGB, never a pose
        # or reference-depth argument. The extra field below separately admits
        # that the positive control's RGB itself encodes a shared DEM render.
        "evidence_generated_at_reference_pose": False,
        "source_atlas_sha256": atlas.archive_sha256,
        "synthetic_rgb_encoded_from_same_candidate_render": shared_candidate_render_encoded,
        "interpretation": "plumbing_identity_ceiling_not_photo_model_generalization",
    }


def _photo_verifier_config() -> PhotoGeometryVerifierConfig:
    return PhotoGeometryVerifierConfig(
        subsample=1,
        min_depth_m=10.0,
        outline_cap_source_px=5.0,
        typed_min_component_px=1,
        min_candidate_outline_px=1,
        skyline_exclusion_source_px=0,
        ordinal_grid_columns=8,
        ordinal_grid_rows=4,
        ordinal_min_samples_per_cell=4,
        ordinal_min_cells=8,
        ordinal_min_photo_span=0.10,
        min_common_depth_px=32,
        minimum_skyline_coverage=0.90,
        minimum_ridge_effective_samples=8.0,
        ridge_horizontal_bins=4,
        minimum_ridge_horizontal_bins=2,
        skyline_weight=0.0,
        outline_weight=0.20,
        ordinal_depth_weight=0.70,
        terrain_overlap_weight=0.10,
        beam_size=8,
        nms_position_m=100.0,
        nms_yaw_deg=20.0,
        rival_position_m=100.0,
        rival_yaw_deg=5.0,
        maximum_selected_score=0.25,
        minimum_rival_margin=0.02,
        stability_folds=4,
        minimum_stable_folds=3,
    )


def _ambiguous_photo_atlas(
    terrain: TerrainMap,
    truth: CameraExtrinsics,
) -> tuple[np.ndarray, SkylineAtlasArchive]:
    config = SkylineAtlasConfig(
        radius_m=200.0,
        spacing_m=200.0,
        yaw_step_deg=90.0,
        yaw_modes_per_position=4,
        yaw_mode_separation_deg=90.0,
        max_observed_columns=WIDTH_PX,
        ray_step_m=40.0,
    )
    physical_skyline = _skyline(terrain, truth, ray_step_m=config.ray_step_m)
    ambiguous_skyline = np.full_like(physical_skyline, np.nanmedian(physical_skyline))
    atlas = build_skyline_atlas(
        terrain,
        ambiguous_skyline,
        HEIGHT_PX,
        FOV_DEG,
        LocalPoint(east_m=0.0, north_m=0.0, up_m=-999.0),
        config=config,
    )
    return ambiguous_skyline, atlas


def _encoded_shared_scene_photo(
    candidate_depth: np.ndarray,
    observed_skyline: np.ndarray,
) -> np.ndarray:
    """Encode exact shared-scene cues in RGB for a plumbing upper bound."""

    finite = np.isfinite(candidate_depth) & (candidate_depth > 0.0)
    log_depth = np.zeros_like(candidate_depth)
    log_depth[finite] = np.log(candidate_depth[finite])
    low = float(log_depth[finite].min())
    high = float(log_depth[finite].max())
    relative_depth = np.divide(
        log_depth - low,
        high - low,
        out=np.zeros_like(log_depth),
        where=finite,
    )
    outlines = extract_typed_outlines(candidate_depth, min_px=1)
    internal_outline = outlines.occlusion | outlines.rib | outlines.couloir

    rgb = np.zeros((*candidate_depth.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.where(internal_outline, 255, 64).astype(np.uint8)
    rgb[..., 1] = np.rint(np.clip(relative_depth, 0.0, 1.0) * 255.0).astype(np.uint8)
    boundary_row = int(round(float(np.nanmedian(observed_skyline))))
    rgb[:boundary_row, :, :] = np.asarray([64, 0, 255], dtype=np.uint8)
    return rgb


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


def test_photo_geometry_verifier_scores_a_frozen_full_pool_and_selects_a_distinct_beam(
    asymmetric_terrain: TerrainMap,
) -> None:
    """Shared-scene cues test the identity ceiling, not photo-model generalization."""

    truth = _truth(asymmetric_terrain, 0.0, 0.0, 0.0)
    observed, atlas = _ambiguous_photo_atlas(asymmetric_terrain, truth)
    frozen_atlas_json = json.dumps(atlas.to_record(), sort_keys=True)
    target_candidates = [
        candidate
        for candidate in atlas.candidates
        if candidate.position.east_m == truth.position.east_m
        and candidate.position.north_m == truth.position.north_m
        and candidate.yaw_deg == truth.yaw_deg
    ]
    assert len(target_candidates) == 1
    target = target_candidates[0]

    # The RGB channels below are generated from the same frozen candidate,
    # terrain, and raycaster that the verifier uses. This is an authoritative
    # shared-scene plumbing ceiling, not independent learned-photo evidence.
    rendered_target = render_cyltan_candidate_depth(
        asymmetric_terrain,
        None,
        target.to_record(),
        WIDTH_PX,
        HEIGHT_PX,
        FOV_DEG,
        subsample=1,
    )
    photo_rgb = _encoded_shared_scene_photo(rendered_target.candidate_ray_depth, observed)
    config = _photo_verifier_config()
    evidence = extract_photo_geometry_evidence(
        photo_rgb,
        observed,
        _EncodedSyntheticEdges(),
        _EncodedSyntheticDepth(),
        observation_provenance=_photo_observation_provenance(
            atlas,
            shared_candidate_render_encoded=True,
        ),
        edge_model_provenance=_photo_model_provenance(_EncodedSyntheticEdges.name, "a"),
        depth_model_provenance=_photo_model_provenance(_EncodedSyntheticDepth.name, "b"),
        config=config,
    )
    assert evidence.usable is True

    archive = build_photo_geometry_verifier(
        asymmetric_terrain,
        None,
        evidence,
        atlas.to_record(),
        config=config,
    )
    frozen_verifier_json = json.dumps(archive.to_record(), sort_keys=True)

    # Numeric reference evaluation happens only after both estimator archives
    # have been serialized and frozen.
    atlas_evaluation = evaluate_skyline_atlas(atlas, truth, top_ks=(1, len(atlas.candidates)))
    evaluation = evaluate_photo_geometry_verifier(
        archive,
        truth,
        top_ks=(1, config.beam_size, len(archive.candidates)),
    )

    assert json.dumps(atlas.to_record(), sort_keys=True) == frozen_atlas_json
    assert json.dumps(archive.to_record(), sort_keys=True) == frozen_verifier_json
    assert atlas_evaluation.winner_errors.errors.horizontal_position_m == 200.0
    assert atlas_evaluation.shortlist_gt_oracle.candidate_id == target.candidate_id
    assert target.estimator_rank > 1

    assert len(archive.candidates) == len(atlas.candidates) == 36
    assert {candidate.candidate_id for candidate in archive.candidates} == {
        candidate.candidate_id for candidate in atlas.candidates
    }
    assert [candidate.verifier_rank for candidate in archive.candidates] == list(range(1, 37))
    assert len(archive.beam_candidate_ids) == config.beam_size < len(archive.candidates)
    assert len(set(archive.beam_candidate_ids)) == config.beam_size

    assert archive.component_winners["skyline"] == atlas.selected.candidate_id
    assert archive.component_winners["symmetric_outline"] == target.candidate_id
    assert archive.component_winners["ordinal_depth"] == target.candidate_id
    assert archive.ranked_winner.candidate_id == target.candidate_id
    assert archive.ranked_winner.original_estimator_rank == target.estimator_rank
    assert archive.ranked_winner.beam_rank == 1
    assert archive.decision.status == "selected"
    assert archive.decision.returned_candidate_id == target.candidate_id
    assert evaluation.ranked_winner_errors.errors.horizontal_position_m == 0.0
    assert evaluation.ranked_winner_errors.errors.yaw_deg == 0.0
    assert evaluation.first_target_verifier_rank == 1
    assert evaluation.first_target_beam_rank == 1
    assert evaluation.returned_candidate_errors is not None
    assert evaluation.returned_candidate_errors.reaches_target is True

    frozen_record = archive.to_record()
    assert frozen_record["candidate_pool"] == "complete_frozen_photo_atlas_pool"
    assert frozen_record["numeric_evaluation_reference_used"] is False
    assert frozen_record["evidence"]["observation"]["synthetic_rgb_encoded_from_same_candidate_render"] is True
    assert frozen_record["evidence"]["observation"]["interpretation"] == (
        "plumbing_identity_ceiling_not_photo_model_generalization"
    )
    assert "candidate_pool_gt_oracle" not in frozen_verifier_json
    assert "horizontal_position_m" not in frozen_verifier_json


def test_photo_geometry_verifier_abstains_on_empty_synthetic_photo_cues(
    asymmetric_terrain: TerrainMap,
) -> None:
    truth = _truth(asymmetric_terrain, 0.0, 0.0, 0.0)
    observed, atlas = _ambiguous_photo_atlas(asymmetric_terrain, truth)
    config = _photo_verifier_config()
    empty_rgb = np.full((HEIGHT_PX, WIDTH_PX, 3), 64, dtype=np.uint8)
    evidence = extract_photo_geometry_evidence(
        empty_rgb,
        observed,
        _EncodedSyntheticEdges(),
        _EncodedSyntheticDepth(),
        observation_provenance=_photo_observation_provenance(
            atlas,
            shared_candidate_render_encoded=False,
        ),
        edge_model_provenance=_photo_model_provenance(_EncodedSyntheticEdges.name, "c"),
        depth_model_provenance=_photo_model_provenance(_EncodedSyntheticDepth.name, "d"),
        config=config,
    )

    assert evidence.usable is False
    assert "insufficient_internal_ridge_support" in evidence.rejection_reasons
    assert "insufficient_relative_depth_span" in evidence.rejection_reasons

    archive = build_photo_geometry_verifier(
        asymmetric_terrain,
        None,
        evidence,
        atlas.to_record(),
        config=config,
    )

    assert len(archive.candidates) == len(atlas.candidates)
    assert archive.decision.status == "abstained"
    assert archive.decision.returned_candidate_id is None
    assert "insufficient_internal_ridge_support" in archive.decision.reasons
    assert "insufficient_relative_depth_span" in archive.decision.reasons
