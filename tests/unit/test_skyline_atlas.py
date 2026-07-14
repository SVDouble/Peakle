from __future__ import annotations

import inspect
import json
import math

import numpy as np
import pytest

from peakle.domain.camera import CameraExtrinsics
from peakle.domain.coordinates import GeoPoint, LocalPoint
from peakle.domain.terrain import TerrainMap, TerrainSpec
from peakle.localize import skyline_atlas as skyline_atlas_module
from peakle.localize.skyline_atlas import (
    SkylineAtlasArchive,
    SkylineAtlasCandidate,
    SkylineAtlasConfig,
    build_skyline_atlas,
    evaluate_skyline_atlas,
)
from peakle.localize.solve import HorizonProfile
from peakle.localize.swissdem import Patch


def _terrain() -> TerrainMap:
    size = 32
    x_m = np.linspace(-300.0, 300.0, size)
    y_m = np.linspace(-300.0, 300.0, size)
    x_grid, y_grid = np.meshgrid(x_m, y_m)
    elevation = np.full((size, size), 100.0)
    return TerrainMap(
        spec=TerrainSpec(
            origin=GeoPoint(latitude_deg=46.0, longitude_deg=8.0, elevation_m=100.0),
            width_m=600.0,
            height_m=600.0,
            grid_width=size,
            grid_height=size,
            min_elevation_m=100.0,
            max_elevation_m=101.0,
            seed=0,
        ),
        x_m=x_m,
        y_m=y_m,
        elevation_m=elevation,
        latitude_deg=46.0 + y_grid / 111_320.0,
        longitude_deg=8.0 + x_grid / 80_000.0,
    )


def _observed() -> np.ndarray:
    return np.asarray([6.0, 6.2, 6.1, np.nan, 6.0, 5.9, 6.1, 6.0, 6.2, 6.0, 5.8, 6.0])


def _small_config() -> SkylineAtlasConfig:
    return SkylineAtlasConfig(
        radius_m=150.0,
        spacing_m=150.0,
        yaw_step_deg=90.0,
        max_observed_columns=12,
        ray_step_m=100.0,
    )


@pytest.fixture(scope="module")
def atlas() -> SkylineAtlasArchive:
    return build_skyline_atlas(
        _terrain(),
        _observed(),
        12,
        60.0,
        LocalPoint(east_m=0.0, north_m=0.0, up_m=999.0),
        config=_small_config(),
    )


def _candidate_far_from_winner(archive: SkylineAtlasArchive) -> SkylineAtlasCandidate:
    winner = archive.selected
    return next(
        candidate
        for candidate in reversed(archive.candidates)
        if math.hypot(candidate.east_m - winner.east_m, candidate.north_m - winner.north_m) > 100.0
    )


def _truth_at(candidate: SkylineAtlasCandidate, *, pitch_deg: float | None = None) -> CameraExtrinsics:
    return CameraExtrinsics(
        position=candidate.position,
        yaw_deg=candidate.yaw_deg,
        pitch_deg=candidate.pitch_deg if pitch_deg is None else pitch_deg,
        roll_deg=0.0,
    )


def test_config_defaults_are_frozen_and_validated() -> None:
    config = SkylineAtlasConfig()

    assert config.radius_m == 500.0
    assert config.spacing_m == 50.0
    assert config.yaw_step_deg == 1.0
    assert config.yaw_modes_per_position == 3
    assert config.yaw_mode_separation_deg == 8.0
    assert config.max_observed_columns == 192
    assert config.residual_cap_px == 60.0
    assert config.high_outlier_weight == 0.02
    assert config.max_abs_roll_nuisance_deg == 10.0
    assert config.eye_height_m == 2.5
    assert config.ray_step_m == 10.0
    with pytest.raises((AttributeError, TypeError)):
        config.radius_m = 1.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="divide 360"):
        SkylineAtlasConfig(yaw_step_deg=7.0)


def test_default_archive_retains_three_separated_modes_per_grid_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_builds = 0

    class FlatHorizonProfile:
        az_deg = np.arange(0.0, 360.0, 0.1)
        el = np.zeros(3_600, dtype=np.float64)

        def __init__(self, *_args, **_kwargs) -> None:
            nonlocal profile_builds
            profile_builds += 1

    monkeypatch.setattr(skyline_atlas_module, "HorizonProfile", FlatHorizonProfile)

    archive = build_skyline_atlas(
        _terrain(),
        _observed(),
        12,
        60.0,
        LocalPoint(east_m=0.0, north_m=0.0, up_m=999.0),
    )

    assert archive.grid_side == 21
    assert profile_builds == 21 * 21
    assert len(archive.candidates) == 21 * 21 * 3 == 1_323
    by_position: dict[tuple[int, int], list[float]] = {}
    for candidate in archive.candidates:
        by_position.setdefault((candidate.grid_row, candidate.grid_column), []).append(candidate.yaw_deg)
    assert {len(yaws) for yaws in by_position.values()} == {3}
    for yaws in by_position.values():
        for first_index, first in enumerate(yaws):
            for second in yaws[first_index + 1 :]:
                separation = abs((first - second + 180.0) % 360.0 - 180.0)
                assert separation >= 8.0


def test_archive_is_deterministic_canonical_and_contains_no_evaluation_truth(atlas: SkylineAtlasArchive) -> None:
    repeated = build_skyline_atlas(
        _terrain(),
        _observed(),
        12,
        60.0,
        LocalPoint(east_m=0.0, north_m=0.0, up_m=999.0),
        config=_small_config(),
    )

    assert atlas.archive_sha256 == repeated.archive_sha256
    assert [candidate.candidate_id for candidate in atlas.candidates] == [
        candidate.candidate_id for candidate in repeated.candidates
    ]
    assert atlas.to_record() == repeated.to_record()
    assert list(inspect.signature(build_skyline_atlas).parameters) == [
        "terrain",
        "observed_skyline",
        "image_height_px",
        "horizontal_fov_deg",
        "prior_position",
        "native_patch",
        "config",
    ]
    encoded = json.dumps(atlas.to_record(), sort_keys=True)
    assert '"numeric_evaluation_reference_used": false' in encoded
    assert "full_lattice_gt_oracle" not in encoded
    assert "horizontal_position_m" not in encoded


def test_evaluation_oracle_can_differ_from_the_truth_free_winner(atlas: SkylineAtlasArchive) -> None:
    target = _candidate_far_from_winner(atlas)

    evaluation = evaluate_skyline_atlas(atlas, _truth_at(target))

    assert evaluation.winner_errors.candidate_id == atlas.selected.candidate_id
    assert evaluation.shortlist_gt_oracle.candidate_id == target.candidate_id
    assert evaluation.shortlist_gt_oracle.normalized_joint_error == 0.0
    assert evaluation.full_lattice_gt_oracle.normalized_joint_error == 0.0
    assert evaluation.winner_errors.candidate_id != evaluation.shortlist_gt_oracle.candidate_id
    assert evaluation.selection_regret.normalized_joint_error > 0.0


def test_full_lattice_oracle_includes_yaws_pruned_from_shortlist(monkeypatch: pytest.MonkeyPatch) -> None:
    class FlatHorizonProfile:
        az_deg = np.arange(0.0, 360.0, 0.1)
        el = np.zeros(3_600, dtype=np.float64)

        def __init__(self, *_args, **_kwargs) -> None:
            pass

    monkeypatch.setattr(skyline_atlas_module, "HorizonProfile", FlatHorizonProfile)
    archive = build_skyline_atlas(
        _terrain(),
        np.full(12, 6.0),
        12,
        60.0,
        LocalPoint(east_m=0.0, north_m=0.0, up_m=0.0),
        config=SkylineAtlasConfig(
            radius_m=0.0,
            yaw_step_deg=90.0,
            yaw_modes_per_position=1,
            ray_step_m=100.0,
        ),
    )
    lattice_position = archive.lattice_positions[0]
    truth = CameraExtrinsics(
        position=LocalPoint(
            east_m=lattice_position.east_m,
            north_m=lattice_position.north_m,
            up_m=lattice_position.up_m,
        ),
        yaw_deg=90.0,
        pitch_deg=0.0,
        roll_deg=0.0,
    )

    evaluation = evaluate_skyline_atlas(archive, truth)

    assert len(archive.candidates) == 1
    assert archive.to_record()["full_score_lattice"]["hypothesis_count"] == 4
    assert evaluation.shortlist_gt_oracle.errors.yaw_deg == 90.0
    assert evaluation.full_lattice_gt_oracle.errors.horizontal_position_m == 0.0
    assert evaluation.full_lattice_gt_oracle.errors.yaw_deg == 0.0
    assert evaluation.full_lattice_gt_oracle.estimator_rank_scope == "full_score_lattice"


def test_oracle_ignores_pitch(atlas: SkylineAtlasArchive) -> None:
    target = _candidate_far_from_winner(atlas)

    low_pitch = evaluate_skyline_atlas(atlas, _truth_at(target, pitch_deg=-70.0))
    high_pitch = evaluate_skyline_atlas(atlas, _truth_at(target, pitch_deg=70.0))

    assert low_pitch.shortlist_gt_oracle.candidate_id == target.candidate_id
    assert high_pitch.shortlist_gt_oracle.candidate_id == target.candidate_id
    assert low_pitch.shortlist_gt_oracle.normalized_joint_error == high_pitch.shortlist_gt_oracle.normalized_joint_error
    assert low_pitch.winner_errors.errors.pitch_deg is None
    assert low_pitch.shortlist_gt_oracle.errors.pitch_deg is None
    assert low_pitch.to_record()["winner_errors"]["errors"]["pitch_deg"] is None
    assert low_pitch.to_record()["target"]["pitch_used_by_oracle"] is False


def test_camera_height_uses_finite_native_patch_and_falls_back_on_nodata() -> None:
    terrain = _terrain()
    prior = LocalPoint(east_m=0.0, north_m=0.0, up_m=-500.0)
    finite_patch = Patch(
        x_m=np.asarray([-100.0, 100.0]),
        y_m=np.asarray([-100.0, 100.0]),
        elevation_m=np.full((2, 2), 450.0),
    )
    nodata_patch = Patch(
        x_m=finite_patch.x_m,
        y_m=finite_patch.y_m,
        elevation_m=np.full((2, 2), np.nan),
    )
    config = SkylineAtlasConfig(radius_m=0.0, yaw_step_deg=180.0, ray_step_m=100.0)

    native = build_skyline_atlas(
        terrain,
        _observed(),
        12,
        60.0,
        prior,
        native_patch=finite_patch,
        config=config,
    )
    fallback = build_skyline_atlas(
        terrain,
        _observed(),
        12,
        60.0,
        prior,
        native_patch=nodata_patch,
        config=config,
    )

    assert {candidate.up_m for candidate in native.candidates} == {452.5}
    assert {candidate.ground_source for candidate in native.candidates} == {"native_patch"}
    assert {candidate.up_m for candidate in fallback.candidates} == {102.5}
    assert {candidate.ground_source for candidate in fallback.candidates} == {"terrain_map"}


def test_top_k_recall_uses_estimator_rank_without_gt_resorting(atlas: SkylineAtlasArchive) -> None:
    target = _candidate_far_from_winner(atlas)
    assert target.estimator_rank > 1

    evaluation = evaluate_skyline_atlas(
        atlas,
        _truth_at(target),
        top_ks=(target.estimator_rank - 1, target.estimator_rank),
    )

    assert evaluation.shortlist_top_k_reach[target.estimator_rank - 1] is False
    assert evaluation.shortlist_top_k_recall[target.estimator_rank - 1] == 0.0
    assert evaluation.shortlist_top_k_reach[target.estimator_rank] is True
    assert evaluation.shortlist_top_k_recall[target.estimator_rank] == 1.0


def test_nonflat_synthetic_skyline_recovers_signed_position_and_wrapped_yaw() -> None:
    size = 81
    x_m = np.linspace(-2_000.0, 2_000.0, size)
    y_m = np.linspace(-2_000.0, 2_000.0, size)
    x_grid, y_grid = np.meshgrid(x_m, y_m)
    elevation = np.full_like(x_grid, 100.0)
    for east_m, north_m, height_m, sigma_m in (
        (900.0, 650.0, 850.0, 220.0),
        (-1_100.0, 750.0, 620.0, 320.0),
        (-450.0, -1_150.0, 720.0, 260.0),
        (1_250.0, -650.0, 470.0, 180.0),
    ):
        distance_squared = (x_grid - east_m) ** 2 + (y_grid - north_m) ** 2
        elevation += height_m * np.exp(-distance_squared / (2.0 * sigma_m**2))
    terrain = TerrainMap(
        spec=TerrainSpec(
            origin=GeoPoint(latitude_deg=46.0, longitude_deg=8.0, elevation_m=100.0),
            width_m=4_000.0,
            height_m=4_000.0,
            grid_width=size,
            grid_height=size,
            min_elevation_m=100.0,
            max_elevation_m=1_000.0,
            seed=0,
        ),
        x_m=x_m,
        y_m=y_m,
        elevation_m=elevation,
        latitude_deg=46.0 + y_grid / 111_320.0,
        longitude_deg=8.0 + x_grid / 80_000.0,
    )
    config = SkylineAtlasConfig(
        radius_m=100.0,
        spacing_m=100.0,
        yaw_step_deg=10.0,
        max_observed_columns=120,
        ray_step_m=20.0,
    )
    truth_position = LocalPoint(
        east_m=100.0,
        north_m=-100.0,
        up_m=terrain.elevation_at(100.0, -100.0) + config.eye_height_m,
    )
    truth_yaw_deg = -190.0
    observed = HorizonProfile(
        terrain,
        truth_position.up_m,
        step=config.ray_step_m,
        cam_e=truth_position.east_m,
        cam_n=truth_position.north_m,
    ).rows_cyl_tan(120, 80, 80.0, truth_yaw_deg)
    observed += math.tan(math.radians(5.0)) * (np.arange(120) - 59.5)

    archive = build_skyline_atlas(
        terrain,
        observed,
        80,
        80.0,
        LocalPoint(east_m=0.0, north_m=0.0, up_m=-999.0),
        config=config,
    )
    evaluation = evaluate_skyline_atlas(
        archive,
        CameraExtrinsics(
            position=truth_position,
            yaw_deg=truth_yaw_deg,
            pitch_deg=0.0,
            roll_deg=0.0,
        ),
    )

    assert archive.selected.position.as_tuple() == pytest.approx(truth_position.as_tuple(), abs=1e-9)
    assert archive.selected.yaw_deg == 170.0
    assert archive.selected.roll_nuisance_deg == pytest.approx(5.0, abs=1e-9)
    assert archive.selected.score == 0.0
    assert evaluation.winner_errors.errors.horizontal_position_m == 0.0
    assert evaluation.winner_errors.errors.yaw_deg == 0.0
    assert evaluation.full_lattice_gt_oracle.candidate_id == archive.selected.candidate_id
