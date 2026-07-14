from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.coordinates import GeoPoint, LocalPoint
from peakle.domain.pose import PosePrior
from peakle.domain.terrain import TerrainMap, TerrainSpec
from peakle.localize.geopose import GeoPoseOriginalMetadata, GeoPoseSample
from peakle.localize.render_match_pnp import CandidateValidationConfig, RenderMatchConfig, RenderMatchPoseResult
from peakle.localize.strategy_bench import (
    ALGORITHMS,
    DEFAULT_ALGORITHMS,
    EvidenceTrack,
    MatrixConfig,
    RenderMatchResources,
    TerrainSelection,
    aggregate_matrix,
    build_matrix_cells,
    build_prior_scenario,
    commit_artifact,
    config_record,
    deterministic_map_center_offset,
    fuse_high_resolution_patch,
    run_sample_matrix,
)
from peakle.localize.swissdem import Patch


def _terrain() -> TerrainMap:
    size = 64
    x_m = np.linspace(-5_000.0, 5_000.0, size)
    y_m = np.linspace(-5_000.0, 5_000.0, size)
    x_grid, y_grid = np.meshgrid(x_m, y_m)
    elevation = np.full((size, size), 1_000.0)
    spec = TerrainSpec(
        origin=GeoPoint(latitude_deg=46.0, longitude_deg=8.0, elevation_m=1_000.0),
        width_m=10_000.0,
        height_m=10_000.0,
        grid_width=size,
        grid_height=size,
        min_elevation_m=1_000.0,
        max_elevation_m=1_001.0,
        seed=0,
    )
    return TerrainMap(
        spec=spec,
        x_m=x_m,
        y_m=y_m,
        elevation_m=elevation,
        latitude_deg=46.0 + y_grid / 111_320.0,
        longitude_deg=8.0 + x_grid / 80_000.0,
    )


def _sample(*, manual: bool = True) -> GeoPoseSample:
    return GeoPoseSample(
        name="sample-a",
        root=Path("/unused"),
        manual=manual,
        lat=46.0,
        lon=8.0,
        elev_m=1_500.0,
        fov_deg=60.0,
        yaw_gt_deg=137.0,
        pitch_gt_deg=-4.0,
        roll_gt_deg=0.0,
    )


def _truth() -> CameraExtrinsics:
    return CameraExtrinsics(
        position=LocalPoint(east_m=1_000.0, north_m=-500.0, up_m=1_500.0),
        yaw_deg=137.0,
        pitch_deg=-4.0,
        roll_deg=0.0,
    )


def test_prior_regimes_are_deterministic_and_do_not_leak_missing_fields() -> None:
    terrain = _terrain()
    sample = _sample()
    truth = _truth()
    args = (sample, terrain, truth, "standard", 0, 77)

    raw = build_prior_scenario(args[0], args[1], args[2], "raw_metadata", *args[3:])
    perturbed = build_prior_scenario(args[0], args[1], args[2], "perturbed_metadata", *args[3:])
    repeated = build_prior_scenario(args[0], args[1], args[2], "perturbed_metadata", *args[3:])
    position_only = build_prior_scenario(args[0], args[1], args[2], "position_only", *args[3:])
    no_prior = build_prior_scenario(args[0], args[1], args[2], "none", *args[3:])

    assert raw.contains_exact_reference is True
    assert raw.prior.position == truth.position
    assert raw.prior.yaw_deg == truth.yaw_deg

    assert perturbed == repeated
    assert perturbed.contains_exact_reference is False
    assert math.hypot(
        perturbed.prior.position.east_m - truth.position.east_m,
        perturbed.prior.position.north_m - truth.position.north_m,
    ) == pytest.approx(200.0)
    assert abs((perturbed.prior.yaw_deg - truth.yaw_deg + 180.0) % 360.0 - 180.0) == pytest.approx(15.0)

    assert position_only.prior.position == perturbed.prior.position
    assert position_only.prior.yaw_deg == 0.0
    assert position_only.prior.pitch_deg == 0.0
    assert position_only.use_position_prior is True
    assert position_only.use_orientation_prior is False

    assert no_prior.prior.position.east_m == 0.0
    assert no_prior.prior.position.north_m == 0.0
    assert no_prior.prior.position != truth.position
    assert no_prior.use_position_prior is False
    assert no_prior.use_orientation_prior is False
    assert no_prior.constructed_from_reference is False


def test_default_matrix_only_schedules_evidence_backed_baselines() -> None:
    assert DEFAULT_ALGORITHMS == ("keep-prior", "horizon")
    assert MatrixConfig(terrain_grid=128).algorithms == DEFAULT_ALGORITHMS
    assert "cmaes" in ALGORITHMS  # explicit historical replay remains available


def test_map_center_is_stably_offset_from_reference() -> None:
    config = MatrixConfig(extent_m=20_000.0, terrain_grid=128, root_seed=11)
    first = deterministic_map_center_offset("sample-a", config)
    second = deterministic_map_center_offset("sample-a", config)

    assert first == second
    assert math.hypot(*first) == pytest.approx(3_200.0)
    assert first != deterministic_map_center_offset("sample-b", config)


def test_core_matrix_marks_every_omitted_or_inapplicable_cell() -> None:
    config = MatrixConfig(
        algorithms=ALGORITHMS,
        evidence_tracks=("pfm_oracle",),
        prior_regimes=("raw_metadata", "position_only", "none"),
        terrain_grid=128,
    )
    cells = {(cell.algorithm, cell.prior_regime): cell for cell in build_matrix_cells(config)}

    assert cells[("horizon", "position_only")].applicable is True
    assert cells[("horizon", "raw_metadata")].applicable is False
    assert cells[("horizon", "raw_metadata")].skip_reason == "excluded_by_core_profile"
    assert cells[("global", "none")].applicable is True
    assert cells[("global", "raw_metadata")].applicable is False
    assert cells[("keep-prior", "none")].skip_reason == "algorithm_requires_a_different_prior_regime"


def test_render_pnp_matrix_uses_only_rgb_and_requires_a_position_seed() -> None:
    config = MatrixConfig(
        algorithms=("keep-prior", "render-pnp"),
        evidence_tracks=("pfm_oracle", "photo_auto", "photo_rgb"),
        prior_regimes=("perturbed_metadata", "position_only", "none"),
        terrain_grid=128,
        render_matcher="sift",
    )
    cells = {(cell.algorithm, cell.prior_regime, cell.evidence_track): cell for cell in build_matrix_cells(config)}

    assert cells[("render-pnp", "perturbed_metadata", "photo_rgb")].applicable is True
    assert cells[("render-pnp", "position_only", "photo_rgb")].applicable is True
    assert cells[("render-pnp", "perturbed_metadata", "pfm_oracle")].applicable is False
    assert (
        cells[("render-pnp", "perturbed_metadata", "pfm_oracle")].skip_reason
        == "algorithm_requires_a_different_evidence_track"
    )
    assert cells[("render-pnp", "none", "photo_rgb")].applicable is False
    assert cells[("render-pnp", "none", "photo_rgb")].skip_reason == "algorithm_requires_a_different_prior_regime"
    assert cells[("keep-prior", "perturbed_metadata", "photo_rgb")].applicable is True


def test_matcher_cache_is_explicit_worker_only_and_persisted_by_cli() -> None:
    from peakle.localize.paths import BASE
    from peakle.scripts.bench_pose_matrix import _matcher_cache_provenance, _parser

    default = MatrixConfig(terrain_grid=128)
    assert default.matcher_cache_dir is None
    assert config_record(default)["matcher_cache_dir"] is None
    assert _parser().parse_args([]).algorithms == ",".join(DEFAULT_ALGORITHMS)
    with pytest.raises(ValueError, match="only for the external worker"):
        MatrixConfig(render_matcher="sift", matcher_cache_dir="local/cache", terrain_grid=128).validate()

    args = _parser().parse_args(["--matcher-cache", "local/cache/correspondences"])
    assert args.matcher_cache == "local/cache/correspondences"
    configured = MatrixConfig(
        render_matcher="worker",
        matcher_command=("python", "/tmp/worker.py"),
        matcher_manifest_path="local/models/manifest.json",
        matcher_cache_dir=args.matcher_cache,
        terrain_grid=128,
    )
    configured.validate()
    provenance = _matcher_cache_provenance(configured)
    assert provenance["enabled"] is True
    assert provenance["configured_path"] == "local/cache/correspondences"
    assert provenance["resolved_path"] == str((BASE / "local/cache/correspondences").resolve())
    assert provenance["mode"] == "read_through_atomic_content_addressed"


def test_candidate_validation_is_enabled_serialized_and_cli_opt_out_reaches_render_config() -> None:
    from peakle.localize import strategy_bench as module
    from peakle.scripts.bench_pose_matrix import _parser

    default = MatrixConfig(render_matcher="sift", terrain_grid=128)
    record = config_record(default)["render_candidate_validation"]
    assert record == {
        "enabled": True,
        "query_grid_columns": 8,
        "query_grid_rows": 6,
        "folds": 4,
        "max_holdout_matches_per_frame": 400,
        "confidence_level": 0.95,
        "minimum_testable_fraction": 0.5,
        "minimum_visibility_consistency_fraction": 0.8,
        "render_resolution_multiplier": 2,
        "maximum_local_absolute_depth_span_m": 250.0,
        "maximum_local_relative_depth_span": 0.08,
        "minimum_depth_tolerance_m": 1.0,
        "maximum_depth_tolerance_m": 3.0,
        "relative_depth_tolerance": 1e-4,
        "minimum_conditional_visibility_trials": 14,
    }
    assert _parser().parse_args([]).disable_candidate_validation is False

    args = _parser().parse_args(["--disable-candidate-validation"])
    configured = MatrixConfig(
        render_matcher="sift",
        terrain_grid=128,
        render_candidate_validation=CandidateValidationConfig(enabled=not args.disable_candidate_validation),
    )
    resources = module._render_match_resources(_terrain(), configured)

    assert resources.config.candidate_validation.enabled is False


def test_explicit_benchmark_sample_list_is_not_silently_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    from peakle.scripts.bench_pose_matrix import _parser, _selected_samples

    samples = [Path("/data/alpha"), Path("/data/beta"), Path("/data/gamma")]
    monkeypatch.setattr("peakle.scripts.bench_pose_matrix.find_sample_dirs", lambda: samples)

    explicit = _parser().parse_args(["--samples", "alpha,beta,gamma"])
    assert _selected_samples(explicit) == samples

    capped = _parser().parse_args(["--samples", "alpha,beta,gamma", "--max-n", "2"])
    assert _selected_samples(capped) == samples[:2]

    default = _parser().parse_args(["--manifest", "/definitely/missing/manifest.txt"])
    assert _selected_samples(default) == samples[:1]


def test_rgb_only_evidence_does_not_invoke_skyline_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    from peakle.localize import strategy_bench as module

    sample = _sample()
    rgb = np.zeros((60, 80, 3), dtype=np.uint8)
    monkeypatch.setattr(module, "resampled_oracle_skyline", lambda *_args: np.full(80, 40.0))

    def unexpected_extraction(*_args, **_kwargs):
        raise AssertionError("photo_rgb must not run an unrelated photo skyline extractor")

    monkeypatch.setattr(module, "extract_candidates", unexpected_extraction)

    tracks = module._evidence_tracks(
        sample,
        rgb,
        MatrixConfig(evidence_tracks=("photo_rgb",), terrain_grid=128),
    )

    assert [track.name for track in tracks] == ["photo_rgb"]
    assert tracks[0].metadata["source_depth_pfm_used_by_estimator"] is False


def test_high_resolution_source_is_fused_only_where_it_covers() -> None:
    terrain = _terrain()
    patch = Patch(
        x_m=np.linspace(-1_000.0, 1_000.0, 21),
        y_m=np.linspace(-1_000.0, 1_000.0, 21),
        elevation_m=np.full((21, 21), 1_234.0),
    )

    fused, count = fuse_high_resolution_patch(terrain, patch)

    assert count > 0
    assert fused.elevation_at(0.0, 0.0) == pytest.approx(1_234.0)
    assert fused.elevation_at(4_000.0, 4_000.0) == pytest.approx(1_000.0)
    assert terrain.elevation_at(0.0, 0.0) == pytest.approx(1_000.0)


def test_reference_centered_patch_is_evaluation_only_and_never_fused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from peakle.localize import strategy_bench as module

    base = _terrain()
    evaluation_patch = Patch(
        x_m=np.asarray([-500.0, 500.0]),
        y_m=np.asarray([-500.0, 500.0]),
        elevation_m=np.full((2, 2), 9_999.0),
    )
    monkeypatch.setattr(module, "load_copernicus_terrain", lambda *_args, **_kwargs: base)

    def fake_cached_patch(_terrain, _center, **kwargs):
        assert kwargs == {
            "center_source": "evaluation_reference_pose",
            "uses_reference_truth": True,
            "used_by_estimator": False,
        }
        return module.TerrainPatchProvision(
            evaluation_patch,
            {
                "center_source": "evaluation_reference_pose",
                "uses_reference_truth": True,
                "used_by_estimator": False,
                "coverage": {"available": True},
            },
        )

    monkeypatch.setattr(module, "_cached_patch_at_local_position", fake_cached_patch)
    selected = module.load_benchmark_terrain(
        _sample(),
        MatrixConfig(extent_m=20_000.0, terrain_grid=128),
    )

    assert selected.terrain is base
    assert selected.terrain.elevation_at(0.0, 0.0) == pytest.approx(1_000.0)
    assert selected.evaluation_high_resolution_patch is evaluation_patch
    assert selected.provenance["estimator_search_grid_reference_patch_fused"] is False
    assert selected.provenance["evaluation_reference_patch"]["used_by_estimator"] is False


def test_no_prior_estimator_never_loads_or_receives_reference_centered_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from peakle.localize import strategy_bench as module

    terrain = _terrain()
    scenario = build_prior_scenario(_sample(), terrain, _truth(), "none", "standard", 0, 77)

    def unexpected_patch_load(*_args, **_kwargs):
        raise AssertionError("a no-prior cell must not choose a native-patch centre")

    monkeypatch.setattr(module, "_cached_patch_at_local_position", unexpected_patch_load)
    selected = module._estimator_terrain_for_cell(
        terrain,
        scenario,
        scenario_key=("none", 0),
        patch_cache={},
        provision_patch=True,
    )

    assert selected.terrain is terrain
    assert selected.high_resolution_patch is None
    assert selected.provenance["uses_reference_truth"] is False
    patch_record = selected.provenance["native_patch"]
    assert patch_record["center_source"] == "none_no_position_prior"
    assert patch_record["uses_reference_truth"] is False
    assert patch_record["coverage"]["available"] is False


def test_sample_schema_separates_retention_and_controlled_baselines(monkeypatch: pytest.MonkeyPatch) -> None:
    from peakle.localize import strategy_bench as module

    sample = _sample()
    terrain = _terrain()
    selection = TerrainSelection(
        terrain=terrain,
        evaluation_high_resolution_patch=None,
        truth=_truth(),
        provenance={"far_source": "test"},
    )
    rows = np.linspace(20.0, 45.0, 80)
    monkeypatch.setattr(module, "load_sample", lambda _path: sample)
    monkeypatch.setattr(module, "load_benchmark_terrain", lambda _sample, _config: selection)
    monkeypatch.setattr(module, "_load_photo", lambda _path: np.zeros((60, 80, 3), np.uint8))
    monkeypatch.setattr(
        module,
        "_evidence_tracks",
        lambda _sample, _rgb, _config: [
            EvidenceTrack("pfm_oracle", rows, {"available": True}),
            EvidenceTrack("photo_auto", rows + 1.0, {"available": True}),
        ],
    )
    monkeypatch.setattr(
        module,
        "_pre_solve_quality",
        lambda *_args: ({"policy": "gt_dem_compat_v1", "tier": "MAP_A"}, {"usable": True}),
    )
    config = MatrixConfig(
        algorithms=("keep-prior",),
        evidence_tracks=("pfm_oracle", "photo_auto"),
        prior_regimes=("raw_metadata", "perturbed_metadata", "position_only", "none"),
        terrain_grid=128,
    )

    sample_row, cases = run_sample_matrix(Path("/unused"), config)

    assert len(cases) == 8
    raw = next(case for case in cases if case["prior_regime"] == "raw_metadata")
    perturbed = next(case for case in cases if case["prior_regime"] == "perturbed_metadata")
    position_only = next(case for case in cases if case["prior_regime"] == "position_only")
    no_prior = next(case for case in cases if case["prior_regime"] == "none")
    assert raw["errors"]["horizontal_position_m"] == 0.0
    assert raw["errors"]["yaw_deg"] == 0.0
    assert raw["errors"]["pitch_deg"] is None
    assert raw["ranking_eligible"] is False
    assert perturbed["errors"]["horizontal_position_m"] == pytest.approx(200.0)
    assert perturbed["errors"]["yaw_deg"] == pytest.approx(15.0)
    assert perturbed["ranking_eligible"] is True
    assert position_only["prior"]["yaw_deg"] == 0.0
    assert no_prior["status"] == "skipped"
    assert no_prior["skip_reason"] == "algorithm_requires_a_different_prior_regime"
    assert sample_row["gt_dem_compatibility"]["tier"] == "MAP_A"
    assert sample_row["matrix_case_ids"] == [case["id"] for case in cases]


def test_original_metadata_is_post_solve_nonranking_and_never_an_estimator_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from peakle.localize import strategy_bench as module

    sample = _sample()
    sample.original_metadata = GeoPoseOriginalMetadata(
        lat=45.0,
        lon=6.0,
        elev_m=25.0,
        fov_deg=20.0,
    )
    terrain = _terrain()
    truth = _truth()
    selection = TerrainSelection(
        terrain=terrain,
        evaluation_high_resolution_patch=None,
        truth=truth,
        provenance={"far_source": "test"},
    )
    rows = np.linspace(20.0, 45.0, 80)
    monkeypatch.setattr(module, "load_sample", lambda _path: sample)
    monkeypatch.setattr(module, "load_benchmark_terrain", lambda _sample, _config: selection)
    monkeypatch.setattr(module, "_load_photo", lambda _path: np.zeros((60, 80, 3), np.uint8))
    monkeypatch.setattr(
        module,
        "_evidence_tracks",
        lambda _sample, _rgb, _config: [EvidenceTrack("pfm_oracle", rows, {"available": True})],
    )
    monkeypatch.setattr(
        module,
        "_pre_solve_quality",
        lambda *_args: ({"policy": "gt_dem_compat_v1", "tier": "MAP_A"}, {"usable": True}),
    )
    patch_centers: list[LocalPoint] = []

    def no_cached_patch(_terrain, center, **kwargs):
        patch_centers.append(center)
        assert kwargs["center_source"] == "supplied_position_prior"
        assert kwargs["uses_reference_truth"] is False
        return module.TerrainPatchProvision(
            None,
            {
                "center_source": "supplied_position_prior",
                "uses_reference_truth": False,
                "used_by_estimator": False,
                "coverage": {"available": False},
            },
        )

    monkeypatch.setattr(module, "_cached_patch_at_local_position", no_cached_patch)
    estimator_inputs: dict[str, object] = {}

    def capture_estimator(**kwargs):
        estimator_inputs.update(kwargs)
        raise RuntimeError("captured estimator inputs")

    monkeypatch.setattr(module, "solve_pose", capture_estimator)
    config = MatrixConfig(
        algorithms=("keep-prior", "horizon"),
        evidence_tracks=("pfm_oracle",),
        prior_regimes=("position_only",),
        terrain_grid=128,
    )

    sample_row, cases = run_sample_matrix(Path("/unused"), config)

    baseline = next(case for case in cases if case["algorithm"] == "keep-prior")
    horizon = next(case for case in cases if case["algorithm"] == "horizon")
    original_local = terrain.frame.geo_to_local(GeoPoint(latitude_deg=45.0, longitude_deg=6.0, elevation_m=25.0))
    estimator_intrinsics = estimator_inputs["intrinsics"]
    estimator_prior = estimator_inputs["prior"]
    assert isinstance(estimator_intrinsics, CameraIntrinsics)
    assert isinstance(estimator_prior, PosePrior)
    assert estimator_inputs["truth"] is None
    assert estimator_inputs["horizontal_fov_deg"] == pytest.approx(60.0)
    assert estimator_intrinsics.horizontal_fov_deg() == pytest.approx(60.0)
    assert estimator_prior.position != original_local
    assert patch_centers == [estimator_prior.position]
    assert baseline["errors"]["horizontal_position_m"] == pytest.approx(200.0)
    assert baseline["reference"]["source"] == "refined_geopose_metadata_info_lines_3_6"
    assert horizon["status"] == "error"

    row_diagnostic = sample_row["original_metadata_diagnostic"]
    case_diagnostic = baseline["original_metadata_diagnostic"]
    assert row_diagnostic["available"] is True
    assert row_diagnostic["used_by_estimator"] is False
    assert row_diagnostic["used_for_success_grading"] is False
    assert row_diagnostic["used_for_ranking"] is False
    assert row_diagnostic["estimate_errors_to_original"] is None
    assert row_diagnostic["refined_minus_original"]["horizontal_position_m"] > 100_000.0
    assert row_diagnostic["refined_minus_original"]["vertical_m"] == pytest.approx(1_475.0)
    assert row_diagnostic["refined_minus_original"]["fov_deg"] == pytest.approx(40.0)
    assert case_diagnostic["estimate_errors_to_original"] is not None
    assert case_diagnostic["estimate_errors_to_original"]["fov_estimated_by_solver"] is False


def test_missing_original_metadata_emits_an_explicit_unavailable_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from peakle.localize import strategy_bench as module

    sample = _sample()
    selection = TerrainSelection(
        terrain=_terrain(),
        evaluation_high_resolution_patch=None,
        truth=_truth(),
        provenance={"far_source": "test"},
    )
    monkeypatch.setattr(module, "load_sample", lambda _path: sample)
    monkeypatch.setattr(module, "load_benchmark_terrain", lambda _sample, _config: selection)
    monkeypatch.setattr(module, "_load_photo", lambda _path: np.zeros((60, 80, 3), np.uint8))
    monkeypatch.setattr(
        module,
        "_evidence_tracks",
        lambda _sample, _rgb, _config: [EvidenceTrack("pfm_oracle", np.full(80, 30.0), {"available": True})],
    )
    monkeypatch.setattr(
        module,
        "_pre_solve_quality",
        lambda *_args: ({"policy": "gt_dem_compat_v1", "tier": "MAP_A"}, {"usable": True}),
    )

    sample_row, cases = run_sample_matrix(
        Path("/unused"),
        MatrixConfig(
            algorithms=("keep-prior",),
            evidence_tracks=("pfm_oracle",),
            prior_regimes=("raw_metadata",),
            terrain_grid=128,
        ),
    )

    assert sample_row["original_metadata_diagnostic"]["available"] is False
    assert sample_row["original_metadata_diagnostic"]["refined_minus_original"] is None
    assert cases[0]["original_metadata_diagnostic"]["available"] is False
    assert cases[0]["original_metadata_diagnostic"]["estimate_errors_to_original"] is None


def test_render_pnp_abstention_is_a_paired_attempt_failure_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from peakle.localize import strategy_bench as module

    sample = _sample()
    evaluation_patch = Patch(
        x_m=np.asarray([-50.0, 50.0]),
        y_m=np.asarray([-50.0, 50.0]),
        elevation_m=np.full((2, 2), 9_999.0),
    )
    estimator_patch = Patch(
        x_m=np.asarray([-250.0, 250.0]),
        y_m=np.asarray([-250.0, 250.0]),
        elevation_m=np.full((2, 2), 1_234.0),
    )
    selection = TerrainSelection(
        terrain=_terrain(),
        evaluation_high_resolution_patch=evaluation_patch,
        truth=_truth(),
        provenance={
            "far_source": "test",
            "evaluation_reference_patch": {"used_by_estimator": False},
        },
    )

    class Matcher:
        def identity(self):
            return {"id": "test-matcher"}

        def match(self, query_rgb, render_rgb):
            del query_rgb, render_rgb
            raise AssertionError("the injected solver result should bypass matcher execution")

    resources = RenderMatchResources(
        matcher=Matcher(),
        appearance=None,
        config=RenderMatchConfig(refinement_passes=0),
        provenance={"test": True},
    )
    monkeypatch.setattr(module, "load_sample", lambda _path: sample)
    monkeypatch.setattr(module, "load_benchmark_terrain", lambda _sample, _config: selection)
    monkeypatch.setattr(module, "_load_photo", lambda _path: np.zeros((60, 80, 3), np.uint8))
    monkeypatch.setattr(
        module,
        "_evidence_tracks",
        lambda _sample, _rgb, _config: [
            EvidenceTrack("photo_rgb", None, {"available": True, "source": "query_photo_rgb"})
        ],
    )
    monkeypatch.setattr(
        module,
        "_pre_solve_quality",
        lambda *_args: ({"policy": "gt_dem_compat_v1", "tier": "MAP_A"}, {"usable": True}),
    )
    monkeypatch.setattr(module, "_render_match_resources", lambda _terrain, _config: resources)
    patch_load_centers = []

    def fake_cached_patch(_terrain, center, **kwargs):
        patch_load_centers.append(center)
        assert kwargs["center_source"] == "supplied_position_prior"
        assert kwargs["uses_reference_truth"] is False
        return module.TerrainPatchProvision(
            estimator_patch,
            {
                "center_source": "supplied_position_prior",
                "uses_reference_truth": False,
                "used_by_estimator": True,
                "coverage": {"available": True},
            },
        )

    monkeypatch.setattr(module, "_cached_patch_at_local_position", fake_cached_patch)

    def fake_render_solve(*_args, **kwargs):
        assert kwargs["native_elevation_patch"] is estimator_patch
        assert kwargs["native_elevation_patch"] is not evaluation_patch
        return RenderMatchPoseResult(
            status="abstained",
            extrinsics=None,
            diagnostics={
                "abstain_reason": "insufficient_correspondences",
                "frames": [
                    {
                        "render": {
                            "native_high_resolution_patch_used": True,
                            "native_high_resolution_patch": {"render_stride": 8},
                        }
                    }
                ],
            },
        )

    monkeypatch.setattr(module, "solve_render_match_pose", fake_render_solve)
    config = MatrixConfig(
        algorithms=("keep-prior", "render-pnp"),
        evidence_tracks=("photo_rgb",),
        prior_regimes=("perturbed_metadata",),
        terrain_grid=128,
        render_matcher="sift",
    )

    _sample_row, cases = run_sample_matrix(Path("/unused"), config)

    assert len(cases) == 2
    render_case = next(case for case in cases if case["algorithm"] == "render-pnp")
    assert len(patch_load_centers) == 1
    assert patch_load_centers[0] == LocalPoint.model_validate(render_case["prior"]["position"])
    assert render_case["terrain_inputs"]["uses_reference_truth"] is False
    assert render_case["terrain_inputs"]["native_patch"]["center_source"] == "supplied_position_prior"
    assert render_case["status"] == "ok"
    assert render_case["outcome"] == "abstained"
    assert render_case["success"]["value"] is False
    assert render_case["baseline"]["errors"]["horizontal_position_m"] == pytest.approx(200.0)
    assert "render_match_pnp_experimental_unvalidated" in render_case["ranking_exclusions"]
    assert "workbench_objective_does_not_consume_native_high_resolution_patch" not in render_case["ranking_exclusions"]
    aggregate = aggregate_matrix([render_case])[0]
    assert aggregate["attempted"] == 1
    assert aggregate["errors"] == 0
    assert aggregate["abstained"] == 1
    assert aggregate["evidence_rejected"] == 0
    assert aggregate["success_rate"] == 0.0


def test_aggregate_reports_evidence_rejection_as_an_attempted_failure() -> None:
    case = {
        "id": "rejected",
        "name": "a",
        "algorithm": "horizon",
        "prior_regime": "position_only",
        "evidence_track": "photo_auto",
        "status": "ok",
        "outcome": "evidence_rejected",
        "ranking_eligible": True,
        "success": {"value": False},
        "errors": None,
        "runtime_s": 0.0,
        "compatibility_tier": "MAP_A",
    }

    aggregate = aggregate_matrix([case])[0]

    assert aggregate["attempted"] == 1
    assert aggregate["success_rate"] == 0.0
    assert aggregate["abstained"] == 0
    assert aggregate["evidence_rejected"] == 1


def test_errors_count_as_failures_and_artifact_is_write_once(tmp_path: Path) -> None:
    cases = [
        {
            "id": "ok",
            "name": "a",
            "algorithm": "powell",
            "prior_regime": "perturbed_metadata",
            "evidence_track": "pfm_oracle",
            "status": "ok",
            "ranking_eligible": True,
            "success": {"value": True},
            "errors": {"horizontal_position_m": 10.0, "yaw_deg": 1.0},
            "runtime_s": 2.0,
            "compatibility_tier": "MAP_A",
        },
        {
            "id": "error",
            "name": "b",
            "algorithm": "powell",
            "prior_regime": "perturbed_metadata",
            "evidence_track": "pfm_oracle",
            "status": "error",
            "ranking_eligible": True,
            "success": {"value": False},
            "errors": None,
            "runtime_s": 3.0,
            "compatibility_tier": "MAP_A",
        },
    ]
    aggregate = aggregate_matrix(cases)[0]

    assert aggregate["attempted"] == 2
    assert aggregate["errors"] == 1
    assert aggregate["success_rate"] == 0.5
    assert aggregate["position_success_rate"] == 0.5
    assert aggregate["yaw_success_rate"] == 0.5
    assert aggregate["primary_success_rate"] == 0.5

    output = tmp_path / "run-geopose-bench"
    committed = commit_artifact(output, run_metadata={"created_at": "now"}, rows=[{"name": "a"}], matrix_cases=cases)
    results_bytes = (output / "results.json").read_bytes()
    payload = json.loads(results_bytes)
    assert payload["schema_version"] == 2
    assert payload["matrix_cases"][1]["status"] == "error"
    assert committed["results_sha256"]
    assert json.loads((output / "run.json").read_text())["results_sha256"] == committed["results_sha256"]
    with pytest.raises(FileExistsError):
        commit_artifact(output, run_metadata={}, rows=[], matrix_cases=[])


def test_aggregate_reports_paired_change_from_the_same_prior() -> None:
    cases = [
        {
            "id": "solver",
            "name": "a",
            "algorithm": "cmaes",
            "prior_regime": "perturbed_metadata",
            "evidence_track": "pfm_oracle",
            "status": "ok",
            "ranking_eligible": True,
            "success": {"value": True},
            "errors": {"horizontal_position_m": 40.0, "yaw_deg": 2.0},
            "baseline": {
                "errors": {"horizontal_position_m": 200.0, "yaw_deg": 15.0},
                "success": {"value": False},
            },
            "runtime_s": 2.0,
            "compatibility_tier": "MAP_A",
        }
    ]

    aggregate = aggregate_matrix(cases)[0]

    assert aggregate["paired_prior_attempts"] == 1
    assert aggregate["improved_over_prior"] == 1
    assert aggregate["regressed_from_prior"] == 0
    assert aggregate["position_improved_over_prior"] == 1
    assert aggregate["yaw_improved_over_prior"] == 1
    assert aggregate["median_position_delta_vs_prior_m"] == -160.0
    assert aggregate["median_yaw_delta_vs_prior_deg"] == -13.0
