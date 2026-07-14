from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime

import numpy as np
import pytest

from peakle.config import load_settings
from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.coordinates import LocalPoint
from peakle.localize.synthetic_pipeline_bench import (
    DEFAULT_PRIOR_REGIMES,
    ORACLE_METRIC_DEPTH_METHOD,
    SKYLINE_METHOD,
    SyntheticSearchConfig,
    aggregate_synthetic_cases,
    build_synthetic_candidate_archive,
    controlled_prior,
    evaluate_synthetic_candidate_archive,
    extraction_quality,
)
from peakle.rendering.rasterizer import SyntheticRenderer
from peakle.scripts import bench_synthetic_pipeline as script_module
from peakle.scripts.bench_synthetic_pipeline import (
    _code_provenance,
    _commit_artifact,
    _estimator_terrain,
    _radial_control_scene,
    _summary_markdown,
)
from peakle.terrain.generator import TerrainGenerator


def _rugged_case():
    settings = load_settings()
    terrain = TerrainGenerator(
        settings.terrain.model_copy(
            update={
                "seed": 17,
                "width_m": 6000.0,
                "height_m": 5000.0,
                "grid_width": 49,
                "grid_height": 37,
            }
        )
    ).generate()
    east_m = 0.0
    north_m = -1800.0
    truth = CameraExtrinsics(
        position=LocalPoint(
            east_m=east_m,
            north_m=north_m,
            up_m=terrain.elevation_at(east_m, north_m) + 2.5,
        ),
        yaw_deg=8.0,
        pitch_deg=12.0,
        roll_deg=0.0,
    )
    intrinsics = CameraIntrinsics.from_horizontal_fov(64, 36, 55.0)
    config = SyntheticSearchConfig(
        position_spacing_m=50.0,
        position_radius_steps=1,
        yaw_spacing_deg=5.0,
        yaw_radius_steps=1,
        render_stride=2,
        target_position_m=25.0,
    )
    renderer = SyntheticRenderer()
    geometry = renderer.geometry(terrain, intrinsics, truth, stride=config.render_stride)
    profiles = {"oracle_mask": np.asarray(geometry.skyline_profile, dtype=np.float64)}
    quality = {
        "oracle_mask": extraction_quality(
            profiles["oracle_mask"],
            profiles["oracle_mask"],
            extractor_name="exact_terrain_mask",
            coverage=1.0,
            agreement=1.0,
            config=config,
        )
    }
    return terrain, truth, intrinsics, config, geometry, profiles, quality


def test_archive_build_is_truth_free_deterministic_and_depth_is_explicitly_oracle_only() -> None:
    terrain, truth, intrinsics, config, geometry, profiles, _quality = _rugged_case()
    prior = controlled_prior(terrain, truth, DEFAULT_PRIOR_REGIMES["exact"], config)

    archive = build_synthetic_candidate_archive(
        terrain,
        intrinsics,
        prior,
        profiles,
        geometry.forward_depth_m,
        config=config,
    )
    repeated = build_synthetic_candidate_archive(
        terrain,
        intrinsics,
        prior,
        profiles,
        geometry.forward_depth_m,
        config=config,
    )

    assert archive == repeated
    assert archive["archive_sha256"] == repeated["archive_sha256"]
    assert archive["candidate_pool"]["candidate_count"] == 27
    assert archive["numeric_pose_reference_fields_used_directly_in_scoring"] is False
    assert archive["reference_pose_derived_query_evidence_used"] is True
    assert archive["supplied_prior_may_be_reference_derived"] is True
    assert archive["candidate_pool"]["numeric_truth_used_directly"] is False
    depth_contract = archive["reference_rendered_depth_oracle"]
    assert depth_contract["analysis_only"] is True
    assert depth_contract["production_eligible"] is False
    assert depth_contract["generated_from_exact_reference_pose"] is True
    assert depth_contract["numeric_pose_values_supplied_to_archive_builder"] is False
    assert archive["ranking_methods"][ORACLE_METRIC_DEPTH_METHOD]["analysis_only"] is True
    assert archive["ranking_methods"][ORACLE_METRIC_DEPTH_METHOD]["production_eligible"] is False
    assert archive["ranking_methods"][SKYLINE_METHOD]["production_eligible"] is False
    assert list(inspect.signature(build_synthetic_candidate_archive).parameters) == [
        "terrain",
        "intrinsics",
        "prior",
        "observed_skylines",
        "reference_rendered_depth_oracle",
        "config",
    ]
    encoded = json.dumps(archive, sort_keys=True)
    assert '"truth":' not in encoded
    assert '"errors":' not in encoded
    assert "winner_errors" not in encoded
    assert '"reference_data_used"' not in encoded


def test_post_freeze_evaluation_reports_full_proposal_recall_before_ranking() -> None:
    terrain, truth, intrinsics, config, geometry, profiles, quality = _rugged_case()
    prior = controlled_prior(terrain, truth, DEFAULT_PRIOR_REGIMES["exact"], config)
    archive = build_synthetic_candidate_archive(
        terrain,
        intrinsics,
        prior,
        profiles,
        geometry.forward_depth_m,
        config=config,
    )

    evaluation = evaluate_synthetic_candidate_archive(
        archive,
        truth,
        quality,
        {"oracle_mask": "select"},
        expected_pose_ambiguity=False,
    )

    assert evaluation["proposal"]["evaluation_order"] == "before_any_candidate_ranking"
    assert evaluation["proposal"]["full_pool_recall"] == 1.0
    assert evaluation["proposal"]["reaches_target"] is True
    depth = evaluation["oracle_depth_methods"]["methods"][ORACLE_METRIC_DEPTH_METHOD]
    assert depth["evidence_role"] == "oracle_only_reference_pose_generated"
    assert depth["production_eligible"] is False
    assert depth["winner"]["errors"]["horizontal_position_m"] == 0.0
    assert depth["winner"]["errors"]["yaw_deg"] == 0.0
    assert depth["first_target_rank"] == 1
    assert evaluation["numeric_truth_fields_used_by_archive_builder"] is False
    assert evaluation["reference_derived_evidence_used_by_archive_builder"] is True


def test_radial_negative_control_requires_and_detects_skyline_abstention() -> None:
    config = SyntheticSearchConfig(
        position_spacing_m=50.0,
        position_radius_steps=1,
        yaw_spacing_deg=5.0,
        yaw_radius_steps=2,
        render_stride=2,
        ambiguity_score_delta=0.003,
    )
    scene = _radial_control_scene(
        terrain_width_m=6000.0,
        terrain_height_m=5000.0,
        terrain_grid_width=49,
        terrain_grid_height=37,
        eye_height_m=config.eye_height_m,
    )
    intrinsics = CameraIntrinsics.from_horizontal_fov(64, 36, 55.0)
    geometry = SyntheticRenderer().geometry(
        scene["terrain"],
        intrinsics,
        scene["truth"],
        stride=config.render_stride,
    )
    profile = np.asarray(geometry.skyline_profile, dtype=np.float64)
    quality = {
        "oracle_mask": extraction_quality(
            profile,
            profile,
            extractor_name="exact_terrain_mask",
            coverage=1.0,
            agreement=1.0,
            config=config,
        )
    }
    prior = controlled_prior(scene["terrain"], scene["truth"], DEFAULT_PRIOR_REGIMES["exact"], config)
    archive = build_synthetic_candidate_archive(
        scene["terrain"],
        intrinsics,
        prior,
        {"oracle_mask": profile},
        geometry.forward_depth_m,
        config=config,
    )

    evaluation = evaluate_synthetic_candidate_archive(
        archive,
        scene["truth"],
        quality,
        {"oracle_mask": "select"},
        expected_pose_ambiguity=True,
    )
    skyline = evaluation["tracks"]["oracle_mask"]["methods"][SKYLINE_METHOD]
    depth = evaluation["oracle_depth_methods"]["methods"][ORACLE_METRIC_DEPTH_METHOD]

    assert evaluation["expected_pose_ambiguity"] is True
    assert skyline["expected_action"] == "abstain"
    assert skyline["decision"]["abstained"] is True
    assert "competitive_pose_modes" in skyline["decision"]["reasons"]
    assert skyline["decision"]["competitive_candidate_count"] >= 2
    assert skyline["decision_correct"] is True
    assert depth["selected"] is True
    assert depth["false_accept"] is True
    assert depth["decision_correct"] is False


def test_extraction_acceptance_is_truth_free_even_though_error_is_post_hoc() -> None:
    config = SyntheticSearchConfig()
    oracle = np.linspace(10.0, 20.0, 32)
    observed = oracle + 1.0

    accepted = extraction_quality(
        observed,
        oracle,
        extractor_name="color",
        coverage=0.9,
        agreement=0.8,
        config=config,
    )
    rejected = extraction_quality(
        observed,
        oracle,
        extractor_name="color",
        coverage=0.9,
        agreement=0.0,
        config=config,
    )

    assert accepted["accepted"] is True
    assert rejected["accepted"] is False
    assert accepted["acceptance_uses_truth"] is False
    assert accepted["evaluation"]["reference_mask_used"] is True
    assert accepted["evaluation"]["mae_px"] == pytest.approx(1.0)


def test_archive_tampering_is_rejected() -> None:
    terrain, truth, intrinsics, config, geometry, profiles, quality = _rugged_case()
    prior = controlled_prior(terrain, truth, DEFAULT_PRIOR_REGIMES["exact"], config)
    archive = build_synthetic_candidate_archive(
        terrain,
        intrinsics,
        prior,
        profiles,
        geometry.forward_depth_m,
        config=config,
    )
    archive["candidates"][0]["scores"]["oracle_metric_depth"] = 999.0

    with pytest.raises(ValueError, match="SHA-256"):
        evaluate_synthetic_candidate_archive(
            archive,
            truth,
            quality,
            {"oracle_mask": "select"},
            expected_pose_ambiguity=False,
        )


def test_expected_rejection_is_predeclared_not_copied_from_quality_gate() -> None:
    terrain, truth, intrinsics, config, geometry, profiles, quality = _rugged_case()
    prior = controlled_prior(terrain, truth, DEFAULT_PRIOR_REGIMES["exact"], config)
    archive = build_synthetic_candidate_archive(
        terrain,
        intrinsics,
        prior,
        profiles,
        geometry.forward_depth_m,
        config=config,
    )
    rejected_quality = {"oracle_mask": {**quality["oracle_mask"], "accepted": False}}

    evaluation = evaluate_synthetic_candidate_archive(
        archive,
        truth,
        rejected_quality,
        {"oracle_mask": "select"},
        expected_pose_ambiguity=False,
    )
    skyline = evaluation["tracks"]["oracle_mask"]["methods"][SKYLINE_METHOD]

    assert skyline["decision"]["abstained"] is True
    assert skyline["expected_action"] == "select"
    assert skyline["expected_action_source"] == "predeclared_case_design"
    assert skyline["decision_correct"] is False


def test_coarse_estimator_terrain_is_deterministic_and_declares_shared_renderer_scope() -> None:
    terrain, truth, intrinsics, config, geometry, profiles, quality = _rugged_case()

    coarse, record = _estimator_terrain(terrain, truth, variant="coarse", coarse_factor=2)
    repeated, repeated_record = _estimator_terrain(terrain, truth, variant="coarse", coarse_factor=2)

    assert coarse.elevation_m.shape[0] < terrain.elevation_m.shape[0]
    assert coarse.elevation_m.shape[1] < terrain.elevation_m.shape[1]
    assert np.array_equal(coarse.elevation_m, repeated.elevation_m)
    assert record == repeated_record
    assert record["variant"] == "coarse"
    assert record["downsample_factor"] == 2
    assert record["shared_renderer_family"] == "SyntheticRenderer"
    assert record["independent_renderer_used"] is False
    assert record["authoritative"]["sha256"] != record["estimator"]["sha256"]

    prior = controlled_prior(coarse, truth, DEFAULT_PRIOR_REGIMES["exact"], config)
    archive = build_synthetic_candidate_archive(
        coarse,
        intrinsics,
        prior,
        profiles,
        geometry.forward_depth_m,
        config=config,
    )
    evaluation = evaluate_synthetic_candidate_archive(
        archive,
        truth,
        quality,
        {"oracle_mask": "select"},
        expected_pose_ambiguity=False,
    )

    assert archive["estimator_terrain"]["sha256"] == record["estimator"]["sha256"]
    assert evaluation["proposal"]["full_pool_recall"] == 1.0


def test_aggregates_deduplicate_observations_and_report_depth_once_per_archive() -> None:
    terrain, truth, intrinsics, config, geometry, profiles, quality = _rugged_case()
    prior = controlled_prior(terrain, truth, DEFAULT_PRIOR_REGIMES["exact"], config)
    archive = build_synthetic_candidate_archive(
        terrain,
        intrinsics,
        prior,
        profiles,
        geometry.forward_depth_m,
        config=config,
    )
    evaluation = evaluate_synthetic_candidate_archive(
        archive,
        truth,
        quality,
        {"oracle_mask": "select"},
        expected_pose_ambiguity=False,
    )
    cases = [
        {
            "case_id": f"case-{index}",
            "observation_id": "same-query",
            "estimator_terrain": {"variant": "exact"},
            "evaluation": evaluation,
        }
        for index in range(2)
    ]

    aggregates = aggregate_synthetic_cases(cases)
    extraction = {row["track"]: row for row in aggregates["observation_extraction"]}
    depth = next(
        row
        for row in aggregates["candidate_ranking"]
        if row["track"] == "reference_depth_oracle" and row["method"] == ORACLE_METRIC_DEPTH_METHOD
    )

    assert extraction["oracle_mask"]["unique_observations"] == 1
    assert depth["cases"] == 2
    assert not any(
        row["track"] == "oracle_mask" and row["method"] == ORACLE_METRIC_DEPTH_METHOD
        for row in aggregates["candidate_ranking"]
    )
    summary = _summary_markdown({"aggregates": aggregates})
    assert "selected pose reaches target" in summary
    assert "identity check, not benchmark success" in summary


def test_code_provenance_is_scoped_to_implementation_not_user_settings() -> None:
    provenance = _code_provenance()

    assert provenance["scope"] == "listed_implementation_paths_only"
    assert ".vscode/settings.json" not in str(provenance["implementation_status"])
    assert all(not item["path"].startswith(".vscode/") for item in provenance["implementation"])


def test_artifact_commit_uses_atomic_directory_publish(tmp_path) -> None:
    output = tmp_path / "synthetic-v2"
    now = datetime.now(UTC)
    results = {"cases": [], "schema": "test", "aggregates": {}}

    _commit_artifact(output, results, "# summary\n", started_at=now, finished_at=now)

    assert output.is_dir()
    assert {path.name for path in output.iterdir()} == {"results.json", "summary.md", "run.json"}
    assert (output / "results.json").read_bytes() == script_module.canonical_json_bytes(results)
    assert (output / "summary.md").read_bytes() == b"# summary\n"
    assert json.loads((output / "results.json").read_text()) == results
    run_bytes = (output / "run.json").read_bytes()
    run = json.loads(run_bytes)
    assert run_bytes == script_module.canonical_json_bytes(run)
    assert run["status"] == "complete"
    assert not list(tmp_path.glob(".*.staging-*"))


def test_failed_artifact_staging_never_publishes_final_directory(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "synthetic-v2-failure"
    now = datetime.now(UTC)
    calls = 0
    real_write = script_module._write_once

    def fail_on_run(path, data):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected staging failure")
        real_write(path, data)

    monkeypatch.setattr(script_module, "_write_once", fail_on_run)

    with pytest.raises(OSError, match="injected staging failure"):
        _commit_artifact(
            output,
            {"cases": [], "schema": "test", "aggregates": {}},
            "# summary\n",
            started_at=now,
            finished_at=now,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".*.staging-*"))
