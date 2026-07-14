from __future__ import annotations

import math
from collections import Counter

import pytest

from peakle.annotation.sensitivity_study import (
    AnnotationSensitivityStudy,
    AnnotationSensitivitySuite,
    PerturbationCase,
    default_perturbation_cases,
    run_annotation_sensitivity_study,
    run_annotation_sensitivity_suite,
)
from peakle.config import AppSettings
from peakle.scene.state import SceneState


@pytest.fixture
def study_state(small_settings: AppSettings) -> SceneState:
    settings = small_settings.model_copy(
        update={
            "render": small_settings.render.model_copy(update={"image_width": 160, "image_height": 120}),
            "camera": small_settings.camera.model_copy(update={"view_count": 2}),
        }
    )
    return SceneState.from_settings(settings)


def test_default_cases_are_complete_one_factor_and_stably_named() -> None:
    cases = default_perturbation_cases()

    assert len(cases) == 49
    assert len({case.case_id for case in cases}) == len(cases)
    assert Counter(case.family for case in cases) == {
        "exact": 1,
        "position": 30,
        "yaw": 8,
        "fov": 6,
        "height": 4,
    }
    assert cases[0].case_id == "exact"
    assert "position-forward-left-25m" in {case.case_id for case in cases}
    assert "position-forward-right-400m" in {case.case_id for case in cases}
    assert "yaw-minus-0p5deg" in {case.case_id for case in cases}
    assert "fov-plus-3deg" in {case.case_id for case in cases}
    assert "height-minus-50m" in {case.case_id for case in cases}


def test_direction_offsets_preserve_orientation_and_use_camera_axes(study_state: SceneState) -> None:
    cases = (
        PerturbationCase(case_id="left", family="position", value=25.0, direction="left"),
        PerturbationCase(case_id="right", family="position", value=25.0, direction="right"),
    )
    study = run_annotation_sensitivity_study(study_state, 0, cases=cases, terrain_stride=2)
    reference = study.reference_extrinsics
    yaw = math.radians(reference.yaw_deg)
    forward = (math.sin(yaw), math.cos(yaw))
    right = (math.cos(yaw), -math.sin(yaw))
    reference_clearance = reference.position.up_m - study_state.terrain.elevation_at(
        reference.position.east_m, reference.position.north_m
    )
    realized_up = []

    for result, expected_right_m in zip(study.cases, (-25.0, 25.0), strict=True):
        candidate = result.candidate_extrinsics
        realized = result.realized_pose_delta
        delta = (
            candidate.position.east_m - reference.position.east_m,
            candidate.position.north_m - reference.position.north_m,
        )
        assert math.hypot(*delta) == pytest.approx(25.0)
        assert delta[0] * forward[0] + delta[1] * forward[1] == pytest.approx(0.0, abs=1e-9)
        assert delta[0] * right[0] + delta[1] * right[1] == pytest.approx(expected_right_m)
        assert (realized.east_m, realized.north_m) == pytest.approx(delta)
        assert realized.horizontal_m == pytest.approx(25.0)
        assert realized.candidate_ground_clearance_m == pytest.approx(reference_clearance)
        realized_up.append(realized.up_m)
        assert (candidate.yaw_deg, candidate.pitch_deg, candidate.roll_deg) == (
            reference.yaw_deg,
            reference.pitch_deg,
            reference.roll_deg,
        )
    assert any(abs(value) > 0.1 for value in realized_up)


def test_out_of_bounds_position_is_rejected_without_clipping(study_state: SceneState) -> None:
    outside = PerturbationCase(case_id="outside", family="position", value=100_000.0, direction="along")

    with pytest.raises(ValueError, match="outside.*outside terrain bounds"):
        run_annotation_sensitivity_study(study_state, 0, cases=(outside,))


def test_exact_case_separates_prelayout_and_display_and_roundtrips(study_state: SceneState) -> None:
    exact = PerturbationCase(case_id="exact", family="exact", value=0.0)
    study = run_annotation_sensitivity_study(
        study_state,
        0,
        cases=(exact,),
        terrain_stride=2,
        max_labels=1,
    )
    result = study.cases[0]

    assert result.displayed.f1 == result.pre_layout.f1 == 1.0
    assert result.displayed.anchor_px.p90 == result.pre_layout.anchor_px.p90 == 0.0
    assert result.pre_layout.reference_visible_count > result.displayed.reference_visible_count
    assert result.candidate_extrinsics == study.reference_extrinsics
    assert result.candidate_intrinsics == study.reference_intrinsics
    assert result.candidate_decisions == study.reference_decisions
    assert {decision.reason for decision in study.reference_decisions} >= {"accepted", "label-budget"}
    assert study.truth_contract.model_dump() == {
        "artifact_kind": "annotation_sensitivity_diagnostic",
        "truth_class": "diagnostic_oracle",
        "production_eligible": False,
        "estimator_present": False,
        "controlled_reference_derived_perturbations": True,
        "same_annotation_and_renderer_pipeline_on_both_sides": True,
        "inverse_crime": True,
        "camera_model": "pinhole",
    }
    assert result.realized_pose_delta.horizontal_m == result.realized_pose_delta.up_m == 0.0
    assert study.reference_coverage.occupied_columns == study.reference_intrinsics.width_px
    assert result.coverage.fraction == pytest.approx(1.0)
    assert {item.peak_id for item in study.reference_peak_ranges} == {
        peak_id for stratum in result.distance_strata for peak_id in stratum.peak_ids
    }
    encoded = study.model_dump_json()
    assert '"schema":"peakle_annotation_sensitivity_study_v1"' in encoded
    assert AnnotationSensitivityStudy.model_validate_json(encoded) == study


def test_fov_is_the_only_changed_camera_parameter_and_aggregates_by_key(study_state: SceneState) -> None:
    cases = (
        PerturbationCase(case_id="fov-a", family="fov", value=3.0),
        PerturbationCase(case_id="fov-b", family="fov", value=3.0),
    )
    study = run_annotation_sensitivity_study(study_state, 0, cases=cases, terrain_stride=2)

    assert len(study.aggregates) == 1
    aggregate = study.aggregates[0]
    assert (aggregate.family, aggregate.value, aggregate.direction, aggregate.case_count) == ("fov", 3.0, None, 2)
    for result in study.cases:
        assert result.candidate_extrinsics == study.reference_extrinsics
        assert result.candidate_intrinsics.horizontal_fov_deg() == pytest.approx(
            study.reference_intrinsics.horizontal_fov_deg() + 3.0
        )
        assert result.candidate_intrinsics.principal_x_px == study.reference_intrinsics.principal_x_px
        assert 0 <= result.coverage.occupied_columns <= study.reference_intrinsics.width_px


def test_multi_view_suite_has_typed_global_aggregates_and_unique_views(study_state: SceneState) -> None:
    exact = PerturbationCase(case_id="exact", family="exact", value=0.0)

    suite = run_annotation_sensitivity_suite(study_state, cases=(exact,), terrain_stride=2)

    assert suite.view_ids == ("view-01", "view-02")
    assert tuple(study.view_id for study in suite.studies) == suite.view_ids
    assert len(suite.aggregates) == 1
    assert suite.aggregates[0].case_count == 2
    assert suite.aggregates[0].displayed_f1_defined_count == 2
    assert suite.aggregates[0].pre_layout_f1_defined_count == 2
    assert suite.aggregates[0].displayed_label_position_p90_median_px == 0.0
    assert suite.truth_contract == suite.studies[0].truth_contract == suite.studies[1].truth_contract
    encoded = suite.model_dump_json()
    assert '"schema":"peakle_annotation_sensitivity_suite_v1"' in encoded
    assert AnnotationSensitivitySuite.model_validate_json(encoded) == suite

    with pytest.raises(ValueError, match="non-empty and unique"):
        run_annotation_sensitivity_suite(study_state, camera_indices=(0, 0), cases=(exact,), terrain_stride=2)

    with pytest.raises(ValueError, match="cases must be non-empty"):
        run_annotation_sensitivity_suite(study_state, cases=(), terrain_stride=2)
