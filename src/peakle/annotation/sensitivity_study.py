"""Controlled pinhole pose perturbations for annotation-utility measurement."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from peakle.annotation.labeling import PeakLabeler
from peakle.annotation.sensitivity import AnnotationSensitivity, evaluate_annotation_sensitivity
from peakle.domain.annotations import PeakAnnotation
from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.coordinates import LocalPoint
from peakle.scene.state import SceneState, view_id

PerturbationFamily = Literal["exact", "position", "yaw", "fov", "height"]
PositionDirection = Literal["along", "against", "left", "right", "forward-left", "forward-right"]
AnnotationReason = Literal[
    "accepted",
    "behind-camera",
    "outside-image",
    "not-on-skyline",
    "label-budget",
    "label-overlap",
]
StudySchema = Literal["peakle_annotation_sensitivity_study_v1"]
SuiteSchema = Literal["peakle_annotation_sensitivity_suite_v1"]

POSITION_BEARING_OFFSETS_DEG: dict[PositionDirection, float] = {
    "along": 0.0,
    "against": 180.0,
    "left": -90.0,
    "right": 90.0,
    "forward-left": -45.0,
    "forward-right": 45.0,
}
PRE_LAYOUT_VISIBLE_REASONS = frozenset({"accepted", "label-budget", "label-overlap"})
DISTANCE_BANDS_M: tuple[tuple[float, float | None], ...] = (
    (0.0, 2_000.0),
    (2_000.0, 5_000.0),
    (5_000.0, 10_000.0),
    (10_000.0, None),
)


class _FrozenRecord(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False, populate_by_name=True, serialize_by_alias=True)


class TruthContract(_FrozenRecord):
    artifact_kind: Literal["annotation_sensitivity_diagnostic"] = "annotation_sensitivity_diagnostic"
    truth_class: Literal["diagnostic_oracle"] = "diagnostic_oracle"
    production_eligible: Literal[False] = False
    estimator_present: Literal[False] = False
    controlled_reference_derived_perturbations: Literal[True] = True
    same_annotation_and_renderer_pipeline_on_both_sides: Literal[True] = True
    inverse_crime: Literal[True] = True
    camera_model: Literal["pinhole"] = "pinhole"


class PerturbationCase(_FrozenRecord):
    case_id: str
    family: PerturbationFamily
    value: float
    direction: PositionDirection | None = None

    @model_validator(mode="after")
    def _validate_one_factor(self) -> PerturbationCase:
        if self.family == "exact" and (self.value != 0.0 or self.direction is not None):
            raise ValueError("exact perturbation must have value zero and no direction")
        if self.family == "position" and (self.value <= 0.0 or self.direction is None):
            raise ValueError("position perturbation needs a positive distance and direction")
        if self.family not in {"exact", "position"} and (self.value == 0.0 or self.direction is not None):
            raise ValueError(f"{self.family} perturbation needs a non-zero value and no direction")
        return self


class RenderCoverage(_FrozenRecord):
    occupied_columns: int = Field(ge=0)
    fraction: float = Field(ge=0.0, le=1.0)


class PeakRange(_FrozenRecord):
    peak_id: str
    horizontal_m: float = Field(ge=0.0)
    slant_m: float = Field(ge=0.0)


class RealizedPoseDelta(_FrozenRecord):
    east_m: float
    north_m: float
    up_m: float
    horizontal_m: float = Field(ge=0.0)
    candidate_ground_clearance_m: float


class AnnotationDecision(_FrozenRecord):
    peak_id: str
    reason: AnnotationReason
    displayed: bool


class DistanceStratum(_FrozenRecord):
    lower_m: float = Field(ge=0.0)
    upper_m: float | None
    peak_ids: tuple[str, ...]
    displayed: AnnotationSensitivity
    pre_layout: AnnotationSensitivity


class StudyCaseResult(_FrozenRecord):
    case: PerturbationCase
    candidate_extrinsics: CameraExtrinsics
    candidate_intrinsics: CameraIntrinsics
    realized_pose_delta: RealizedPoseDelta
    candidate_decisions: tuple[AnnotationDecision, ...]
    coverage: RenderCoverage
    displayed: AnnotationSensitivity
    pre_layout: AnnotationSensitivity
    distance_strata: tuple[DistanceStratum, ...]


class SensitivityAggregate(_FrozenRecord):
    family: PerturbationFamily
    value: float
    direction: PositionDirection | None
    case_count: int = Field(gt=0)
    displayed_f1_defined_count: int = Field(ge=0)
    pre_layout_f1_defined_count: int = Field(ge=0)
    displayed_f1_mean: float | None
    pre_layout_f1_mean: float | None
    displayed_anchor_p90_median_px: float | None
    pre_layout_anchor_p90_median_px: float | None
    displayed_label_position_p90_median_px: float | None
    pre_layout_label_position_p90_median_px: float | None
    minimum_occupied_column_fraction: float


class AnnotationSensitivityStudy(_FrozenRecord):
    schema_id: StudySchema = Field("peakle_annotation_sensitivity_study_v1", alias="schema")
    truth_contract: TruthContract = TruthContract()
    view_id: str
    reference_extrinsics: CameraExtrinsics
    reference_intrinsics: CameraIntrinsics
    reference_coverage: RenderCoverage
    reference_peak_ranges: tuple[PeakRange, ...]
    reference_decisions: tuple[AnnotationDecision, ...]
    cases: tuple[StudyCaseResult, ...]
    aggregates: tuple[SensitivityAggregate, ...]


class AnnotationSensitivitySuite(_FrozenRecord):
    schema_id: SuiteSchema = Field("peakle_annotation_sensitivity_suite_v1", alias="schema")
    truth_contract: TruthContract = TruthContract()
    view_ids: tuple[str, ...]
    studies: tuple[AnnotationSensitivityStudy, ...]
    aggregates: tuple[SensitivityAggregate, ...]

    @model_validator(mode="after")
    def _validate_views(self) -> AnnotationSensitivitySuite:
        if not self.view_ids or len(set(self.view_ids)) != len(self.view_ids):
            raise ValueError("suite view_ids must be non-empty and unique")
        if tuple(study.view_id for study in self.studies) != self.view_ids:
            raise ValueError("suite view_ids must match studies in order")
        if any(study.truth_contract != self.truth_contract for study in self.studies):
            raise ValueError("suite studies must share its truth contract")
        return self


def default_perturbation_cases() -> tuple[PerturbationCase, ...]:
    cases = [PerturbationCase(case_id="exact", family="exact", value=0.0)]
    for distance_m in (25.0, 50.0, 100.0, 200.0, 400.0):
        for direction in POSITION_BEARING_OFFSETS_DEG:
            cases.append(
                PerturbationCase(
                    case_id=f"position-{direction}-{_value_slug(distance_m)}m",
                    family="position",
                    value=distance_m,
                    direction=direction,
                )
            )
    for family, values, suffix in (
        ("yaw", (-0.5, 0.5, -1.0, 1.0, -2.0, 2.0, -5.0, 5.0), "deg"),
        ("fov", (-1.0, 1.0, -3.0, 3.0, -5.0, 5.0), "deg"),
        ("height", (-10.0, 10.0, -50.0, 50.0), "m"),
    ):
        for value in values:
            cases.append(
                PerturbationCase(
                    case_id=(f"{family}-{'plus' if value > 0 else 'minus'}-{_value_slug(abs(value))}{suffix}"),
                    family=family,
                    value=value,
                )
            )
    return tuple(cases)


def run_annotation_sensitivity_study(
    state: SceneState,
    camera_index: int,
    *,
    cases: Sequence[PerturbationCase] | None = None,
    terrain_stride: int = 1,
    max_labels: int = 8,
) -> AnnotationSensitivityStudy:
    """Evaluate declared perturbations around one trusted synthetic camera."""

    if terrain_stride < 1:
        raise ValueError("terrain_stride must be positive")
    if max_labels < 1:
        raise ValueError("max_labels must be positive")
    if not 0 <= camera_index < len(state.true_cameras):
        raise IndexError(f"camera_index {camera_index} is outside the true-camera list")
    selected_cases = tuple(default_perturbation_cases() if cases is None else cases)
    if not selected_cases:
        raise ValueError("perturbation cases must be non-empty")
    if len({case.case_id for case in selected_cases}) != len(selected_cases):
        raise ValueError("perturbation case IDs must be unique")
    reference = state.true_cameras[camera_index]
    prepared = tuple(_prepare_case(state, reference, state.intrinsics, case) for case in selected_cases)
    labeler = PeakLabeler()
    reference_geometry = state.renderer.geometry(state.terrain, state.intrinsics, reference, stride=terrain_stride)
    reference_annotations = labeler.build_annotations(
        state.peaks,
        state.intrinsics,
        reference,
        reference_geometry.skyline_profile,
        max_labels=max_labels,
    )
    peak_ranges = _peak_ranges(state, reference)
    results: list[StudyCaseResult] = []
    for case, extrinsics, intrinsics in prepared:
        geometry = state.renderer.geometry(state.terrain, intrinsics, extrinsics, stride=terrain_stride)
        candidate_annotations = labeler.build_annotations(
            state.peaks,
            intrinsics,
            extrinsics,
            geometry.skyline_profile,
            max_labels=max_labels,
        )
        results.append(
            _case_result(
                case,
                state.intrinsics,
                extrinsics,
                intrinsics,
                _realized_pose_delta(state, reference, extrinsics),
                _annotation_decisions(candidate_annotations),
                reference_annotations,
                candidate_annotations,
                peak_ranges,
                geometry.terrain_mask,
            )
        )
    frozen_results = tuple(results)
    return AnnotationSensitivityStudy(
        view_id=view_id(camera_index),
        reference_extrinsics=reference,
        reference_intrinsics=state.intrinsics,
        reference_coverage=_coverage(reference_geometry.terrain_mask),
        reference_peak_ranges=peak_ranges,
        reference_decisions=_annotation_decisions(reference_annotations),
        cases=frozen_results,
        aggregates=aggregate_study_cases(frozen_results),
    )


def run_annotation_sensitivity_suite(
    state: SceneState,
    *,
    camera_indices: Sequence[int] | None = None,
    cases: Sequence[PerturbationCase] | None = None,
    terrain_stride: int = 1,
    max_labels: int = 8,
) -> AnnotationSensitivitySuite:
    indices = tuple(range(len(state.true_cameras))) if camera_indices is None else tuple(camera_indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("camera_indices must be non-empty and unique")
    studies = tuple(
        run_annotation_sensitivity_study(
            state,
            index,
            cases=cases,
            terrain_stride=terrain_stride,
            max_labels=max_labels,
        )
        for index in indices
    )
    return AnnotationSensitivitySuite(
        view_ids=tuple(study.view_id for study in studies),
        studies=studies,
        aggregates=aggregate_study_cases(tuple(result for study in studies for result in study.cases)),
    )


def aggregate_study_cases(cases: Sequence[StudyCaseResult]) -> tuple[SensitivityAggregate, ...]:
    grouped: dict[tuple[str, float, str], list[StudyCaseResult]] = defaultdict(list)
    for result in cases:
        grouped[(result.case.family, result.case.value, result.case.direction or "")].append(result)
    aggregates = []
    for family, value, direction_text in sorted(grouped, key=_aggregate_sort_key):
        group = grouped[(family, value, direction_text)]
        first = group[0].case
        aggregates.append(
            SensitivityAggregate(
                family=first.family,
                value=value,
                direction=first.direction,
                case_count=len(group),
                displayed_f1_defined_count=sum(item.displayed.f1 is not None for item in group),
                pre_layout_f1_defined_count=sum(item.pre_layout.f1 is not None for item in group),
                displayed_f1_mean=_mean(item.displayed.f1 for item in group),
                pre_layout_f1_mean=_mean(item.pre_layout.f1 for item in group),
                displayed_anchor_p90_median_px=_median(item.displayed.anchor_px.p90 for item in group),
                pre_layout_anchor_p90_median_px=_median(item.pre_layout.anchor_px.p90 for item in group),
                displayed_label_position_p90_median_px=_median(item.displayed.label_position_px.p90 for item in group),
                pre_layout_label_position_p90_median_px=_median(
                    item.pre_layout.label_position_px.p90 for item in group
                ),
                minimum_occupied_column_fraction=min(item.coverage.fraction for item in group),
            )
        )
    return tuple(aggregates)


def _prepare_case(
    state: SceneState,
    reference: CameraExtrinsics,
    reference_intrinsics: CameraIntrinsics,
    case: PerturbationCase,
) -> tuple[PerturbationCase, CameraExtrinsics, CameraIntrinsics]:
    east_m, north_m, up_m = reference.position.as_tuple()
    yaw_deg = reference.yaw_deg
    intrinsics = reference_intrinsics
    if case.family == "position":
        assert case.direction is not None
        bearing = math.radians(reference.yaw_deg + POSITION_BEARING_OFFSETS_DEG[case.direction])
        east_m += case.value * math.sin(bearing)
        north_m += case.value * math.cos(bearing)
        _validate_xy(state, east_m, north_m, case.case_id)
        clearance = reference.position.up_m - state.terrain.elevation_at(
            reference.position.east_m, reference.position.north_m
        )
        up_m = state.terrain.elevation_at(east_m, north_m) + clearance
    elif case.family == "yaw":
        yaw_deg = _wrap180(yaw_deg + case.value)
    elif case.family == "fov":
        target_fov = reference_intrinsics.horizontal_fov_deg() + case.value
        generated = CameraIntrinsics.from_horizontal_fov(
            reference_intrinsics.width_px, reference_intrinsics.height_px, target_fov
        )
        intrinsics = generated.model_copy(
            update={
                "principal_x_px": reference_intrinsics.principal_x_px,
                "principal_y_px": reference_intrinsics.principal_y_px,
            }
        )
    elif case.family == "height":
        up_m += case.value
    _validate_xy(state, east_m, north_m, case.case_id)
    return (
        case,
        CameraExtrinsics(
            position=LocalPoint(east_m=east_m, north_m=north_m, up_m=up_m),
            yaw_deg=yaw_deg,
            pitch_deg=reference.pitch_deg,
            roll_deg=reference.roll_deg,
        ),
        intrinsics,
    )


def _case_result(
    case: PerturbationCase,
    reference_intrinsics: CameraIntrinsics,
    candidate: CameraExtrinsics,
    candidate_intrinsics: CameraIntrinsics,
    realized_pose_delta: RealizedPoseDelta,
    candidate_decisions: tuple[AnnotationDecision, ...],
    reference_annotations: Sequence[PeakAnnotation],
    candidate_annotations: Sequence[PeakAnnotation],
    peak_ranges: tuple[PeakRange, ...],
    terrain_mask: np.ndarray,
) -> StudyCaseResult:
    reference_pre_layout = _pre_layout(reference_annotations)
    candidate_pre_layout = _pre_layout(candidate_annotations)
    distance_strata = []
    for lower_m, upper_m in DISTANCE_BANDS_M:
        peak_ids = frozenset(
            item.peak_id
            for item in peak_ranges
            if item.horizontal_m >= lower_m and (upper_m is None or item.horizontal_m < upper_m)
        )
        distance_strata.append(
            DistanceStratum(
                lower_m=lower_m,
                upper_m=upper_m,
                peak_ids=tuple(sorted(peak_ids)),
                displayed=evaluate_annotation_sensitivity(
                    _select(reference_annotations, peak_ids),
                    _select(candidate_annotations, peak_ids),
                    reference_intrinsics,
                ),
                pre_layout=evaluate_annotation_sensitivity(
                    _select(reference_pre_layout, peak_ids),
                    _select(candidate_pre_layout, peak_ids),
                    reference_intrinsics,
                ),
            )
        )
    return StudyCaseResult(
        case=case,
        candidate_extrinsics=candidate,
        candidate_intrinsics=candidate_intrinsics,
        realized_pose_delta=realized_pose_delta,
        candidate_decisions=candidate_decisions,
        coverage=_coverage(terrain_mask),
        displayed=evaluate_annotation_sensitivity(reference_annotations, candidate_annotations, reference_intrinsics),
        pre_layout=evaluate_annotation_sensitivity(reference_pre_layout, candidate_pre_layout, reference_intrinsics),
        distance_strata=tuple(distance_strata),
    )


def _pre_layout(annotations: Sequence[PeakAnnotation]) -> tuple[PeakAnnotation, ...]:
    return tuple(item.model_copy(update={"visible": item.reason in PRE_LAYOUT_VISIBLE_REASONS}) for item in annotations)


def _annotation_decisions(annotations: Sequence[PeakAnnotation]) -> tuple[AnnotationDecision, ...]:
    return tuple(
        AnnotationDecision(
            peak_id=item.peak_id,
            reason=cast(AnnotationReason, item.reason),
            displayed=item.visible,
        )
        for item in annotations
    )


def _select(annotations: Sequence[PeakAnnotation], peak_ids: frozenset[str]) -> tuple[PeakAnnotation, ...]:
    return tuple(item for item in annotations if item.peak_id in peak_ids)


def _peak_ranges(state: SceneState, reference: CameraExtrinsics) -> tuple[PeakRange, ...]:
    ranges = []
    for peak in state.peaks:
        dx = peak.local_position.east_m - reference.position.east_m
        dy = peak.local_position.north_m - reference.position.north_m
        dz = peak.local_position.up_m - reference.position.up_m
        horizontal = math.hypot(dx, dy)
        ranges.append(PeakRange(peak_id=peak.id, horizontal_m=horizontal, slant_m=math.hypot(horizontal, dz)))
    return tuple(sorted(ranges, key=lambda item: item.peak_id))


def _realized_pose_delta(
    state: SceneState, reference: CameraExtrinsics, candidate: CameraExtrinsics
) -> RealizedPoseDelta:
    east_m = candidate.position.east_m - reference.position.east_m
    north_m = candidate.position.north_m - reference.position.north_m
    return RealizedPoseDelta(
        east_m=east_m,
        north_m=north_m,
        up_m=candidate.position.up_m - reference.position.up_m,
        horizontal_m=math.hypot(east_m, north_m),
        candidate_ground_clearance_m=candidate.position.up_m
        - state.terrain.elevation_at(candidate.position.east_m, candidate.position.north_m),
    )


def _coverage(mask: np.ndarray) -> RenderCoverage:
    occupied = int(np.count_nonzero(np.any(np.asarray(mask, dtype=np.bool_), axis=0)))
    total = int(mask.shape[1])
    return RenderCoverage(occupied_columns=occupied, fraction=occupied / total)


def _validate_xy(state: SceneState, east_m: float, north_m: float, case_id: str) -> None:
    if not (float(state.terrain.x_m[0]) <= east_m <= float(state.terrain.x_m[-1])) or not (
        float(state.terrain.y_m[0]) <= north_m <= float(state.terrain.y_m[-1])
    ):
        raise ValueError(f"perturbation {case_id!r} places the camera outside terrain bounds")


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return statistics.fmean(finite) if finite else None


def _median(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return float(statistics.median(finite)) if finite else None


def _aggregate_sort_key(key: tuple[str, float, str]) -> tuple[int, float, str]:
    return ({"exact": 0, "position": 1, "yaw": 2, "fov": 3, "height": 4}[key[0]], key[1], key[2])


def _wrap180(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _value_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")
