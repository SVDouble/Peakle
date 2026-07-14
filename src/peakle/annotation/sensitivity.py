"""Pure image-space sensitivity metrics for peak annotations."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from peakle.domain.annotations import PeakAnnotation
from peakle.domain.camera import CameraIntrinsics
from peakle.domain.contours import ImagePoint


@dataclass(frozen=True, slots=True)
class PeakDisplacement:
    peak_id: str
    anchor_px: float
    anchor_angle_deg: float
    label_position_px: float


@dataclass(frozen=True, slots=True)
class MetricSummary:
    count: int
    median: float | None
    p90: float | None


@dataclass(frozen=True, slots=True)
class AnnotationSensitivity:
    reference_visible_count: int
    candidate_visible_count: int
    precision: float | None
    recall: float | None
    f1: float | None
    lost_ids: tuple[str, ...]
    gained_ids: tuple[str, ...]
    common_ids: tuple[str, ...]
    displacements: tuple[PeakDisplacement, ...]
    anchor_px: MetricSummary
    anchor_angle_deg: MetricSummary
    label_position_px: MetricSummary


def evaluate_annotation_sensitivity(
    reference: Sequence[PeakAnnotation],
    candidate: Sequence[PeakAnnotation],
    intrinsics: CameraIntrinsics,
) -> AnnotationSensitivity:
    reference_by_id = _unique_by_id(reference, "reference")
    candidate_by_id = _unique_by_id(candidate, "candidate")
    reference_visible = {peak_id for peak_id, item in reference_by_id.items() if item.visible}
    candidate_visible = {peak_id for peak_id, item in candidate_by_id.items() if item.visible}
    common = tuple(sorted(reference_visible & candidate_visible))
    lost = tuple(sorted(reference_visible - candidate_visible))
    gained = tuple(sorted(candidate_visible - reference_visible))
    true_positives = len(common)
    precision = true_positives / len(candidate_visible) if candidate_visible else None
    recall = true_positives / len(reference_visible) if reference_visible else None
    f1_denominator = 2 * true_positives + len(lost) + len(gained)
    f1 = 2 * true_positives / f1_denominator if f1_denominator else None
    displacements = tuple(
        PeakDisplacement(
            peak_id=peak_id,
            anchor_px=_pixel_distance(reference_by_id[peak_id].anchor, candidate_by_id[peak_id].anchor),
            anchor_angle_deg=_angular_distance_deg(
                reference_by_id[peak_id].anchor,
                candidate_by_id[peak_id].anchor,
                intrinsics,
            ),
            label_position_px=_pixel_distance(
                reference_by_id[peak_id].label_position,
                candidate_by_id[peak_id].label_position,
            ),
        )
        for peak_id in common
    )
    return AnnotationSensitivity(
        reference_visible_count=len(reference_visible),
        candidate_visible_count=len(candidate_visible),
        precision=precision,
        recall=recall,
        f1=f1,
        lost_ids=lost,
        gained_ids=gained,
        common_ids=common,
        displacements=displacements,
        anchor_px=_summary([item.anchor_px for item in displacements]),
        anchor_angle_deg=_summary([item.anchor_angle_deg for item in displacements]),
        label_position_px=_summary([item.label_position_px for item in displacements]),
    )


def _unique_by_id(annotations: Sequence[PeakAnnotation], name: str) -> dict[str, PeakAnnotation]:
    result: dict[str, PeakAnnotation] = {}
    for annotation in annotations:
        if annotation.peak_id in result:
            raise ValueError(f"{name} annotations contain duplicate peak_id {annotation.peak_id!r}")
        result[annotation.peak_id] = annotation
    return result


def _pixel_distance(first: ImagePoint, second: ImagePoint) -> float:
    return math.hypot(second.x_px - first.x_px, second.y_px - first.y_px)


def _angular_distance_deg(first: ImagePoint, second: ImagePoint, intrinsics: CameraIntrinsics) -> float:
    def ray(point: ImagePoint) -> tuple[float, float, float]:
        return (
            (point.x_px - intrinsics.principal_x_px) / intrinsics.focal_length_px,
            (point.y_px - intrinsics.principal_y_px) / intrinsics.focal_length_px,
            1.0,
        )

    first_ray, second_ray = ray(first), ray(second)
    dot = sum(a * b for a, b in zip(first_ray, second_ray, strict=True))
    norm_product = math.sqrt(sum(value * value for value in first_ray) * sum(value * value for value in second_ray))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / norm_product))))


def _summary(values: Sequence[float]) -> MetricSummary:
    if not values:
        return MetricSummary(count=0, median=None, p90=None)
    ordered = sorted(values)
    position = 0.9 * (len(ordered) - 1)
    lower = math.floor(position)
    fraction = position - lower
    p90 = ordered[lower] + fraction * (ordered[math.ceil(position)] - ordered[lower])
    return MetricSummary(count=len(ordered), median=float(statistics.median(ordered)), p90=p90)
