from __future__ import annotations

import math

import pytest

from peakle.annotation.sensitivity import evaluate_annotation_sensitivity
from peakle.domain.annotations import LabelBox, PeakAnnotation
from peakle.domain.camera import CameraIntrinsics
from peakle.domain.contours import ImagePoint

INTRINSICS = CameraIntrinsics(
    width_px=101,
    height_px=101,
    focal_length_px=100.0,
    principal_x_px=50.0,
    principal_y_px=50.0,
)


def _annotation(
    peak_id: str,
    anchor: tuple[float, float],
    *,
    label: tuple[float, float] | None = None,
    visible: bool = True,
) -> PeakAnnotation:
    label = anchor if label is None else label
    return PeakAnnotation(
        peak_id=peak_id,
        peak_name=peak_id,
        anchor=ImagePoint(x_px=anchor[0], y_px=anchor[1]),
        label_position=ImagePoint(x_px=label[0], y_px=label[1]),
        label_box=LabelBox(x_px=label[0], y_px=label[1], width_px=20.0, height_px=10.0),
        visible=visible,
        reason="accepted" if visible else "hidden",
    )


def test_exact_annotations_have_perfect_identity_and_zero_displacement() -> None:
    annotation = _annotation("peak", (50.0, 50.0), label=(40.0, 30.0))

    result = evaluate_annotation_sensitivity([annotation], [annotation.model_copy(deep=True)], INTRINSICS)

    assert (result.precision, result.recall, result.f1) == (1.0, 1.0, 1.0)
    assert result.lost_ids == result.gained_ids == ()
    assert result.common_ids == ("peak",)
    assert result.displacements[0].anchor_px == pytest.approx(0.0)
    assert result.displacements[0].anchor_angle_deg == pytest.approx(0.0)
    assert result.displacements[0].label_position_px == pytest.approx(0.0)
    assert result.anchor_px.median == result.anchor_px.p90 == 0.0


def test_shifted_lost_and_gained_annotations_are_separated() -> None:
    reference = [
        _annotation("common", (50.0, 50.0), label=(0.0, 0.0)),
        _annotation("lost", (20.0, 20.0)),
        _annotation("gained", (80.0, 80.0), visible=False),
    ]
    candidate = [
        _annotation("common", (53.0, 54.0), label=(6.0, 8.0)),
        _annotation("lost", (20.0, 20.0), visible=False),
        _annotation("gained", (80.0, 80.0)),
    ]

    result = evaluate_annotation_sensitivity(reference, candidate, INTRINSICS)

    assert (result.precision, result.recall, result.f1) == (0.5, 0.5, 0.5)
    assert result.lost_ids == ("lost",)
    assert result.gained_ids == ("gained",)
    assert result.common_ids == ("common",)
    assert result.displacements[0].anchor_px == pytest.approx(5.0)
    assert result.displacements[0].anchor_angle_deg == pytest.approx(math.degrees(math.atan(0.05)))
    assert result.displacements[0].label_position_px == pytest.approx(10.0)


@pytest.mark.parametrize("duplicate_side", ["reference", "candidate"])
def test_duplicate_peak_ids_are_rejected(duplicate_side: str) -> None:
    duplicate = [_annotation("same", (10.0, 10.0)), _annotation("same", (20.0, 20.0), visible=False)]
    reference = duplicate if duplicate_side == "reference" else []
    candidate = duplicate if duplicate_side == "candidate" else []

    with pytest.raises(ValueError, match=rf"{duplicate_side}.*duplicate peak_id 'same'"):
        evaluate_annotation_sensitivity(reference, candidate, INTRINSICS)


def test_empty_visible_overlap_has_explicit_empty_summaries() -> None:
    result = evaluate_annotation_sensitivity(
        [_annotation("reference", (10.0, 10.0))],
        [_annotation("candidate", (90.0, 90.0))],
        INTRINSICS,
    )

    assert (result.precision, result.recall, result.f1) == (0.0, 0.0, 0.0)
    assert result.displacements == ()
    for summary in (result.anchor_px, result.anchor_angle_deg, result.label_position_px):
        assert summary.count == 0
        assert summary.median is None
        assert summary.p90 is None


def test_p90_uses_linear_interpolation() -> None:
    reference = [_annotation("a", (50.0, 50.0)), _annotation("b", (20.0, 20.0))]
    candidate = [_annotation("a", (50.0, 50.0)), _annotation("b", (30.0, 20.0))]

    result = evaluate_annotation_sensitivity(reference, candidate, INTRINSICS)

    assert result.anchor_px.median == pytest.approx(5.0)
    assert result.anchor_px.p90 == pytest.approx(9.0)
