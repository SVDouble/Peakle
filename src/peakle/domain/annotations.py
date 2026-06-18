"""Annotation domain models."""

from __future__ import annotations

from pydantic import BaseModel

from peakle.domain.contours import ImagePoint


class LabelBox(BaseModel):
    """Image-space label bounding box.

    Attributes:
        x_px: Left coordinate in pixels.
        y_px: Top coordinate in pixels.
        width_px: Box width in pixels.
        height_px: Box height in pixels.
    """

    x_px: float
    y_px: float
    width_px: float
    height_px: float


class PeakAnnotation(BaseModel):
    """Projected annotation for a peak.

    Attributes:
        peak_id: Identifier of the annotated peak.
        peak_name: Display name.
        anchor: Projected peak position in image coordinates.
        label_position: Label origin in image coordinates.
        label_box: Label bounding box.
        visible: Whether the label is accepted for drawing.
        reason: Visibility or rejection reason.
    """

    peak_id: str
    peak_name: str
    anchor: ImagePoint
    label_position: ImagePoint
    label_box: LabelBox
    visible: bool
    reason: str
