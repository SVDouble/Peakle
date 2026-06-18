"""Peak label placement."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from peakle.domain.annotations import LabelBox, PeakAnnotation
from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.contours import ImagePoint
from peakle.domain.peaks import Peak
from peakle.rendering.pinhole import project_local_point


class PeakLabeler:
    """Projects peaks and accepts non-overlapping visible labels."""

    def build_annotations(
        self,
        peaks: list[Peak],
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        skyline_profile: NDArray[np.float64],
        max_labels: int = 8,
    ) -> list[PeakAnnotation]:
        """Builds deterministic peak annotations.

        Args:
            peaks: Candidate peaks.
            intrinsics: Camera intrinsics.
            extrinsics: Camera pose used for projection.
            skyline_profile: Predicted skyline profile for visibility checks.
            max_labels: Maximum number of visible labels to accept.

        Returns:
            Accepted and rejected annotations.
        """

        accepted_boxes: list[LabelBox] = []
        annotations: list[PeakAnnotation] = []
        visible_count = 0
        sorted_peaks = sorted(
            peaks,
            key=lambda peak: (-peak.prominence_m, -peak.elevation_m, peak.id),
        )

        for peak in sorted_peaks:
            u_px, v_px, _depth, valid = project_local_point(
                peak.local_position,
                intrinsics,
                extrinsics,
            )
            reason = "accepted"
            visible = True
            if not valid:
                visible = False
                reason = "behind-camera"
            elif not (0.0 <= u_px < intrinsics.width_px and 0.0 <= v_px < intrinsics.height_px):
                visible = False
                reason = "outside-image"
            elif not self._near_skyline(u_px, v_px, skyline_profile):
                visible = False
                reason = "not-on-skyline"

            label_box = self._label_box(peak.name, u_px, v_px, intrinsics)
            if visible and visible_count >= max_labels:
                visible = False
                reason = "label-budget"
            if visible and any(_boxes_overlap(label_box, box) for box in accepted_boxes):
                visible = False
                reason = "label-overlap"
            if visible:
                accepted_boxes.append(label_box)
                visible_count += 1

            annotations.append(
                PeakAnnotation(
                    peak_id=peak.id,
                    peak_name=peak.name,
                    anchor=ImagePoint(x_px=float(u_px), y_px=float(v_px)),
                    label_position=ImagePoint(x_px=label_box.x_px, y_px=label_box.y_px),
                    label_box=label_box,
                    visible=visible,
                    reason=reason,
                )
            )
        return annotations

    def _near_skyline(
        self,
        u_px: float,
        v_px: float,
        skyline_profile: NDArray[np.float64],
    ) -> bool:
        column = int(round(u_px))
        if not 0 <= column < skyline_profile.size:
            return False
        skyline_y = float(skyline_profile[column])
        return abs(v_px - skyline_y) <= 34.0

    def _label_box(
        self,
        text: str,
        u_px: float,
        v_px: float,
        intrinsics: CameraIntrinsics,
    ) -> LabelBox:
        width = max(54.0, len(text) * 7.5 + 16.0)
        height = 22.0
        x_px = float(np.clip(u_px - width / 2.0, 6.0, intrinsics.width_px - width - 6.0))
        y_px = float(np.clip(v_px - 34.0, 6.0, intrinsics.height_px - height - 6.0))
        return LabelBox(x_px=x_px, y_px=y_px, width_px=width, height_px=height)


def _boxes_overlap(a: LabelBox, b: LabelBox, padding_px: float = 4.0) -> bool:
    return not (
        a.x_px + a.width_px + padding_px < b.x_px
        or b.x_px + b.width_px + padding_px < a.x_px
        or a.y_px + a.height_px + padding_px < b.y_px
        or b.y_px + b.height_px + padding_px < a.y_px
    )
