"""Skyline contour detectors.

A `ContourDetector` turns a rendered (or, later, a real) image into a
`SkylineContour`. Only the synthetic mask detector exists today; real edge
detectors implement the same protocol later without changing the solve flow.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from peakle.domain.contours import SkylineContour
from peakle.rendering.rasterizer import RenderArrays
from peakle.rendering.skyline import extract_skyline_from_mask


@runtime_checkable
class ContourDetector(Protocol):
    """Extracts a skyline contour from a rendered view."""

    kind: str

    def detect(self, render: RenderArrays) -> SkylineContour:
        """Extracts the skyline contour."""
        ...


class SyntheticMaskDetector:
    """Extracts the skyline from the renderer's terrain mask."""

    kind = "synthetic-mask"

    def __init__(self, smoothing_sigma: float = 1.0) -> None:
        self.smoothing_sigma = smoothing_sigma

    def detect(self, render: RenderArrays) -> SkylineContour:
        """Extracts the top terrain pixel per column from the mask."""

        return extract_skyline_from_mask(render.terrain_mask, smoothing_sigma=self.smoothing_sigma)


DEFAULT_DETECTOR = SyntheticMaskDetector()
