"""The evidence record that flows through the photo-processing pipeline.

A single `Evidence` object starts from an image (or a synthetic view) and is
*augmented* by independent stages: EXIF -> intrinsics -> position prior, contour,
depth, and any other cues. Optimization later consumes whatever evidence is
present. Stages never depend on each other's internals — only on fields being set
— so they can be added, reordered, or skipped freely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from peakle.domain.camera import CameraIntrinsics
from peakle.domain.contours import SkylineContour
from peakle.domain.pose import PosePrior
from peakle.segmentation import RidgeField


@dataclass
class ExifData:
    """Camera/GPS metadata read from an image's EXIF.

    Attributes:
        present: Whether any EXIF was found.
        focal_length_mm: Lens focal length in mm.
        focal_length_35mm_mm: 35mm-equivalent focal length in mm.
        datetime: Capture timestamp string.
        make: Camera maker.
        model: Camera model.
        gps_lat_deg: Latitude in signed degrees.
        gps_lon_deg: Longitude in signed degrees.
        gps_alt_m: Altitude in meters.
        heading_deg: Image direction (compass) in degrees from north.
    """

    present: bool = False
    focal_length_mm: float | None = None
    focal_length_35mm_mm: float | None = None
    datetime: str | None = None
    make: str | None = None
    model: str | None = None
    gps_lat_deg: float | None = None
    gps_lon_deg: float | None = None
    gps_alt_m: float | None = None
    heading_deg: float | None = None


@dataclass
class Evidence:
    """Progressively-augmented evidence about one photo / view.

    Attributes:
        source: Origin label or path.
        image: RGB image as `(H, W, 3)` float in [0, 1].
        pil: Decoded PIL image (kept for EXIF), or None for in-memory views.
        image_width_px / image_height_px: Image size.
        exif / intrinsics / prior / contour / depth / ridges: Filled by stages.
        provenance: One line per stage describing what it added.
        extra: Open bag for additional cues (snow line, peaks, ...).
    """

    source: str
    image: NDArray[np.float64]
    image_width_px: int
    image_height_px: int
    pil: Image.Image | None = None
    exif: ExifData | None = None
    intrinsics: CameraIntrinsics | None = None
    prior: PosePrior | None = None
    contour: SkylineContour | None = None
    depth: NDArray[np.float64] | None = None
    edges: NDArray[np.float64] | None = None
    ridges: RidgeField | None = None
    provenance: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: str) -> Evidence:
        """Builds evidence from an image file (keeps the PIL image for EXIF)."""

        pil = Image.open(path).convert("RGB")
        image = np.asarray(pil, dtype=np.float64) / 255.0
        return cls(source=path, image=image, image_width_px=pil.width, image_height_px=pil.height, pil=pil)

    @classmethod
    def from_array(cls, image: NDArray[np.float64], source: str = "view") -> Evidence:
        """Builds evidence from an in-memory image array (e.g. a synthetic render)."""

        array = np.asarray(image, dtype=np.float64)
        if array.max() > 1.0:
            array = array / 255.0
        height, width = array.shape[:2]
        return cls(source=source, image=array, image_width_px=int(width), image_height_px=int(height))

    def log(self, message: str) -> None:
        """Records a provenance line for a stage's contribution."""

        self.provenance.append(message)
