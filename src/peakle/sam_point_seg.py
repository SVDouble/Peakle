"""Point-prompted SAM segmentation backend (MobileSAM / SAM 2.1 via ultralytics).

A non-gated fallback / alternative to SAM 3: ultralytics ships MobileSAM and SAM 2.1 with
auto-download weights. Sky comes from point prompts along the top of the frame; terrain
instances come from SAM's "segment everything" automatic masks. Implements the two
`Segmenter` primitives, so it drops into the same ridge pipeline as SAM 3.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from peakle.segmenters import Segmenter

_WEIGHTS = {"mobile_sam": "mobile_sam.pt", "mobilesam": "mobile_sam.pt",
            "sam2": "sam2.1_b.pt", "sam2.1": "sam2.1_b.pt"}


class UltralyticsSamSegmenter(Segmenter):
    """MobileSAM / SAM2.1 point+everything segmentation via ultralytics."""

    def __init__(self, kind: str = "mobile_sam", weights: str | None = None) -> None:
        from ultralytics import SAM  # noqa: PLC0415

        self.name = kind.lower()
        self._model = SAM(weights or _WEIGHTS.get(self.name, "mobile_sam.pt"))

    def _u8(self, rgb: NDArray[np.float64]):
        return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    def sky_mask(self, rgb: NDArray[np.float64]) -> NDArray[np.bool_]:
        h, w = rgb.shape[:2]
        pts = [[int(w * f), max(2, h // 40)] for f in (0.12, 0.3, 0.5, 0.7, 0.88)]
        res = self._model(self._u8(rgb), points=pts, labels=[1] * len(pts), verbose=False)
        masks = res[0].masks
        if masks is None:
            return np.zeros((h, w), bool)
        sky = masks.data.cpu().numpy().any(0)
        from scipy.ndimage import label  # noqa: PLC0415

        lab, n = label(sky)  # keep only sky touching the top of the frame
        if n:
            top = set(np.unique(lab[:3])) - {0}
            sky = np.isin(lab, list(top)) if top else sky
        return sky

    def instance_masks(self, rgb: NDArray[np.float64], threshold: float) -> list[NDArray[np.bool_]]:
        # "segment everything": lower conf ⇒ more, finer instances (matches the base-class sweep)
        res = self._model(self._u8(rgb), conf=float(threshold), verbose=False)
        masks = res[0].masks
        if masks is None:
            return []
        return [m.astype(bool) for m in masks.data.cpu().numpy()]
