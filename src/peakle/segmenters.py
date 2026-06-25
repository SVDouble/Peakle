"""Pluggable scene segmentation for the ridge pipeline.

A `Segmenter` turns a photo into (a) a sky mask and (b) a set of region/instance masks.
Concrete backends only implement those two primitives; everything the ridge pipeline needs
on top — the accumulated ridge-**silhouette map** and the depth-ordered **terrain layers** —
is shared here, so any backend (SAM3 text-prompted, SAM2/MobileSAM point-prompted, …) drops
into the pipeline unchanged. Pick one with `load_segmenter("sam3")`.

The ridge contours we want are silhouettes — the outlines where one piece of terrain stands
in front of another (or the sky). Every instance mask contributes its boundary; summing the
boundaries of multi-scale instances (a whole range plus its sub-peaks) recovers fine crests a
single coarse split would miss, while staying on real region boundaries (not raw texture).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Segmenter(ABC):
    """Backend-agnostic segmenter. Implement `sky_mask` + `instance_masks`; inherit the rest."""

    name: str = "segmenter"

    # ---- primitives each backend must provide ----
    @abstractmethod
    def sky_mask(self, rgb: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Boolean sky mask (True = sky). Mountain/terrain is then ``~sky``."""

    @abstractmethod
    def instance_masks(
        self, rgb: NDArray[np.float64], threshold: float
    ) -> list[NDArray[np.bool_]]:
        """Region/instance masks at a detector confidence ``threshold`` in [0, 1].
        LOWER threshold ⇒ MORE, finer instances (SAM3: detection score; SAM-auto: stability/
        IoU). The shared methods below sweep several thresholds, so this is the only knob a
        backend must map onto its own API."""

    def terrain_mask(
        self, rgb: NDArray[np.float64], threshold: float = 0.25, min_area: int = 2000
    ) -> NDArray[np.bool_]:
        """The mountain/terrain region, prompted **directly** (union of the backend's terrain
        instances, holes filled) — NOT ``~sky``. Prompting for the thing we want excludes sky
        AND foreground non-terrain (people, etc.) in one step, so their outlines never become
        spurious ridge contours."""
        from scipy.ndimage import binary_fill_holes  # noqa: PLC0415

        out = np.zeros(rgb.shape[:2], bool)
        for m in self.instance_masks(rgb, threshold):
            if int(m.sum()) >= min_area:
                out |= m
        return binary_fill_holes(out)

    # ---- shared, backend-agnostic ----
    def silhouette_map(
        self,
        rgb: NDArray[np.float64],
        mountain: NDArray[np.bool_],
        thresholds: tuple[float, ...] = (0.3, 0.15, 0.08),
        min_area: int = 1500,
        min_instances: int = 6,
    ) -> NDArray[np.float64]:
        """Accumulated ridge-silhouette map (per-pixel count of instance boundaries).

        Thresholds are tried high→low and descended **adaptively**: stop as soon as one
        threshold yields ``min_instances`` large instances. Distant multi-layer scenes are
        already rich at the high threshold (stop there → clean, precise); close-up single
        faces return only 1-2 huge instances there and so descend to a lower threshold that
        splits the face into sub-peaks (→ recall), without adding clutter where it isn't
        needed.
        """
        from scipy.ndimage import binary_erosion  # noqa: PLC0415
        from skimage.segmentation import find_boundaries  # noqa: PLC0415

        inner = binary_erosion(mountain, iterations=2)
        acc = np.zeros(mountain.shape, float)
        for th in thresholds:
            big = [m for m in self.instance_masks(rgb, th) if int(m.sum()) >= min_area]
            for m in big:
                acc += (find_boundaries(m, mode="inner") & inner).astype(float)
            if len(big) >= min_instances:
                break
        return acc

    def terrain_layers(
        self,
        rgb: NDArray[np.float64],
        depth: NDArray[np.float64] | None = None,
        threshold: float = 0.3,
        min_area: int = 1500,
        max_layers: int = 6,
    ) -> list[NDArray[np.bool_]]:
        """Instance masks ordered front (nearest) → back. ``depth`` (Depth-Anything: larger =
        nearer) orders them; if omitted, lower image rows = farther → back."""
        masks = [m for m in self.instance_masks(rgb, threshold) if int(m.sum()) >= min_area]
        if not masks:
            return []
        key = []
        for m in masks:
            if depth is not None:
                key.append(float(np.median(depth[m])) if m.any() else -np.inf)
            else:
                ys, _ = np.where(m)
                key.append(float(np.median(ys)) if len(ys) else -np.inf)
        order = np.argsort(key)[::-1]  # nearest (largest depth / lowest row) first
        return [masks[i] for i in order[:max_layers]]


def load_segmenter(kind: str = "sam3", **kwargs) -> Segmenter | None:
    """Factory: return a ready `Segmenter` of the given kind, or None if unavailable.

    kind: "sam3" (text-prompted `facebook/sam3`, default/best) | "mobile_sam" / "sam2"
    (point-prompted via ultralytics).
    """
    kind = kind.lower()
    try:
        if kind in ("sam3", "sam-3"):
            from peakle.sam3_seg import Sam3Segmenter  # noqa: PLC0415

            return Sam3Segmenter(**kwargs)
        if kind in ("mobile_sam", "mobilesam", "sam2", "sam2.1", "ultralytics"):
            from peakle.sam_point_seg import UltralyticsSamSegmenter  # noqa: PLC0415

            return UltralyticsSamSegmenter(kind=kind, **kwargs)
    except Exception:  # noqa: BLE001 - missing deps / gated weights / OOM → caller falls back
        return None
    raise ValueError(f"unknown segmenter kind: {kind!r}")
