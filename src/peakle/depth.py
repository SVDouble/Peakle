"""Monocular depth estimation for photos (pluggable source).

Depth turns the ambiguous outline into a 2.5D surface and, crucially, lets us
extract ridge separation lines the same way the renderer does (as depth-drop
crests) — which suppresses foreground rock texture that pollutes edge-based
ridges. The default `HazeDepth` is classical (atmospheric perspective: distant
terrain is washed toward the airlight). A learned model (Depth-Anything / MiDaS
via `transformers`) is a drop-in upgrade when installed.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, minimum_filter


class DepthEstimator(Protocol):
    """Estimates a dense relative depth map (higher = farther) in [0, 1]."""

    name: str

    def estimate(self, rgb: NDArray[np.float64]) -> NDArray[np.float64]: ...


class HazeDepth:
    """Classical depth from atmospheric haze (dark channel + desaturation)."""

    name = "haze"

    def __init__(self, window: int = 7) -> None:
        self.window = window

    def estimate(self, rgb: NDArray[np.float64]) -> NDArray[np.float64]:
        # Dark channel: distant hazy terrain is washed toward the bright airlight,
        # so its per-pixel minimum channel is high; near dark rock is low.
        dark = minimum_filter(rgb.min(axis=2), size=self.window)
        maximum = rgb.max(axis=2)
        minimum = rgb.min(axis=2)
        saturation = np.where(maximum > 0.0, (maximum - minimum) / np.maximum(maximum, 1e-6), 0.0)
        depth = gaussian_filter(0.5 * _normalize(dark) + 0.5 * (1.0 - saturation), 2.0)
        return _normalize(depth)


class LearnedDepth:
    """Monocular depth via a `transformers` depth-estimation pipeline (if installed)."""

    name = "learned"

    def __init__(self, model: str = "depth-anything/Depth-Anything-V2-Small-hf") -> None:
        from transformers import pipeline  # noqa: PLC0415 - optional heavy dependency

        self._pipe = pipeline("depth-estimation", model=model)

    def estimate(self, rgb: NDArray[np.float64]) -> NDArray[np.float64]:
        from PIL import Image  # noqa: PLC0415

        image = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")
        predicted = np.asarray(self._pipe(image)["depth"], dtype=np.float64)
        # Model depth is near=large; invert to our convention (far = large).
        return _normalize(predicted.max() - predicted)


def load_learned_depth() -> DepthEstimator | None:
    """Returns a learned estimator if `transformers`+`torch` are installed, else None."""

    try:
        import torch  # noqa: F401, PLC0415
        import transformers  # noqa: F401, PLC0415
    except ImportError:
        return None
    return LearnedDepth()


def estimate_depth(rgb: NDArray[np.float64], estimator: DepthEstimator | None = None) -> NDArray[np.float64]:
    """Estimates a relative depth map, defaulting to the classical haze estimator."""

    return (estimator or HazeDepth()).estimate(rgb)


def _normalize(values: NDArray[np.float64]) -> NDArray[np.float64]:
    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.0))
    if high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)
