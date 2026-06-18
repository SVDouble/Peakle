"""Skyline profile and contour helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from peakle.domain.contours import ImagePoint, SkylineContour


def contour_from_profile(
    profile: NDArray[np.float64],
    image_height_px: int,
    source: str | None = None,
) -> SkylineContour:
    """Builds a skyline contour from a dense profile.

    Args:
        profile: One y-coordinate per image column.
        image_height_px: Source image height in pixels.
        source: Optional source artifact name.

    Returns:
        Ordered skyline contour.
    """

    points = [
        ImagePoint(x_px=float(column), y_px=float(y_px)) for column, y_px in enumerate(profile) if np.isfinite(y_px)
    ]
    return SkylineContour(
        image_width_px=int(profile.size),
        image_height_px=image_height_px,
        points=points,
        source=source,
    )


def extract_skyline_from_mask(
    mask: NDArray[np.bool_] | NDArray[np.uint8],
    smoothing_sigma: float = 1.0,
    source: str | None = None,
) -> SkylineContour:
    """Extracts the top terrain pixel from each image column.

    Args:
        mask: Boolean or integer mask where terrain pixels are true/non-zero.
        smoothing_sigma: Gaussian smoothing sigma in image columns.
        source: Optional source artifact name.

    Returns:
        Ordered skyline contour extracted from the mask.
    """

    terrain_mask = mask.astype(bool)
    height, width = terrain_mask.shape
    profile = np.full(width, np.nan, dtype=np.float64)
    occupied_columns = np.any(terrain_mask, axis=0)
    if np.any(occupied_columns):
        profile[occupied_columns] = np.argmax(terrain_mask[:, occupied_columns], axis=0)
    profile = interpolate_profile(profile, fallback=float(height - 1))
    if smoothing_sigma > 0.0:
        profile = gaussian_filter1d(profile, sigma=smoothing_sigma, mode="nearest")
    profile = np.clip(profile, 0.0, float(height - 1))
    return contour_from_profile(profile, height, source=source)


def interpolate_profile(
    profile: NDArray[np.float64],
    fallback: float,
) -> NDArray[np.float64]:
    """Interpolates missing profile columns.

    Args:
        profile: Dense profile with NaN for missing columns.
        fallback: Value used when there are fewer than two valid columns.

    Returns:
        Profile with all columns filled.
    """

    columns = np.arange(profile.size)
    valid = np.isfinite(profile)
    if np.count_nonzero(valid) == 0:
        return np.full_like(profile, fallback, dtype=np.float64)
    if np.count_nonzero(valid) == 1:
        return np.full_like(profile, float(profile[valid][0]), dtype=np.float64)
    filled = np.interp(columns, columns[valid], profile[valid])
    return filled.astype(np.float64)
