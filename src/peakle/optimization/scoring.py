"""Scoring helpers for pose optimization."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import huber


def robust_contour_residuals(
    predicted_profile: NDArray[np.float64],
    observed_profile: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Computes residuals for columns valid in both profiles."""

    valid = np.isfinite(predicted_profile) & np.isfinite(observed_profile)
    return predicted_profile[valid] - observed_profile[valid]


def huber_mean(residuals: NDArray[np.float64], delta: float = 8.0) -> float:
    """Computes a mean Huber loss.

    Args:
        residuals: Residual vector.
        delta: Huber transition point.

    Returns:
        Mean robust loss.
    """

    if residuals.size == 0:
        return 1_000_000.0
    return float(np.mean(huber(delta, residuals)))


MATCH_THRESHOLD_FRAC = 0.035
MIN_MATCH_THRESHOLD_PX = 8.0


def match_confidence(
    p95_px: float,
    coverage: float,
    image_height: int,
    threshold_frac: float = MATCH_THRESHOLD_FRAC,
) -> tuple[float, bool]:
    """Confidence in [0, 1] and a match decision from a skyline residual.

    Args:
        p95_px: 95th-percentile absolute residual between observed and predicted.
        coverage: Fraction of observed columns the prediction also covers.
        image_height: Image height in pixels (sets the tolerance scale).
        threshold_frac: Match tolerance as a fraction of image height.

    Returns:
        Tuple `(confidence, is_match)`.
    """

    threshold_px = max(threshold_frac * image_height, MIN_MATCH_THRESHOLD_PX)
    confidence = float(np.clip(1.0 - p95_px / (2.0 * threshold_px), 0.0, 1.0))
    is_match = bool(p95_px <= threshold_px and coverage >= 0.6)
    return confidence, is_match


def residual_summary(residuals: NDArray[np.float64]) -> tuple[float, float, int]:
    """Summarizes contour residuals.

    Args:
        residuals: Residual vector in pixels.

    Returns:
        Tuple `(mae, p95, count)`.
    """

    if residuals.size == 0:
        return 1_000_000.0, 1_000_000.0, 0
    absolute = np.abs(residuals)
    return float(np.mean(absolute)), float(np.percentile(absolute, 95.0)), int(residuals.size)
