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
