"""Pose prior and estimate models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from peakle.domain.camera import CameraExtrinsics
from peakle.domain.coordinates import LocalPoint


class PosePrior(BaseModel):
    """Noisy pose prior for bounded camera optimization.

    Attributes:
        position: Noisy camera position.
        yaw_deg: Noisy yaw estimate.
        pitch_deg: Noisy pitch estimate.
        horizontal_sigma_m: Horizontal prior standard deviation in meters.
        vertical_sigma_m: Vertical prior standard deviation in meters.
        yaw_sigma_deg: Yaw prior standard deviation in degrees.
        pitch_sigma_deg: Pitch prior standard deviation in degrees.
    """

    position: LocalPoint
    yaw_deg: float
    pitch_deg: float
    horizontal_sigma_m: float = Field(gt=0.0)
    vertical_sigma_m: float = Field(gt=0.0)
    yaw_sigma_deg: float = Field(gt=0.0)
    pitch_sigma_deg: float = Field(gt=0.0)


class FitMetrics(BaseModel):
    """Pose optimization diagnostics.

    Attributes:
        score: Final scalar objective value.
        contour_mae_px: Mean absolute contour residual in pixels.
        contour_p95_px: 95th percentile contour residual in pixels.
        valid_columns: Number of columns used for residual scoring.
        iterations: Number of local optimizer iterations.
        success: Whether the local optimizer reported success.
        message: Optimizer status message.
        confidence: How well the predicted skyline reproduces the observed one,
            in [0, 1] (low = the fit is ambiguous / untrustworthy).
        position_error_m: Optional true position error for synthetic demos.
        yaw_error_deg: Optional true yaw error for synthetic demos.
        pitch_error_deg: Optional true pitch error for synthetic demos.
    """

    score: float
    contour_mae_px: float
    contour_p95_px: float
    valid_columns: int
    iterations: int
    success: bool
    message: str
    position_error_m: float | None
    yaw_error_deg: float | None
    pitch_error_deg: float | None
    confidence: float | None = None


class PoseEstimate(BaseModel):
    """Estimated camera pose and fit diagnostics."""

    extrinsics: CameraExtrinsics
    metrics: FitMetrics
