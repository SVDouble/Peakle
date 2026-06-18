"""Camera models."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, model_validator

from peakle.domain.coordinates import LocalPoint


class CameraIntrinsics(BaseModel):
    """Pinhole camera intrinsics.

    Attributes:
        width_px: Image width in pixels.
        height_px: Image height in pixels.
        focal_length_px: Focal length in pixels.
        principal_x_px: Principal point x-coordinate in pixels.
        principal_y_px: Principal point y-coordinate in pixels.
    """

    width_px: int = Field(ge=1)
    height_px: int = Field(ge=1)
    focal_length_px: float = Field(gt=0.0)
    principal_x_px: float
    principal_y_px: float

    @classmethod
    def from_horizontal_fov(
        cls,
        width_px: int,
        height_px: int,
        horizontal_fov_deg: float,
    ) -> CameraIntrinsics:
        """Builds intrinsics from a horizontal field of view.

        Args:
            width_px: Image width in pixels.
            height_px: Image height in pixels.
            horizontal_fov_deg: Horizontal field of view in degrees.

        Returns:
            Camera intrinsics with square pixels.
        """

        if not 1.0 < horizontal_fov_deg < 179.0:
            msg = "horizontal_fov_deg must be between 1 and 179 degrees"
            raise ValueError(msg)
        focal_length_px = width_px / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
        return cls(
            width_px=width_px,
            height_px=height_px,
            focal_length_px=focal_length_px,
            principal_x_px=(width_px - 1) / 2.0,
            principal_y_px=(height_px - 1) / 2.0,
        )

    @model_validator(mode="after")
    def _validate_principal_point(self) -> CameraIntrinsics:
        if not 0.0 <= self.principal_x_px <= self.width_px - 1:
            msg = "principal_x_px must be inside the image"
            raise ValueError(msg)
        if not 0.0 <= self.principal_y_px <= self.height_px - 1:
            msg = "principal_y_px must be inside the image"
            raise ValueError(msg)
        return self


class CameraExtrinsics(BaseModel):
    """Pinhole camera extrinsics in the local terrain frame.

    Attributes:
        position: Camera position in local meters.
        yaw_deg: Bearing in degrees, where 0 looks north and 90 looks east.
        pitch_deg: Tilt in degrees, positive looking upward.
        roll_deg: Image-plane roll in degrees.
    """

    position: LocalPoint
    yaw_deg: float = Field(ge=-360.0, le=360.0)
    pitch_deg: float = Field(ge=-89.0, le=89.0)
    roll_deg: float = Field(ge=-180.0, le=180.0)
