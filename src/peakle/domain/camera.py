"""Camera models."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, model_validator

from peakle.domain.coordinates import LocalPoint
from peakle.domain.projection import (
    ImageProjectionName,
    azimuths_deg,
    focal_length_px,
    pitch_deg_from_vertical_shift_px,
    rows_from_elevation_rad,
    vertical_fov_deg,
)

ProjectionKind = ImageProjectionName


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

    def horizontal_fov_deg(self) -> float:
        """Returns the horizontal field of view in degrees."""

        return math.degrees(2.0 * math.atan(self.width_px / (2.0 * self.focal_length_px)))

    @model_validator(mode="after")
    def _validate_principal_point(self) -> CameraIntrinsics:
        if not 0.0 <= self.principal_x_px <= self.width_px - 1:
            msg = "principal_x_px must be inside the image"
            raise ValueError(msg)
        if not 0.0 <= self.principal_y_px <= self.height_px - 1:
            msg = "principal_y_px must be inside the image"
            raise ValueError(msg)
        return self


class CameraModel(BaseModel):
    """Camera model of the view being localized.

    ``CameraIntrinsics`` remains the pinhole model used by the DEM rasterizer. This model describes
    the image coordinate system itself, including GeoPose3K's true cylindrical/tan crop geometry.
    """

    width_px: int = Field(ge=1)
    height_px: int = Field(ge=1)
    horizontal_fov_deg: float = Field(gt=1.0, lt=179.0)
    projection: ProjectionKind = "pinhole"

    @classmethod
    def from_intrinsics(cls, intrinsics: CameraIntrinsics, projection: ProjectionKind = "pinhole") -> CameraModel:
        """Builds a camera model from existing pinhole intrinsics."""

        return cls(
            width_px=intrinsics.width_px,
            height_px=intrinsics.height_px,
            horizontal_fov_deg=intrinsics.horizontal_fov_deg(),
            projection=projection,
        )

    def focal_length_px(self) -> float:
        """Returns the image-plane focal length used by this projection."""

        return focal_length_px(self.width_px, self.horizontal_fov_deg, self.projection)

    def vertical_fov_deg(self) -> float:
        """Returns the perspective-camera vertical FOV equivalent for this view."""

        return vertical_fov_deg(self.width_px, self.height_px, self.horizontal_fov_deg, self.projection)

    def pitch_deg_from_vertical_shift_px(self, shift_px: float) -> float:
        """Returns the pitch represented by a vertical image shift."""

        return pitch_deg_from_vertical_shift_px(self.width_px, self.horizontal_fov_deg, self.projection, shift_px)

    def azimuths_deg(self, yaw_deg: float):
        """Returns the world azimuth sampled by each image column."""

        return azimuths_deg(self.width_px, self.horizontal_fov_deg, yaw_deg, self.projection)

    def rows_from_elevation_rad(self, elevation_rad, *, pitch_deg: float = 0.0):
        """Projects terrain elevation angles to this view's image rows."""

        return rows_from_elevation_rad(
            elevation_rad,
            self.width_px,
            self.height_px,
            self.horizontal_fov_deg,
            self.projection,
            pitch_deg=pitch_deg,
        )


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
