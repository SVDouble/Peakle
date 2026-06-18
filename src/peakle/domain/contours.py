"""Image contour domain models."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field, model_validator


class ImagePoint(BaseModel):
    """Image-space point in pixels.

    Attributes:
        x_px: Horizontal pixel coordinate.
        y_px: Vertical pixel coordinate.
    """

    x_px: float
    y_px: float


class SkylineContour(BaseModel):
    """Ordered skyline contour sampled in image coordinates.

    Attributes:
        image_width_px: Source image width in pixels.
        image_height_px: Source image height in pixels.
        points: Contour points ordered by increasing x-coordinate.
        source: Optional source artifact name.
    """

    image_width_px: int = Field(ge=1)
    image_height_px: int = Field(ge=1)
    points: list[ImagePoint]
    source: str | None

    @model_validator(mode="after")
    def _validate_points(self) -> SkylineContour:
        previous_x = -float("inf")
        for point in self.points:
            if point.x_px < previous_x:
                msg = "contour points must be ordered by x_px"
                raise ValueError(msg)
            previous_x = point.x_px
        return self

    def to_profile(self) -> np.ndarray:
        """Converts the contour to a dense vertical profile.

        Returns:
            Array of length `image_width_px` containing y-coordinates or NaN for
            missing columns.
        """

        profile = np.full(self.image_width_px, np.nan, dtype=np.float64)
        for point in self.points:
            column = int(round(point.x_px))
            if 0 <= column < self.image_width_px:
                profile[column] = point.y_px
        return profile
