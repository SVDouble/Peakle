"""Peak domain models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from peakle.domain.coordinates import GeoPoint, LocalPoint


class PeakDetectionSpec(BaseModel):
    """Parameters for synthetic local-maximum peak detection.

    Attributes:
        neighborhood_px: Odd local maximum window size in grid cells.
        prominence_radius_px: Radius used for synthetic prominence scoring.
        min_prominence_m: Minimum synthetic prominence in meters.
        max_peaks: Maximum number of peaks to keep.
    """

    neighborhood_px: int = Field(ge=3)
    prominence_radius_px: int = Field(ge=3)
    min_prominence_m: float = Field(ge=0.0)
    max_peaks: int = Field(ge=1)


class Peak(BaseModel):
    """Detected or generated mountain peak.

    Attributes:
        id: Stable peak identifier.
        name: Human-readable synthetic peak name.
        local_position: Peak position in local meters.
        geo_position: Peak position in geodetic coordinates.
        elevation_m: Elevation in meters.
        prominence_m: Synthetic prominence in meters.
    """

    id: str
    name: str
    local_position: LocalPoint
    geo_position: GeoPoint
    elevation_m: float
    prominence_m: float
